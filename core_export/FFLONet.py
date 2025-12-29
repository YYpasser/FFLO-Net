import torch
import torch.nn as nn
import torch.nn.functional as F
from core_export.extractor import FeatureExtractor, ContextExtractor
from core_export.fusion import CostFusion, hourglass
from core_export.geometry import Combined_Geo_Encoding_Volume
from core_export.update import LSTMMultiUpdateBlock
from core_export.submodule import ConvBlock, Conv2x, Conv2x_IN
from core_export.utils.utils import build_gwc_volume, disparity_regression, context_upsample
import torch.amp
from typing import Union, List

class FFLONet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.autocast = torch.amp.autocast
        self.feature_net = FeatureExtractor()
        self.stem_02 = nn.Sequential(
            ConvBlock(3, 32, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(32, 32, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.stem_04 = nn.Sequential(
            ConvBlock(32, 48, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(48, 48, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.feat02_proj = nn.Sequential(
            ConvBlock(64, 32, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            ConvBlock(32, 32, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(32, 32, 1, 1, 0))
        self.feat04_proj = nn.Sequential(
            ConvBlock(96, 96, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(96, 96, 1, 1, 0))
        
        self.cost_fusion = CostFusion()
        self.hourglass = hourglass(8)
        self.classifier1 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            nn.Conv3d(8, 1, 3, 1, 1, bias=False))
        self.classifier2 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            nn.Conv3d(8, 1, 3, 1, 1, bias=False))
        
        self.context_net = ContextExtractor()
        self.update_block = LSTMMultiUpdateBlock()
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(128, 128*4, 3, padding=3//2) for _ in range(3)])       
        self.spx = nn.ConvTranspose2d(32*2, 9, 4, 2, 1)
        self.spx_2 = Conv2x_IN(32, 32)
        self.spx_4 = nn.Sequential(
            ConvBlock(96, 32, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(32, 32, 3, 1, 1, norm_fn='in', activation_fn='leaky'))
        self.spx_2_lstm = Conv2x(32, 32)
        self.spx_lstm = nn.ConvTranspose2d(2*32, 9, 4, 2, 1)
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()   

    def freeze_bn3d(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def freeze_backbone(self):
        self.feature_net.freeze_backbone()
        self.context_net.freeze_backbone()

    def unfreeze_backbone(self):
        self.feature_net.unfreeze_backbone()
        self.context_net.unfreeze_backbone()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_lstm(mask_feat_4, stem_2x)
        spx_pred = self.spx_lstm(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred, 4).unsqueeze(1)
        return up_disp

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]] :
        """ Estimate disparity between pair of frames """
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        features_left = self.feature_net(image1)
        features_right = self.feature_net(image2)

        stem02_left  = self.stem_02(image1)
        stem04_left  = self.stem_04(stem02_left)
        stem02_right = self.stem_02(image2)
        stem04_right = self.stem_04(stem02_right)           
        stem02_left  = torch.cat((features_left[0], stem02_left), 1)
        stem02_left  = self.feat02_proj(stem02_left)
        
        features_left[1]  = torch.cat((features_left[1], stem04_left), 1)
        features_right[1] = torch.cat((features_right[1], stem04_right), 1)
        features_left[1]  = self.feat04_proj(features_left[1])
        features_right[1] = self.feat04_proj(features_right[1])

        cost04 = build_gwc_volume(features_left[1], features_right[1], self.args.max_disp//4, 8)
        cost08 = build_gwc_volume(features_left[2], features_right[2], self.args.max_disp//8, 8)
        cost16 = build_gwc_volume(features_left[3], features_right[3], self.args.max_disp//16, 8)
        
        cost04 = self.cost_fusion(cost04, cost08, cost16)
        del cost08, cost16
          
        cost04 = self.hourglass(cost04, features_left)

        prob = self.classifier2(cost04)
        prob = torch.squeeze(prob, 1)
        prob = F.softmax(prob, dim=1)
        disp_x04_init = disparity_regression(prob, self.args.max_disp//4)

        cnet_list = self.context_net(image1)
        net_h = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//4, dim=1)) for i,
                    conv in zip(inp_list, self.context_zqr_convs)] 

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(features_left[1].float(), features_right[1].float(), cost04.float())
        b, _, h, w = features_left[1].shape
        coords = torch.arange(w).float().to(features_left[1].device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = disp_x04_init

        del prob, cost04

        netC = net_h
        for _ in range(self.args.valid_iters):
            geo_feat = geo_fn(disp, coords)
            netC, net_h, mask_feat_4, delta_disp = self.update_block(netC, net_h, inp_list, geo_feat, disp, iter16=True, iter08=True, iter04=True, update=True)
            disp = disp + delta_disp

        disp_up = self.upsample_disp(disp, mask_feat_4, stem02_left)

        return disp_up
