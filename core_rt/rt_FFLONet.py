import torch
import torch.nn as nn
import torch.nn.functional as F
from core_rt.extractor import FeatureExtractor
from core_rt.fusion import CostFusion, hourglass
from core_rt.geometry import Combined_Geo_Encoding_Volume
from core_rt.update import LSTMMultiUpdateBlock
from core_rt.submodule import ConvBlock, Conv2x, Conv2x_IN
from core_rt.utils.utils import build_gwc_volume, disparity_regression, context_upsample
import torch.amp
from typing import Union, List

class FFLONet(nn.Module):
    def __init__(self, args):
        super(FFLONet, self).__init__()
        self.args = args
        self.autocast = torch.amp.autocast
        self.feature_net = FeatureExtractor()
        self.stem_02 = nn.Sequential(
            ConvBlock(3, 16, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(16, 16, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.stem_04 = nn.Sequential(
            ConvBlock(16, 24, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(24, 24, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.feat02_proj = nn.Sequential(
            ConvBlock(32, 32, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(32, 32, 1, 1, 0))
        self.feat04_proj = nn.Sequential(
            ConvBlock(48, 48, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(48, 48, 1, 1, 0))
        
        self.cost_fusion = CostFusion()
        self.hourglass = hourglass(8)
        self.classifier = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            nn.Conv3d(8, 1, 3, 1, 1, bias=False))
        
        # 移除上下文网络, 缩小隐藏状态宽度
        self.hnet = nn.Sequential(
            ConvBlock(48, 48, 3, 1, 1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False))
        self.cnet = ConvBlock(48, 48, kernel_size=3, stride=1, padding=1)

        self.update_block = LSTMMultiUpdateBlock()
        self.context_ifco_convs = nn.Conv2d(48, 48*4, 3, padding=3//2)
        self.spx = nn.ConvTranspose2d(32*2, 9, 4, 2, 1)
        self.spx_2 = Conv2x_IN(32, 32)
        self.spx_4 = nn.Sequential(
            ConvBlock(48, 32, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(32, 32, 3, 1, 1, norm_fn='in', activation_fn='leaky'))
        self.spx_2_lstm = Conv2x(32, 32)
        self.spx_lstm = nn.ConvTranspose2d(2*32, 9, 4, 2, 1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        with self.autocast('cuda', enabled=self.args.mixed_precision):
            xspx = self.spx_2_lstm(mask_feat_4, stem_2x)
            spx_pred = self.spx_lstm(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred, 4).unsqueeze(1)
        return up_disp

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int=12, test_mode: bool=False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]] :
        """ Estimate disparity between pair of frames """
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with self.autocast('cuda', enabled=self.args.mixed_precision):

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

            if not test_mode:
                prob = self.classifier(cost04)
                prob = torch.squeeze(prob, 1)
                prob = F.softmax(prob, dim=1)
                disp_x04_fuse = disparity_regression(prob, self.args.max_disp//4)            

            cost04 = self.hourglass(cost04, features_left)

            prob = self.classifier(cost04)
            prob = torch.squeeze(prob, 1)
            prob = F.softmax(prob, dim=1)
            disp_x04_init = disparity_regression(prob, self.args.max_disp//4)
            
            # 超像素辅助? 区域一致性约束, 同一超像素内视差平滑?
            if not test_mode:
                # xspx = self.spx_2(features_left[0], stem02_left)
                xspx = self.spx_4(features_left[1]) # 1/4尺度提取 超像素? superpixel 信息
                xspx = self.spx_2(xspx, stem02_left) # xspx最近邻插值上采样到 1/2 尺度, 融合stem02_left
                spx_pred = self.spx(xspx) # xspx转置卷积上采样到 全分辨率 尺度
                spx_pred = F.softmax(spx_pred, 1) # 超像素概率分布

            hidden = self.hnet(features_left[1])
            net_h = torch.tanh(hidden)
            context = self.cnet(features_left[1])
            context = list(self.context_ifco_convs(context).split(split_size=48, dim=1))

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(features_left[1].float(), features_right[1].float(), cost04.float())
        b, _, h, w = features_left[1].shape
        coords = torch.arange(w).float().to(features_left[1].device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = disp_x04_init
        disp_preds = []

        del prob, cost04

        netC = net_h
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)          
            with self.autocast('cuda', enabled=self.args.mixed_precision):
                netC, net_h, mask_feat_4, delta_disp = self.update_block(netC, net_h, context, geo_feat, disp)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem02_left)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        disp_x04_fuse = context_upsample(disp_x04_fuse*4., spx_pred.float(), 4).unsqueeze(1)
        disp_x04_init = context_upsample(disp_x04_init*4., spx_pred.float(), 4).unsqueeze(1)

        return disp_x04_fuse, disp_x04_init, disp_preds    