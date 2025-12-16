import torch
import torch.nn as nn
import torch.nn.functional as F
from core_depthany.extractor import FeatureExtractor, ContextExtractor
from core_depthany.fusion import CostFusion, hourglass
from core_depthany.geometry import Combined_Geo_Encoding_Volume
from core_depthany.update import LSTMMultiUpdateBlock
from core_depthany.submodule import ConvBlock, Conv2x, Conv2x_IN
from core_depthany.utils.utils import build_gwc_volume, disparity_regression, context_upsample
from core_depthany.depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder
from typing import Union, List

class FFLONet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(f'./pretrained_models/dpt/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)
        del depth_anything, state_dict_dpt, depth_anything_decoder

        self.feature_net = FeatureExtractor(dim_list)

        self.stem_02 = nn.Sequential(
            ConvBlock(  3,  32, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock( 32,  32, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.stem_04 = nn.Sequential(
            ConvBlock( 32,  48, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock( 48,  48, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.stem_08 = nn.Sequential(
            ConvBlock( 48,  96, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock( 96,  96, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        self.stem_16 = nn.Sequential(
            ConvBlock( 96, 192, 3, 2, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(192, 192, 3, 1, 1, norm_fn='in', activation_fn='relu'))
        
        self.feat04_proj = nn.Sequential(
            ConvBlock(96, 96, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(96, 96, 1, 1, 0))
        self.feat08_proj = nn.Sequential(
            ConvBlock(96+64, 96+64, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(96+64, 96+64, 1, 1, 0))
        self.feat16_proj = nn.Sequential(
            ConvBlock(192+192, 192+192, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(192+192, 192+192, 1, 1, 0))
        
        self.cost_fusion = CostFusion()
        self.hourglass = hourglass(8)
        self.classifier1 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            nn.Conv3d(8, 1, 3, 1, 1, bias=False))
        self.classifier2 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            nn.Conv3d(8, 1, 3, 1, 1, bias=False))
        
        self.context_net = ContextExtractor(dim_list, 128)
        self.update_block = LSTMMultiUpdateBlock()
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(128, 128*4, 3, padding=3//2) for _ in range(3)])       
        self.spx = nn.ConvTranspose2d(32*2, 9, 4, 2, 1)
        self.spx_2 = Conv2x_IN(24, 32)
        self.spx_4 = nn.Sequential(
            ConvBlock(96, 24, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(24, 24, 3, 1, 1, norm_fn='in', activation_fn='leaky'))
        self.spx_2_lstm = Conv2x(32, 32)
        self.spx_lstm = nn.ConvTranspose2d(2*32, 9, 4, 2, 1)
        
    def infer_mono(self, image1: torch.Tensor, image2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()   

    def upsample_disp(self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor) -> torch.Tensor:
        xspx = self.spx_2_lstm(mask_feat_4, stem_2x)
        spx_pred = self.spx_lstm(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred, 4).unsqueeze(1)
        return up_disp

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int=12, test_mode: bool=False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]] :
        """ Estimate disparity between pair of frames """
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # Depth Anything V2 Backbone Feature Extraction
        features_mono_left, features_mono_right = self.infer_mono(image1, image2)

        features_left  = self.feature_net(features_mono_left)
        features_right = self.feature_net(features_mono_right)

        stem02_left  = self.stem_02(image1)
        stem04_left  = self.stem_04(stem02_left)
        stem08_left  = self.stem_08(stem04_left)
        stem16_left  = self.stem_16(stem08_left)
        stem02_right = self.stem_02(image2)
        stem04_right = self.stem_04(stem02_right)
        stem08_right = self.stem_08(stem04_right)
        stem16_right = self.stem_16(stem08_right)
        
        stem_left_list = [stem16_left, stem08_left, stem04_left]

        features_left[0] = torch.cat((features_left[0], stem04_left), 1)
        features_right[0] = torch.cat((features_right[0], stem04_right), 1)
        features_left[1] = torch.cat((features_left[1], stem08_left), 1)
        features_right[1] = torch.cat((features_right[1], stem08_right), 1)
        features_left[2] = torch.cat((features_left[2], stem16_left), 1)
        features_right[2] = torch.cat((features_right[2], stem16_right), 1)
        
        match_left = []
        match_left.append(self.feat04_proj(features_left[0]))        
        match_left.append(self.feat08_proj(features_left[1]))        
        match_left.append(self.feat16_proj(features_left[2]))       
        match_right = []
        match_right.append(self.feat04_proj(features_right[0]))
        match_right.append(self.feat08_proj(features_right[1]))
        match_right.append(self.feat16_proj(features_right[2]))
        
        cost04 = build_gwc_volume(match_left[0], match_right[0], self.args.max_disp//4, 8)
        cost08 = build_gwc_volume(match_left[1], match_right[1], self.args.max_disp//8, 8)
        cost16 = build_gwc_volume(match_left[2], match_right[2], self.args.max_disp//16, 8)
        
        cost04 = self.cost_fusion(cost04, cost08, cost16)
        del cost08, cost16

        if not test_mode:
            prob = self.classifier1(cost04)
            prob = torch.squeeze(prob, 1)
            prob = F.softmax(prob, dim=1)
            disp_x04_fuse = disparity_regression(prob, self.args.max_disp//4)            

        cost04 = self.hourglass(cost04, features_left)

        prob = self.classifier2(cost04)
        prob = torch.squeeze(prob, 1)
        prob = F.softmax(prob, dim=1)
        disp_x04_init = disparity_regression(prob, self.args.max_disp//4)
        
        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem02_left)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        cnet_list = self.context_net(features_mono_left, stem_left_list)
        net_h = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//4, dim=1)) for i,
                    conv in zip(inp_list, self.context_zqr_convs)] 

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left[0].float(), match_right[0].float(), cost04.float())
        b, _, h, w = match_left[0].shape
        coords = torch.arange(w).float().to(features_left[1].device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = disp_x04_init
        disp_preds = []

        del prob, cost04

        net_c = net_h
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)          
            
            net_c, net_h, mask_feat_4, delta_disp = self.update_block(net_c, net_h, inp_list, geo_feat, disp, iter16=True, iter08=True, iter04=True, update=True)

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