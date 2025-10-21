import torch
import torch.nn as nn
import torch.nn.functional as F
from core_dpt.fusion import CostFusion, hourglass
from core_dpt.geometry import Combined_Geo_Encoding_Volume
from core_dpt.update import LSTMMultiUpdateBlock
from core_dpt.submodule import ConvBlock, Conv2x, Conv2x_IN
from core_dpt.utils.utils import build_gwc_volume, disparity_regression, context_upsample
import sys
sys.path.append('depth_anything_v2')
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder
import torch.amp
from typing import Union, List

class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list



class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = ConvBlock(int(48+dim_list[0]), 48, 5, 1, 2, norm_fn='in', activation_fn='relu')
        self.conv8x = ConvBlock(int(64+dim_list[0]), 64, 5, 1, 2, norm_fn='in', activation_fn='relu')
        self.conv16x = ConvBlock(int(192+dim_list[0]), 192, 5, 1, 2, norm_fn='in', activation_fn='relu')
        self.conv32x = ConvBlock(int(dim_list[0]), 160, 3, 1, 1, norm_fn='in', activation_fn='relu')

        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)

    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list


class FFLONet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.autocast = torch.amp.autocast

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


        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, 128)

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
        
        self.cost_fusion = CostFusion()
        self.hourglass = hourglass(8)
        self.classifier = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            nn.Conv3d(8, 1, 3, 1, 1, bias=False))

        self.update_block = LSTMMultiUpdateBlock()
        self.context_ifco_convs = nn.ModuleList([nn.Conv2d(128, 128*4, 3, padding=3//2) for _ in range(3)])       
        self.spx = nn.ConvTranspose2d(32*2, 9, 4, 2, 1)
        self.spx_2 = Conv2x_IN(24, 32)
        self.spx_4 = nn.Sequential(
            ConvBlock(96, 24, 3, 1, 1, norm_fn='in', activation_fn='leaky'),
            ConvBlock(24, 24, 3, 1, 1, norm_fn='in', activation_fn='leaky'))
        self.spx_2_lstm = Conv2x(32, 32)
        self.spx_lstm = nn.ConvTranspose2d(2*32, 9, 4, 2, 1)

    def infer_mono(self, image1, image2):
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
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
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
        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            features_mono_left, features_mono_right = self.infer_mono(image1, image2)

        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)

        stem02_left  = self.stem_02(image1)
        stem04_left  = self.stem_04(stem02_left)
        stem08_left  = self.stem_08(stem04_left)
        stem16_left  = self.stem_16(stem08_left)
        stem02_right = self.stem_02(image2)
        stem04_right = self.stem_04(stem02_right)

        stem_left_list = [stem16_left, stem08_left, stem04_left]
        features_left[0] = torch.cat((features_left[0], stem04_left), 1)
        features_right[0] = torch.cat((features_right[0], stem04_right), 1)
        
        match_left  = self.feat04_proj(features_left[0])
        match_right = self.feat04_proj(features_right[0])

        cost04 = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
        cost08 = build_gwc_volume(features_left[1], features_right[1], self.args.max_disp//8, 8)
        cost16 = build_gwc_volume(features_left[2], features_right[2], self.args.max_disp//16, 8)
        
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
            xspx = self.spx_4(features_left[0]) # 1/4尺度提取 超像素? superpixel 信息
            xspx = self.spx_2(xspx, stem02_left) # xspx最近邻插值上采样到 1/2 尺度, 融合stem02_left
            spx_pred = self.spx(xspx) # xspx转置卷积上采样到 全分辨率 尺度
            spx_pred = F.softmax(spx_pred, 1) # 超像素概率分布

        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_left_list)
        net_h = [torch.tanh(x[0]) for x in cnet_list]    # ConvLSTMs隐藏状态(tanh激活)
        inp_list = [torch.relu(x[1]) for x in cnet_list] # ConvLSTMs输入状态(relu激活)
        inp_list = [list(conv(i).split(split_size=conv.out_channels//4, dim=1)) for i,
                    conv in zip(inp_list, self.context_ifco_convs)] 
        # 拆分为 输入门bi, 遗忘门bf, 细胞状态bc, 输出门bo

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), cost04.float())
        b, _, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = disp_x04_init
        disp_preds = []

        del prob, cost04

        netC = net_h
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)          
            with self.autocast('cuda', enabled=self.args.mixed_precision):
                netC, net_h, mask_feat_4, delta_disp = self.update_block(netC, net_h, inp_list, geo_feat, disp, iter16=True, iter08=True, iter04=True, update=True)

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