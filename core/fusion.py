import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from core.submodule import ConvBlock

class FeatureFusion(nn.Module):
    
    def __init__(self, in_channels, out_channels, norm_fn='bn', activation_fn='leaky'):
        super().__init__()

        self.fusion = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm_fn=norm_fn, activation_fn=activation_fn),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_fn=norm_fn, activation_fn=activation_fn))
        
    def forward(self, x1, *x2):
        y = torch.cat(x2, 1)
        x = torch.cat([x1, y], 1)

        return self.fusion(x)
    

class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),)

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn    
        pattn2 = self.pa(initial, pattn1)
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    

class FeatureAtt(nn.Module):
    """## 引导代价体积激励 (CoEX - GCE)

    ### Args:
        - `cv_chan`: 代价体通道数
        - `feat_chan`: 特征通道数
    """
    def __init__(self, cv_chan, feat_chan):
        super().__init__()
        self.feat_att = nn.Sequential(
            ConvBlock(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, kernel_size=1, stride=1, padding=0))
        
    def forward(self, cost_volume, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cost_volume = torch.sigmoid(feat_att)*cost_volume
        return cost_volume



class hourglass(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.enlayer1 = nn.Sequential(
            ConvBlock(in_channels,   in_channels*2, 3, 2, 1, is_3d=True),
            ConvBlock(in_channels*2, in_channels*2, 3, 1, 1, is_3d=True))
        self.enlayer2 = nn.Sequential(
            ConvBlock(in_channels*2, in_channels*4, 3, 2, 1, is_3d=True),
            ConvBlock(in_channels*4, in_channels*4, 3, 1, 1, is_3d=True))                             
        self.enlayer3 = nn.Sequential(
            ConvBlock(in_channels*4, in_channels*6, 3, 2, 1, is_3d=True),
            ConvBlock(in_channels*6, in_channels*6, 3, 1, 1, is_3d=True))
        
        self.delayer1 = ConvBlock(in_channels*6, in_channels*4, (4, 4, 4), (2, 2, 2), (1, 1, 1), transpose_conv=True, is_3d=True)
        self.delayer2 = ConvBlock(in_channels*4, in_channels*2, (4, 4, 4), (2, 2, 2), (1, 1, 1), transpose_conv=True, is_3d=True)
        self.delayer3 = ConvBlock(in_channels*2, in_channels  , (4, 4, 4), (2, 2, 2), (1, 1, 1), transpose_conv=True, is_3d=True)
        
        self.agg_x16 = nn.Sequential(
            ConvBlock(in_channels*8, in_channels*4, 1, 1, 0, is_3d=True),
            ConvBlock(in_channels*4, in_channels*4, 3, 1, 1, is_3d=True),
            ConvBlock(in_channels*4, in_channels*4, 3, 1, 1, is_3d=True))
        self.agg_x08 = nn.Sequential(
            ConvBlock(in_channels*4, in_channels*2, 1, 1, 0, is_3d=True),
            ConvBlock(in_channels*2, in_channels*2, 3, 1, 1, is_3d=True),
            ConvBlock(in_channels*2, in_channels*2, 3, 1, 1, is_3d=True))
        
        self.feature_att_8     = FeatureAtt(in_channels*2,  80)
        self.feature_att_16    = FeatureAtt(in_channels*4, 224)
        self.feature_att_32    = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 224)
        self.feature_att_up_8  = FeatureAtt(in_channels*2,  80)

    def forward(self, x, features):
        x08 = self.enlayer1(x)
        x08 = self.feature_att_8(x08, features[2])
        x16 = self.enlayer2(x08)
        x16 = self.feature_att_16(x16, features[3])
        x32 = self.enlayer3(x16)
        x32 = self.feature_att_32(x32, features[4])
        x16up = self.delayer1(x32)
        x16 = torch.cat([x16up, x16], dim=1)
        x16 = self.agg_x16(x16)
        x16 = self.feature_att_up_16(x16, features[3])
        x08up = self.delayer2(x16)
        x08 = torch.cat([x08up, x08], dim=1)
        x08 = self.agg_x08(x08)
        x08 = self.feature_att_up_8(x08, features[2])
        x04up = self.delayer3(x08)
        return x04up
    
class HourglassFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.enlayer1   = nn.Conv3d(in_channels, in_channels, 3, 2, 1)
        self.combinex08 = nn.Sequential(
            ConvBlock(in_channels*2, in_channels, 1, 1, 0, is_3d=True),
            ConvBlock(in_channels, in_channels, 3, 1, 1, is_3d=True))
        self.enlayer3   = nn.Conv3d(in_channels, in_channels, 3, 2, 1)
        self.combinex16 = nn.Sequential(
            ConvBlock(in_channels*2, in_channels, 1, 1, 0, is_3d=True),
            ConvBlock(in_channels, in_channels, 3, 1, 1, is_3d=True))
        self.delayer2  = ConvBlock(in_channels, in_channels, 4, 2, 1, transpose_conv=True, is_3d=True)
        self.delayer3  = ConvBlock(in_channels, in_channels, 4, 2, 1, transpose_conv=True, is_3d=True)
        self.dirx04 = ConvBlock(in_channels, in_channels, 1, 1, 0, activation_fn='none', is_3d=True)
        self.dirx08 = ConvBlock(in_channels, in_channels, 1, 1, 0, activation_fn='none', is_3d=True)

    def forward(self, cost04, cost08, cost16):
        x08 = self.enlayer1(cost04)
        x08 = torch.cat([x08, cost08], dim=1)
        x08 = self.combinex08(x08)
        x16 = self.enlayer3(x08)
        x16 = torch.cat([x16, cost16], dim=1)
        x16 = self.combinex16(x16)
        x08 = F.leaky_relu(self.delayer2(x16) + self.dirx08(x08), inplace=True)
        x04 = F.leaky_relu(self.delayer3(x08) + self.dirx04(cost04), inplace=True)
        return x04


class CostFusion(nn.Module):
    def __init__(self):
        super(CostFusion, self).__init__()
        self.res1_x04 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            ConvBlock(8, 8, 3, 1, 1, is_3d=True))
        self.res2_x04 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            ConvBlock(8, 8, 3, 1, 1, activation_fn='none', is_3d=True))
        self.res1_x08 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            ConvBlock(8, 8, 3, 1, 1, is_3d=True))
        self.res2_x08 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            ConvBlock(8, 8, 3, 1, 1, activation_fn='none', is_3d=True))
        self.res1_x16 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            ConvBlock(8, 8, 3, 1, 1, is_3d=True))
        self.res2_x16 = nn.Sequential(
            ConvBlock(8, 8, 3, 1, 1, is_3d=True),
            ConvBlock(8, 8, 3, 1, 1, activation_fn='none', is_3d=True))
        self.fuse = HourglassFusion(8)

    def forward(self, cost04, cost08, cost16):
        x04 = self.res1_x04(cost04)
        x04 = self.res2_x04(x04) + x04
        x08 = self.res1_x08(cost08)
        x08 = self.res2_x08(x08) + x08
        x16 = self.res1_x16(cost16)
        x16 = self.res2_x16(x16) + x16
        x04 = self.fuse(x04, x08, x16)
        return x04