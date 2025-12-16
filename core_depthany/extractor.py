import torch.nn as nn
import torch.nn.init as init
from core_depthany.submodule import *
from core_depthany.fusion import *


class FeatDown(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(FeatDown, self).__init__()
        self.maxpool = nn.MaxPool2d(scale)
        self.conv = ConvBlock(in_channels, out_channels, 3, 1, 1, norm_fn='in', activation_fn='leaky')

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class FeatureExtractor(nn.Module):
    
    def __init__(self, dim_list):
        super().__init__()

        en_chans  = [dim_list[0], dim_list[0], dim_list[0], dim_list[0]]
        de_chans  = [48, 64, 192, 160]
        cat_chans = [en_chans[0]+de_chans[0],
                     en_chans[0]+en_chans[1]+de_chans[1],
                     en_chans[0]+en_chans[1]+en_chans[2]+de_chans[2],
                     en_chans[0]+en_chans[1]+en_chans[2]+en_chans[3]+de_chans[3]]

        self.feat_x32 = ConvBlock(en_chans[3], de_chans[3], 3, 1, 1, norm_fn='in', activation_fn='leaky')

        self.featdown_x04_scale02 = FeatDown(en_chans[0], en_chans[0], 2)
        self.featdown_x04_scale12 = FeatDown(en_chans[0], en_chans[0], 2)
        self.featdown_x08_scale02 = FeatDown(en_chans[1], en_chans[1], 2)

        self.upsample32_16  = ConvBlock(de_chans[3], de_chans[2], 4, 2, 1, norm_fn='in', activation_fn='leaky', transpose_conv=True)
        self.decoder_block4 = FeatureFusion(cat_chans[2], de_chans[2], norm_fn='in', activation_fn='leaky')
        self.upsample16_08  = ConvBlock(de_chans[2], de_chans[1], 4, 2, 1, norm_fn='in', activation_fn='leaky', transpose_conv=True)
        self.decoder_block3 = FeatureFusion(cat_chans[1], de_chans[1], norm_fn='in', activation_fn='leaky')
        self.upsample08_04  = ConvBlock(de_chans[1], de_chans[0], 4, 2, 1, norm_fn='in', activation_fn='leaky', transpose_conv=True)
        self.decoder_block2 = FeatureFusion(cat_chans[0], de_chans[0], norm_fn='in', activation_fn='leaky')

        self.expand_x16 = ConvBlock(en_chans[2], de_chans[2], 1, 1, 0, norm_fn='in', activation_fn='leaky')
        self.expand_x08 = ConvBlock(en_chans[1], de_chans[1], 1, 1, 0, norm_fn='in', activation_fn='leaky')
        self.expand_x04 = ConvBlock(en_chans[0], de_chans[0], 1, 1, 0, norm_fn='in', activation_fn='leaky')

        self.fuse_x16 = CGAFusion(de_chans[2], 8)
        self.fuse_x08 = CGAFusion(de_chans[1], 8)
        self.fuse_x04 = CGAFusion(de_chans[0], 8)


    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        
        # Encoder
        x32 = self.feat_x32(features[3])

        # shortcut
        x04_down_x08 = self.featdown_x04_scale02(features[0])
        x04_down_x16 = self.featdown_x04_scale12(x04_down_x08)
        x08_down_x16 = self.featdown_x08_scale02(features[1])

        # Decoder
        x16_up = self.upsample32_16(x32)
        x16_up = self.decoder_block4(x16_up, features[2], x08_down_x16, x04_down_x16)
        x08_up = self.upsample16_08(x16_up)
        x08_up = self.decoder_block3(x08_up, features[1], x04_down_x08)
        x04_up = self.upsample08_04(x08_up)
        x04_up = self.decoder_block2(x04_up, features[0])

        # CGA Fusion
        x04_up = self.fuse_x04(x04_up, self.expand_x04(features[0]))
        x08_up = self.fuse_x08(x08_up, self.expand_x08(features[1]))
        x16_up = self.fuse_x16(x16_up, self.expand_x16(features[2]))

        return [x04_up, x08_up, x16_up, x32]


class ContextExtractor(nn.Module):
    def __init__(self, dim_list, output_dim):
        super().__init__()
        
        self.decoder_block1_1 = nn.Sequential(
            ConvBlock(dim_list[0]+48, output_dim, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1))
        self.decoder_block1_2 = nn.Sequential(
            ConvBlock(dim_list[0]+48, output_dim, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1))
        
        self.decoder_block2_1 = nn.Sequential(
            ConvBlock(dim_list[0]+96, output_dim, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1))
        self.decoder_block2_2 = nn.Sequential(
            ConvBlock(dim_list[0]+96, output_dim, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1))
        
        self.decoder_block3_1 = nn.Sequential(
            ConvBlock(dim_list[0]+192, output_dim, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1))
        self.decoder_block3_2 = nn.Sequential(
            ConvBlock(dim_list[0]+192, output_dim, 1, 1, 0, norm_fn='in', activation_fn='leaky'),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1))

    def forward(self, features: list[torch.Tensor], stem_x_list: list[torch.Tensor]) -> list[list[torch.Tensor]]:

        # Decoder
        x16_1 = self.decoder_block3_1(torch.cat((features[2], stem_x_list[0]), 1))
        x16_2 = self.decoder_block3_2(torch.cat((features[2], stem_x_list[0]), 1))
        x08_1 = self.decoder_block2_1(torch.cat((features[1], stem_x_list[1]), 1))
        x08_2 = self.decoder_block2_2(torch.cat((features[1], stem_x_list[1]), 1))
        x04_1 = self.decoder_block1_1(torch.cat((features[0], stem_x_list[2]), 1))
        x04_2 = self.decoder_block1_2(torch.cat((features[0], stem_x_list[2]), 1))

        return [[x04_1,x04_2], [x08_1,x08_2], [x16_1,x16_2]]