import torch.nn as nn
import torch.nn.init as init
import timm
from core_dpt.submodule import *
from core_dpt.fusion import *


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    # def weight_init(self, module):
    #     if isinstance(module, (nn.Conv2d, nn.Conv3d)):
    #         init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #         if module.bias is not None:
    #             init.constant_(module.bias, 0)
    #     elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
    #         init.constant_(module.weight, 1)
    #         init.constant_(module.bias, 0)

    # def init_new_layers(self, pretrained_names):
    #     """
    #     只初始化不在 pretrained_names 里的层
    #     """
    #     for name, module in self.named_modules():
    #         # 去掉可能的前缀（因为我们包了 Sequential）
    #         clean_name = name.replace('net_stem.', '').replace('encoder_block', 'blocks.')
                
    #         # 检查是否是预训练模型里的层
    #         if not any(clean_name.startswith(p) for p in pretrained_names):
    #             self.weight_init(module)


class FeatDown(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(FeatDown, self).__init__()
        self.maxpool = nn.MaxPool2d(scale)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class FeatureExtractor(SubModule):
    
    def __init__(self):
        super().__init__()

        model = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
        # pretrained_names = {name for name, _ in model.named_modules()}
        
        layers = [1,2,3,5,6]
        en_chans  = [16, 24, 40, 112, 160]
        de_chans  = [32, 48, 80, 224, 160]
        cat_chans = [en_chans[0]+de_chans[0],
                     en_chans[0]+en_chans[1]+de_chans[1],
                     en_chans[0]+en_chans[1]+en_chans[2]+de_chans[2],
                     en_chans[0]+en_chans[1]+en_chans[2]+en_chans[3]+de_chans[3]]

        self.net_stem = nn.Sequential(model.conv_stem, model.bn1, model.act1)
        self.encoder_block1 = nn.Sequential(*model.blocks[0:layers[0]])
        self.encoder_block2 = nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.encoder_block3 = nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.encoder_block4 = nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.encoder_block5 = nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.featdown_x02_scale02 = FeatDown(en_chans[0], en_chans[0], 2)
        self.featdown_x02_scale12 = FeatDown(en_chans[0], en_chans[0], 2)
        self.featdown_x02_scale22 = FeatDown(en_chans[0], en_chans[0], 2)
        self.featdown_x04_scale02 = FeatDown(en_chans[1], en_chans[1], 2)
        self.featdown_x04_scale12 = FeatDown(en_chans[1], en_chans[1], 2)
        self.featdown_x08_scale02 = FeatDown(en_chans[2], en_chans[2], 2)

        self.upsample32_16 = ConvBlock(en_chans[4], de_chans[3], 2, 2, 0, transpose_conv=True)
        self.decoder_block4 = FeatureFusion(cat_chans[3], de_chans[3])
        self.upsample16_8 = ConvBlock(de_chans[3], de_chans[2], 2, 2, 0, transpose_conv=True)
        self.decoder_block3 = FeatureFusion(cat_chans[2], de_chans[2])
        self.upsample8_4 = ConvBlock(de_chans[2], de_chans[1], 2, 2, 0, transpose_conv=True)
        self.decoder_block2 = FeatureFusion(cat_chans[1], de_chans[1])
        self.upsample4_2 = ConvBlock(de_chans[1], de_chans[0], 2, 2, 0, transpose_conv=True)
        self.decoder_block1 = FeatureFusion(cat_chans[0], de_chans[0])

        self.expand_x16 = ConvBlock(en_chans[3], de_chans[3], 1, 1, 0)
        self.expand_x08 = ConvBlock(en_chans[2], de_chans[2], 1, 1, 0)
        self.expand_x04 = ConvBlock(en_chans[1], de_chans[1], 1, 1, 0)
        self.expand_x02 = ConvBlock(en_chans[0], de_chans[0], 1, 1, 0)

        self.fuse_x16 = CGAFusion(de_chans[3], 8)
        self.fuse_x08 = CGAFusion(de_chans[2], 8)
        self.fuse_x04 = CGAFusion(de_chans[1], 8)
        self.fuse_x02 = CGAFusion(de_chans[0], 8)

        # self.init_new_layers(pretrained_names)

    def freeze_backbone(self):
        for param in self.encoder_block1.parameters():
            param.requires_grad = False
        for param in self.encoder_block2.parameters():
            param.requires_grad = False
        for param in self.encoder_block3.parameters():
            param.requires_grad = False
        for param in self.encoder_block4.parameters():
            param.requires_grad = False
        for param in self.encoder_block5.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder_block1.parameters():
            param.requires_grad = True
        for param in self.encoder_block2.parameters():
            param.requires_grad = True
        for param in self.encoder_block3.parameters():
            param.requires_grad = True
        for param in self.encoder_block4.parameters():
            param.requires_grad = True
        for param in self.encoder_block5.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Encoder
        x = self.net_stem(x)
        x02 = self.encoder_block1(x)
        x04 = self.encoder_block2(x02)
        x08 = self.encoder_block3(x04)
        x16 = self.encoder_block4(x08)
        x32 = self.encoder_block5(x16)

        # shortcut
        x02_down_x04 = self.featdown_x02_scale02(x02)
        x02_down_x08 = self.featdown_x02_scale12(x02_down_x04)
        x02_down_x16 = self.featdown_x02_scale22(x02_down_x08)
        x04_down_x08 = self.featdown_x04_scale02(x04)
        x04_down_x16 = self.featdown_x04_scale12(x04_down_x08)
        x08_down_x16 = self.featdown_x08_scale02(x08)

        # Decoder
        x16_up = self.upsample32_16(x32)
        x16_up = self.decoder_block4(x16_up, x16, x08_down_x16, x04_down_x16, x02_down_x16)
        x08_up = self.upsample16_8(x16_up)
        x08_up = self.decoder_block3(x08_up, x08, x04_down_x08, x02_down_x08)
        x04_up = self.upsample8_4(x08_up)
        x04_up = self.decoder_block2(x04_up, x04, x02_down_x04)
        x02_up = self.upsample4_2(x04_up)
        x02_up = self.decoder_block1(x02_up, x02)

        # CGA Fusion
        x02_up = self.fuse_x02(x02_up, self.expand_x02(x02))
        x04_up = self.fuse_x04(x04_up, self.expand_x04(x04))
        x08_up = self.fuse_x08(x08_up, self.expand_x08(x08))
        x16_up = self.fuse_x16(x16_up, self.expand_x16(x16))

        return [x02_up, x04_up, x08_up, x16_up, x32]


class ContextExtractor(SubModule):
    def __init__(self):
        super().__init__()
        
        model = timm.create_model('mobilevitv2_200.cvnets_in1k', pretrained=True, features_only=True)
        
        self.net_stem = model.stem
        self.encoder_block1 = torch.nn.Sequential(*model.stages_0)
        self.encoder_block2 = torch.nn.Sequential(*model.stages_1)
        self.encoder_block3 = torch.nn.Sequential(*model.stages_2)
        self.encoder_block4 = torch.nn.Sequential(*model.stages_3)

        self.decoder_block1_1 = nn.Sequential(
            ConvBlock(256, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 3, 1, 1))
        self.decoder_block1_2 = nn.Sequential(
            ConvBlock(256, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 3, 1, 1))
        self.decoder_block2_1 = nn.Sequential(
            ConvBlock(512, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 3, 1, 1))
        self.decoder_block2_2 = nn.Sequential(
            ConvBlock(512, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 3, 1, 1))
        self.decoder_block3_1 = nn.Sequential(
            ConvBlock(768, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 3, 1, 1))
        self.decoder_block3_2 = nn.Sequential(
            ConvBlock(768, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 3, 1, 1))

    def freeze_backbone(self):
        for param in self.encoder_block1.parameters():
            param.requires_grad = False
        for param in self.encoder_block2.parameters():
            param.requires_grad = False
        for param in self.encoder_block3.parameters():
            param.requires_grad = False
        for param in self.encoder_block4.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder_block1.parameters():
            param.requires_grad = True
        for param in self.encoder_block2.parameters():
            param.requires_grad = True
        for param in self.encoder_block3.parameters():
            param.requires_grad = True
        for param in self.encoder_block4.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        # Encoder
        x02 = self.net_stem(x)
        x02 = self.encoder_block1(x02)
        x04 = self.encoder_block2(x02)
        x08 = self.encoder_block3(x04)
        x16 = self.encoder_block4(x08)

        # Decoder
        x16_1 = self.decoder_block3_1(x16)
        x16_2 = self.decoder_block3_2(x16)
        x08_1 = self.decoder_block2_1(x08)
        x08_2 = self.decoder_block2_2(x08)
        x04_1 = self.decoder_block1_1(x04)
        x04_2 = self.decoder_block1_2(x04)

        return [[x04_1,x04_2], [x08_1,x08_2], [x16_1,x16_2]]