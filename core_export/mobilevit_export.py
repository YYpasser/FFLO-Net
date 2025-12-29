import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.mobilevit import MobileVitV2Block, LinearTransformerBlock, num_groups, LayerFn
from timm.layers import make_divisible, GroupNorm1  # 导入GroupNorm1

# 自定义动态MobileVitV2Block
class DynamicMobileVitV2Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        layers = kwargs.get('layers', LayerFn())
        groups = num_groups(kwargs.get('group_size', 1), kwargs.get('in_chs'))
        self.in_chs = kwargs.get('in_chs')
        self.out_chs = kwargs.get('out_chs', self.in_chs)
        self.bottle_ratio = kwargs.get('bottle_ratio', 1.0)
        self.transformer_dim = kwargs.get('transformer_dim', make_divisible(self.bottle_ratio * self.in_chs))
        self.transformer_depth = kwargs.get('transformer_depth', 2)
        self.patch_size = kwargs.get('patch_size', 2)
        self.drop_path_rate = kwargs.get('drop_path_rate', 0.)
        self.attn_drop = kwargs.get('attn_drop', 0.)
        self.drop = kwargs.get('drop', 0.)
        self.mlp_ratio = kwargs.get('mlp_ratio', 2.0)
        self.transformer_norm_layer = kwargs.get('transformer_norm_layer', GroupNorm1)

        self.conv_kxk = layers.conv_norm_act(
            self.in_chs, self.in_chs, kernel_size=3,
            stride=1, groups=groups, dilation=(1,1))
        self.conv_1x1 = nn.Conv2d(self.in_chs, self.transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            LinearTransformerBlock(
                self.transformer_dim,
                mlp_ratio=self.mlp_ratio,
                attn_drop=self.attn_drop,
                drop=self.drop,
                drop_path=self.drop_path_rate,
                act_layer=layers.act,
                norm_layer=self.transformer_norm_layer
            )
            for _ in range(self.transformer_depth)
        ])

        if self.transformer_norm_layer is GroupNorm1:
            self.norm = self.transformer_norm_layer(self.transformer_dim)
        else:
            self.norm = self.transformer_norm_layer(1, self.transformer_dim)

        self.conv_proj = layers.conv_norm_act(self.transformer_dim, self.out_chs, kernel_size=1, stride=1, apply_act=False)

        self.coreml_exportable = False
        self.training = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size, self.patch_size
        
        num_patch_h = (H + patch_h - 1) // patch_h
        num_patch_w = (W + patch_w - 1) // patch_w
        new_h = num_patch_h * patch_h
        new_w = num_patch_w * patch_w
        num_patches = num_patch_h * num_patch_w

        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        x = self.conv_kxk(x)
        x = self.conv_1x1(x)
        C = x.shape[1]

        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)

        x = self.transformer(x)
        x = self.norm(x)

        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, new_h, new_w)

        x = self.conv_proj(x)
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


def replace_mobilevit_blocks(model):
    """递归替换模型中所有MobileVitV2Block为自定义的DynamicMobileVitV2Block"""
    for name, module in model.named_children():
        if isinstance(module, MobileVitV2Block):
            kwargs = {
                'in_chs': module.conv_kxk.conv.in_channels,
                'out_chs': module.conv_proj.conv.out_channels,
                'bottle_ratio': module.conv_1x1.out_channels / module.conv_kxk.conv.in_channels,
                'transformer_depth': len(module.transformer),
                'patch_size': module.patch_size[0],
                'attn_drop': module.transformer[0].attn.attn_drop.p if hasattr(module.transformer[0].attn, 'attn_drop') else 0.,
                'drop': module.transformer[0].attn.out_drop.p if hasattr(module.transformer[0].attn, 'out_drop') else 0.,
                'drop_path_rate': module.transformer[0].drop_path1.drop_prob if hasattr(module.transformer[0], 'drop_path1') else 0.,
                'mlp_ratio': module.transformer[0].mlp.hidden_features / module.transformer[0].mlp.in_features if hasattr(module.transformer[0].mlp, 'hidden_features') else 2.0,
                'transformer_norm_layer': type(module.norm),
                'layers': LayerFn(act=type(module.conv_kxk.act) if hasattr(module.conv_kxk, 'act') else nn.SiLU),
            }
            setattr(model, name, DynamicMobileVitV2Block(**kwargs))
        else:
            replace_mobilevit_blocks(module)
    return model