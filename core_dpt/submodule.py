import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Literal

class ConvBlock(nn.Module):
    """## ConvBlock 
    
    ### Brief: Convolution -> Normalization -> Activation
    
    ### Args:
        - `in_channels`: number of input channels
        - `out_channels`: number of output channels
        - `kernel_size`: kernel size of convolution
        - `stride`: stride of convolution
        - `padding`: padding of convolution
        - `dilation`: dilation of convolution
        - `groups`: groups of convolution
        - `bias`: whether to use bias in convolution
        - `norm_fn`: normalization function, 'bn' or 'in' or 'none'
        - `activation_fn`: activation function, 'relu' or 'leaky' or 'mish' or 'silu' or none
        - `transpose_conv`: whether to use transpose convolution
        - `dropout`: dropout rate
        - `is_3d`: whether to use 3D convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int], tuple[int, int, int]],
        stride: Union[int, tuple[int, int], tuple[int, int, int]],
        padding: Union[int, tuple[int, int], tuple[int, int, int], str],
        dilation: Union[int, tuple[int, int], tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        norm_fn: Optional[Literal['bn', 'in', 'none']] = 'bn',
        activation_fn: Optional[Literal['leaky', 'relu', 'mish', 'silu', 'none']] = 'leaky',
        transpose_conv: bool = False,
        dropout: Optional[float] = None,
        is_3d: bool = False
    ):
        super().__init__()

        # 根据 `is_3d` 选择相应的卷积类型
        conv_cls = nn.Conv3d if is_3d else nn.Conv2d
        conv_transpose_cls = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        bn_norm_cls = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        in_norm_cls = nn.InstanceNorm3d if is_3d else nn.InstanceNorm2d

        # 卷积层
        if transpose_conv:
            self.conv = conv_transpose_cls(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = conv_cls(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        
        # 归一化层
        if norm_fn == 'bn':
            self.norm = bn_norm_cls(out_channels)
        elif norm_fn == 'in':
            self.norm = in_norm_cls(out_channels)
        else:
            self.norm = nn.Identity()
        
        # 激活
        if activation_fn == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation_fn == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == 'mish':
            self.activation = nn.Mish(inplace=True)
        elif activation_fn == 'silu':
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.conv(x))))

class BasicBlock(nn.Module):
    """## BasicBlock for ResNet 基础残差块

    ### Args:
        - `in_channels`: number of input channels
        - `out_channels`: number of output channels
        - `stride`: stride of convolution
        - `norm_fn`: normalization function, 'bn' or 'in' or 'none'
        - `activation_fn`: activation function, 'relu' or 'leaky' or 'none'
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, tuple[int, int]],
        norm_fn: Optional[Literal['bn', 'in', 'none']] = 'bn',
        activation_fn: Optional[Literal['leaky', 'relu', 'mish', 'silu', 'none']] = 'leaky'
    ):
        super().__init__()

        self.left = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, norm_fn=norm_fn, activation_fn=activation_fn),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm_fn=norm_fn, activation_fn='none'))
        
        self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, norm_fn=norm_fn, activation_fn='none') \
            if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.left(x) + self.shortcut(x), inplace=True)


class BottleNeck(nn.Module):
    """## BottleNeck for ResNet 瓶颈层
    
    ### Args:
        - `inplanes`: number of input channels
        - `planes`: number of output channels
        - `stride`: stride of convolution
        - `norm_fn`: normalization function, 'bn' or 'in' or 'none'
        - `activation_fn`: activation function, 'relu' or 'leaky' or 'none'
    """
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, tuple[int, int]] = 1,
        norm_fn: Optional[Literal['bn', 'in', 'none']] = 'bn',
        activation_fn: Optional[Literal['leaky', 'relu', 'mish', 'silu', 'none']] = 'leaky'
    ):
        super().__init__()
        
        self.left = nn.Sequential(
            ConvBlock(inplanes, planes//4, kernel_size=1, stride=1, padding=0, bias=False, norm_fn=norm_fn, activation_fn=activation_fn),
            ConvBlock(planes//4, planes//4, kernel_size=3, stride=stride, padding=1, norm_fn=norm_fn, activation_fn=activation_fn),
            ConvBlock(planes//4, planes, kernel_size=1, stride=1, padding=0, bias=False, norm_fn=norm_fn, activation_fn='none'))

        self.shortcut = ConvBlock(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False, activation_fn='none') \
            if stride != 1 or inplanes != planes else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.left(x) + self.shortcut(x), inplace=True)

class Conv2x(nn.Module):
    """## 特征拼接, BN, 若特征尺寸不同, 则进行最近邻插值

    ### Args:
        - `in_channels`: number of input channels
        - `out_channels`: number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 4, 2, 1)
        self.conv2 = ConvBlock(out_channels*2, out_channels*2, 3, 1, 1)

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        x = torch.cat((x, rem), 1) # concat
        x = self.conv2(x)
        return x


class Conv2x_IN(nn.Module):
    """## 特征拼接, IN, 若特征尺寸不同, 则进行最近邻插值

    ### Args:
        - `in_channels`: number of input channels
        - `out_channels`: number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 4, 2, 1, norm_fn='in')
        self.conv2 = ConvBlock(out_channels*2, out_channels*2, 3, 1, 1, norm_fn='in')

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        x = torch.cat((x, rem), 1)
        x = self.conv2(x)
        return x