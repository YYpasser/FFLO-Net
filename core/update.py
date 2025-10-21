import torch
import torch.nn as nn
import torch.nn.functional as F


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvLSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.conv_it = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_c_t = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_ft = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_ot = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, c, h, bi, bf, bc, bo, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        ft = torch.sigmoid(self.conv_ft(hx) + bf)
        it = torch.sigmoid(self.conv_it(hx) + bi)
        c_t = torch.tanh(self.conv_c_t(hx) + bc)
        ct = c * ft + it * c_t
        ot = torch.sigmoid(self.conv_ot(hx) + bo)
        ht = ot * torch.tanh(ct)
        return ct, ht


class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        corr_levels = 2
        corr_radius = 4
        cor_planes = corr_levels * (2*corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp: torch.Tensor, corr: torch.Tensor):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x: torch.Tensor):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x: torch.Tensor):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x: torch.Tensor, dest: torch.Tensor):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class LSTMMultiUpdateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dims=[128, 128, 128]
        self.encoder = BasicMotionEncoder()
        encoder_output_dim = 128
        self.lstm04 = ConvLSTM(hidden_dims[2], encoder_output_dim + hidden_dims[1])
        self.lstm08 = ConvLSTM(hidden_dims[1], hidden_dims[0] + hidden_dims[2])
        self.lstm16 = ConvLSTM(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, netC, netH, inp, corr=None, flow=None, iter04=True, iter08=True, iter16=True, update=True):
        if iter16:
            netC[2], netH[2] = self.lstm16(netC[2], netH[2], *(inp[2]), pool2x(netH[1]))
        if iter08:
            netC[1], netH[1] = self.lstm08(netC[1], netH[1], *(inp[1]), pool2x(netH[0]),
                                        interp(netH[2], netH[1]))
        if iter04:
            motion_features = self.encoder(flow, corr)
            netC[0], netH[0] = self.lstm04(netC[0], netH[0], *(inp[0]), motion_features,
                                        interp(netH[1], netH[0]))

        if not update:
            return netC, netH
        
        delta_disp = self.disp_head(netH[0])
        mask_feat_4 = self.mask_feat_4(netH[0])
        return netC, netH, mask_feat_4, delta_disp
