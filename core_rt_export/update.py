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

    def forward(self, c, h, bi, bf, bc, bo, x):
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
        self.convc1 = nn.Conv2d(cor_planes, 24, 1, padding=0)
        self.convc2 = nn.Conv2d(24, 24, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 24, 7, padding=3)
        self.convd2 = nn.Conv2d(24, 24, 3, padding=1)
        self.conv = nn.Conv2d(24+24, 48-1, 3, padding=1)

    def forward(self, disp: torch.Tensor, corr: torch.Tensor):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)
    

class LSTMMultiUpdateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dims=48
        self.encoder = BasicMotionEncoder()
        encoder_output_dim = 48
        self.lstm = ConvLSTM(hidden_dims, encoder_output_dim)
        self.disp_head = DispHead(hidden_dims, hidden_dim=48, output_dim=1)
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims, 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, netC, netH, inp, corr=None, flow=None):
        motion_features = self.encoder(flow, corr)
        netC, netH = self.lstm(netC, netH, *inp, motion_features)

        delta_disp = self.disp_head(netH)
        mask_feat_4 = self.mask_feat_4(netH)
        return netC, netH, mask_feat_4, delta_disp
