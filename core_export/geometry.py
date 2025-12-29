import torch
import torch.nn.functional as F
from core_export.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1: torch.Tensor, init_fmap2: torch.Tensor, geo_volume: torch.Tensor):
        corr_levels = 2
        corr_radius = 4
        self.num_levels = corr_levels
        self.radius = corr_radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        
        # for _ in range(self.num_levels-1):
        #     geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
        #     self.geo_volume_pyramid.append(geo_volume)

        # for _ in range(self.num_levels-1):
        #     init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
        #     self.init_corr_pyramid.append(init_corr)
        
        # !!!ONNX导出动态图像尺寸时需要用下面的代码替代, 否则avg_pool2d会报错
        for _ in range(self.num_levels - 1):
            geo_volume = self._downsample_disp(geo_volume)
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels - 1):
            init_corr = self._downsample_disp(init_corr)
            self.init_corr_pyramid.append(init_corr)

    @staticmethod
    def _downsample_disp(x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, 1, D]
        ONNX-safe 1D average pooling along D
        """
        n, c, _, d = x.shape

        # 保证偶数长度
        d2 = d // 2 * 2
        x = x[..., :d2]

        # reshape + mean 替代 avg_pool
        x = x.view(n, c, 1, d2 // 2, 2)
        x = x.mean(dim=-1)

        return x
    
    def __call__(self, disp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
   
    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr
    