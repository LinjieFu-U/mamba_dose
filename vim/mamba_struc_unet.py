## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import sys
sys.path.append('./')
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from vim.models_mamba import VisionMamba


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights druing initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :].to(x.device) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MambaUnet(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        inp_struc_channels=5,
        out_channels=1,
        dim=48,
        bias=False,
        T=1000,
        ch=64,
        ch_mult=[1, 2, 2, 4, 4],
        attn=[1],
        num_res_blocks=2,
        dropout=0,
    ):
        super(MambaUnet, self).__init__()

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )

        self.reduce_chan_level1 = nn.Conv2d(
            int(dim * 2**1), int(dim * 2**0), kernel_size=1, bias=bias
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up1_0 = Upsample(int(dim * 2**0))

        self.head = nn.Conv2d(inp_channels, 24, kernel_size=3, stride=1, padding=1)
        self.e_1 = VisionMamba(
            img_size=128, patch_size=2, embed_dim=48, depth=4, channels=24, **{}
        )
        self.e_2 = VisionMamba(
            img_size=64, patch_size=2, embed_dim=96, depth=6, channels=48, **{}
        )
        self.e_3 = VisionMamba(
            img_size=32, patch_size=2, embed_dim=192, depth=6, channels=96, **{}
        )
        self.e_4 = VisionMamba(
            img_size=16, patch_size=2, embed_dim=384, depth=8, channels=192, **{}
        )



        self.head_struc = nn.Conv2d(inp_struc_channels, 24, kernel_size=3, stride=1, padding=1)
        self.e_1_struc = VisionMamba(
            img_size=128, patch_size=2, embed_dim=48, depth=4, channels=24, **{}
        )
        self.e_2_struc = VisionMamba(
            img_size=64, patch_size=2, embed_dim=96, depth=6, channels=48, **{}
        )
        self.e_3_struc = VisionMamba(
            img_size=32, patch_size=2, embed_dim=192, depth=6, channels=96, **{}
        )
        self.e_4_struc = VisionMamba(
            img_size=16, patch_size=2, embed_dim=384, depth=8, channels=192, **{}
        )

        # self.m = VisionMamba(
        #     img_size=16, patch_size=1, embed_dim=384, depth=8, channels=192, **{}
        # )
        self.d_4 = VisionMamba(
            img_size=16, patch_size=1, embed_dim=192, depth=6, channels=192, **{}
        )
        self.d_3 = VisionMamba(
            img_size=32, patch_size=1, embed_dim=96, depth=6, channels=96, **{}
        )
        self.d_2 = VisionMamba(
            img_size=64, patch_size=1, embed_dim=48, depth=4, channels=48, **{}
        )
        self.d_1 = VisionMamba(
            img_size=128, patch_size=1, embed_dim=48, depth=4, channels=48, **{}
        )
        self.output = nn.Conv2d(
            48, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        tdim = 24 * 4
        self.time_embedding = TimeEmbedding(tdim)
        self.temb_proj_e1 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 48),
        )
        self.temb_proj_e2 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 96),
        )
        self.temb_proj_e3 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 192),
        )
        self.temb_proj_e4 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 384),
        )

        self.temb_proj_m = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 384),
        )


        self.temb_proj_d4 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 192),
        )
        self.temb_proj_d3 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 96),
        )
        self.temb_proj_d2 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 48),
        )
        self.temb_proj_d1 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, 48),
        )
        # self.refinement = VisionMamba(img_size=128, patch_size=1, embed_dim=96, depth=4, channels=96, **{})
        
    def forward(self, inp_img,structure, t):
        temb = self.time_embedding(t)
        temb_e1 = self.temb_proj_e1(temb)[:, :, None, None]
        temb_e2 = self.temb_proj_e2(temb)[:, :, None, None]
        temb_e3 = self.temb_proj_e3(temb)[:, :, None, None]
        temb_e4 = self.temb_proj_e4(temb)[:, :, None, None]
        temb_d1 = self.temb_proj_d1(temb)[:, :, None, None]
        temb_d2 = self.temb_proj_d2(temb)[:, :, None, None]
        temb_d3 = self.temb_proj_d3(temb)[:, :, None, None]
        temb_d4 = self.temb_proj_d4(temb)[:, :, None, None]
        # temb_m = self.temb_proj_m(temb)[:, :, None, None]
        inp = self.head(inp_img)
        inp_struc=self.head_struc(structure)

        out_enc_level1 = self.e_1(inp)
        out_enc_struc_level1 = self.e_1_struc(inp_struc)
        out_enc_level1 = out_enc_level1 + out_enc_struc_level1 + temb_e1

        out_enc_level2 = self.e_2(out_enc_level1)
        out_enc_struc_level2 = self.e_2_struc(out_enc_struc_level1)
        out_enc_level2 = out_enc_level2 + out_enc_struc_level2 + temb_e2

        out_enc_level3 = self.e_3(out_enc_level2)
        out_enc_struc_level3 = self.e_3_struc(out_enc_struc_level2)
        out_enc_level3 = out_enc_level3 + out_enc_struc_level3 + temb_e3
        
        out_enc_level4 = self.e_4(out_enc_level3)
        out_enc_struc_level4 = self.e_4_struc(out_enc_struc_level3)
        out_enc_level4 = out_enc_level4 + out_enc_struc_level4 + temb_e4

        # out_m=self.m(out_enc_level4)
        # out_m=out_m+temb_m


        inp_dec_level4 = self.up4_3(out_enc_level4)
        inp_dec_level4 = torch.cat([inp_dec_level4, out_enc_level3], 1)
        inp_dec_level4 = self.reduce_chan_level3(inp_dec_level4)
        out_dec_level4 = self.d_4(inp_dec_level4)
        out_dec_level4 = out_dec_level4 + temb_d4

        inp_dec_level3 = self.up3_2(out_dec_level4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)
        inp_dec_level3 = self.reduce_chan_level2(inp_dec_level3)
        out_dec_level3 = self.d_3(inp_dec_level3)
        out_dec_level3 = out_dec_level3 + temb_d3

        inp_dec_level2 = self.up2_1(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level1], 1)
        inp_dec_level2 = self.reduce_chan_level1(inp_dec_level2)
        out_dec_level2 = self.d_2(inp_dec_level2)
        out_dec_level2 = out_dec_level2 + temb_d2

        inp_dec_level1 = self.up1_0(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, inp], 1)
        out_dec_level1 = self.d_1(inp_dec_level1)
        out_dec_level1 = out_dec_level1 + temb_d1

        out_dec_level = self.output(out_dec_level1)

        return out_dec_level


if __name__ == "__main__":
    x = torch.randn(8, 2, 256, 256).cuda()
    t = torch.randint(1000, size=(8,)).cuda()
    model = MambaUnet().cuda()
    y = model(x, t)
    print(y.shape)
