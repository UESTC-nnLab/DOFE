import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from droid_slam.modules.depthGuideCorr import Attnc
import numbers
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,embed_dim):
        super().__init__()

        self.Embedd_s1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.Embedd_m1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2,padding=1)
        self.Embedd_l1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2,dilation=1,padding=1)

    def forward(self, f1):

        f1_s = self.Embedd_s1(f1)
        f1_m = self.Embedd_m1(f1)
        f1_l = self.Embedd_l1(f1)

        f1_patched = torch.cat([f1_s,f1_m,f1_l],dim=1)

        return f1_patched

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class fftCorrelation(nn.Module):
    def __init__(self, dim, bias):
        super(fftCorrelation, self).__init__()

        self.Embedd_d = nn.Sequential(nn.Conv2d(2*dim, dim * 2 * 3, kernel_size=2, stride=2),
                                      nn.BatchNorm2d(dim * 2 * 3),
                                      nn.Tanh(),
                                      )

        self.f1_to_hidden = nn.Conv2d(3*dim, dim * 2 * 3, kernel_size=1, bias=bias)
        self.f2_to_hidden = nn.Conv2d(3*dim, dim * 2 * 3, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim * 2 * 3, dim * 4, kernel_size=1, padding=0, bias=bias)

        self.patch_size = 4

        self.upSample = nn.PixelShuffle(2)
        self.patchEmbeded = PatchEmbed(dim)

        self.amp_attn = Attnc()
        self.pha_attn = Attnc()

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

    def forward(self, f1, disps_embed, ii, jj):
        b,_,h,w = f1.shape

        f1_patched = self.patchEmbeded(f1)

        d_embed = self.Embedd_d(disps_embed)

        d_embed = d_embed[jj]
        f1 = f1_patched[ii]
        f2 = f1_patched[jj]
        q = self.f1_to_hidden(f1)
        k = self.f2_to_hidden(f2)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        d_patch = rearrange(d_embed, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q_patch.float()+1e-7, norm='backward')
        k_fft = torch.fft.rfft2(k_patch.float()+1e-7, norm='backward')
        d_fft = torch.fft.rfft2(d_patch.float()+1e-7, norm='backward')

        k_amp = torch.abs(k_fft)
        k_pha = torch.angle(k_fft)

        d_amp = torch.abs(d_fft)
        d_pha = torch.angle(d_fft)

        N,C,h,w,h1,w1 = d_amp.shape

        k_amp = k_amp.view(N, C, h * w, h1 * w1)
        k_pha = k_pha.view(N, C, h * w, h1 * w1)

        d_amp = d_amp.view(N, C, h * w, h1 * w1)
        d_pha = d_pha.view(N, C, h * w, h1 * w1)

        k_amp = self.amp_attn(k_amp, d_amp)
        k_pha = self.pha_attn(k_pha, d_pha)

        k_amp = k_amp.view(N, C, h, w, h1, w1)
        k_pha = k_pha.view(N, C, h, w, h1, w1)

        real = k_amp * torch.cos(k_pha) + 1e-7
        imag = k_amp * torch.sin(k_pha) + 1e-7
        k_fft = torch.complex(real, imag) + 1e-7

        out = q_fft * k_fft
        out = torch.fft.irfft2(out+1e-7, s=(self.patch_size, self.patch_size), norm='backward')
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        output = self.project_out(out)
        output = self.norm(self.upSample(output))

        return output

