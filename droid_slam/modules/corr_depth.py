import torch
import torch.nn.functional as F
import defCorrSample
import dispsSample
import droid_backends
class CorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = defCorrSample.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = defCorrSample.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class DefCorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, offset, radius):
        offset = offset.float()
        volume = volume.float()
        ctx.save_for_backward(volume,coords,offset)
        ctx.radius = radius
        corr, = defCorrSample.defCorr_index_forward(volume, coords, offset, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords, offset = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume,offset_grad = defCorrSample.defCorr_index_backward(volume, coords, offset, grad_output, ctx.radius)
        return grad_volume, None, offset_grad, None

class DepthSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, disps_embed, coords, radius):
        disps_embed = disps_embed.float()
        ctx.save_for_backward(disps_embed,coords)
        ctx.radius = radius
        corr, = dispsSample.dispsSample_forward(disps_embed, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        disps_embed,coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = dispsSample.dispsSample_backward(disps_embed, coords, grad_output, ctx.radius)
        return grad_volume, None, None, None

def per_Corr_Normalization(x, normalIndex, eps=1e-5):
    mean = torch.mean(x, dim=normalIndex)
    mean = mean.unsqueeze(dim=normalIndex[0]).unsqueeze(dim=normalIndex[1]).unsqueeze(dim=normalIndex[2])
    var = torch.var(x, dim=normalIndex, unbiased=False)+eps
    var = torch.sqrt(var).unsqueeze(dim=normalIndex[0]).unsqueeze(dim=normalIndex[1]).unsqueeze(dim=normalIndex[2])
    t = x - mean
    t = t / var
    return t

class CorrBlock_withDepth:
    def __init__(self, fmap1, fmap2, disps_embedjj, num_levels=4, radius=3):

        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.disps_pyramid = []

        corr = CorrBlock_withDepth.corr(fmap1,fmap2)

        # all pairs correlation
        b, n, c, s, h, w = corr.shape

        corr = corr.view(b,n*c,h,w,h,w)

        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2)

        disps_embedjj = disps_embedjj

        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)

        for i in range(self.num_levels):
            sz = (batch*num, h // 2 ** i, w // 2 ** i, 2)
            fmap_lvl = disps_embedjj.permute(0, 2, 3, 1).contiguous()
            self.disps_pyramid.append(fmap_lvl.view(*sz))
            disps_embedjj = F.avg_pool2d(disps_embedjj, 2, stride=2)



    def __call__(self, coords):
        corr_out_pyramid = []
        disps_out_pyramid = []
        batch, num, ht, wd, _ = coords.shape
        coords = coords.permute(0,1,4,2,3)
        coords = coords.contiguous().view(batch*num, 2, ht, wd)



        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords / 2 ** i, self.radius)
            corr_out_pyramid.append(corr.view(batch, num, -1, ht, wd))
        for i in range(self.num_levels):
            disp_list = []
            for c in range(2):
                dispi = DepthSampler.apply(self.disps_pyramid[i][:, :, :, c].contiguous(), coords / 2**i, self.radius - 1)
                disp_list.append(dispi.view(batch,num,-1,ht,wd))
            dispi = torch.cat(disp_list,dim=2)
            disps_out_pyramid.append(dispi.view(batch, num, -1, ht, wd))

        return torch.cat(corr_out_pyramid, dim=2), torch.cat(disps_out_pyramid, dim=2)

    def cat(self, other):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
        for i in range(self.num_levels):
            self.disps_pyramid[i] = torch.cat([self.disps_pyramid[i], other.disps_pyramid[i]], 0)
        return self


    def __getitem__(self, index):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][index]
        for i in range(self.num_levels):
            self.disps_pyramid[i] = self.disps_pyramid[i][index]
        return self


    @staticmethod
    def corr(fmap1, fmap2):
        """ all-pairs correlation """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, 1,  ht*wd, ht, wd)

class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        ctx.r = r
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = droid_backends.altcorr_forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            droid_backends.altcorr_backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None

class AltCorrBlock:
    def __init__(self, fmaps, disps_embedjj, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.offset = []

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B * N, C, H, W) / 4.0

        disps_embedjj = disps_embedjj.view(B*N,2,H,W)

        self.disps_pyramid = []
        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, H // 2 ** i, W // 2 ** i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(fmap_lvl.view(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2)

        for i in range(self.num_levels):
            sz = (B * N, H // 2 ** i, W // 2 ** i, 2)
            fmap_lvl = disps_embedjj.permute(0, 2, 3, 1).contiguous()
            self.disps_pyramid.append(fmap_lvl.view(*sz))
            disps_embedjj = F.avg_pool2d(disps_embedjj, 2, stride=2)

    def corr_fn(self, coords, ii, jj):
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5)

        corr_list = []
        disps_out_pyramid = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][:, ii]
            fmap2_i = self.pyramid[i][:, jj]

            coords_i = (coords / 2 ** i).reshape(B * N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B * N,) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B * N,) + fmap2_i.shape[2:])

            corr, = droid_backends.altcorr_forward(fmap1_i.float(), fmap2_i.float(), coords_i, self.radius)
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        coords =coords.view(-1,H,W,2).permute(0,3,1,2).contiguous()

        for i in range(self.num_levels):
            disp_list = []
            for i in range(2):
                dispi = DepthSampler.apply(self.disps_pyramid[i][:, :, :, i].contiguous(), coords/2**i,
                                           self.radius - 1)
                disp_list.append(dispi.view(B, N, -1, H, W))

            dispi = torch.cat(disp_list, dim=2)
            disps_out_pyramid.append(dispi.view(B, N, -1, H, W))

        corr = torch.cat(corr_list, dim=2)
        return corr, torch.cat(disps_out_pyramid, dim=2)

    def __call__(self, coords, ii, jj):
        squeeze_output = False
        if len(coords.shape) == 5:
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        corr, disps = self.corr_fn(coords, ii, jj)

        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous(), disps.contiguous()