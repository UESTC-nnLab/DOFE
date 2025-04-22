import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from droid_slam.modules.extractor import BasicEncoder
from modules.UpdateModule import UpdateModule3D

from modules.depthGuideCorr import CrossGuide


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3, 3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8 * ht, 8 * wd, dim)

    return up_data


def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch * num, ht, wd, 1)
    mask = mask.view(batch * num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8 * ht, 8 * wd)


from dpt.models import DPTDepthModel
class DPTModel(nn.Module):
    def __init__(self):
        super(DPTModel, self).__init__()
        dpt = DPTDepthModel(
            path="/home/honsen/honsen/depthEstimation/DPT-main/weights/dpt_hybrid-midas-501f0c75.pt",
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.depth_model = dpt.cuda()
        self.depth_model.requires_grad_(False)
        self.depth_model.eval()

    def forward(self, x):
        output_48x64, output = self.depth_model(x)
        s = .7 * torch.quantile(output.float(), .98)
        output = output/s
        return  output_48x64, output

from droid_slam.modules.patchEmbed.toPatch import fftCorrelation
class DOFE_Net(nn.Module):
    def __init__(self):
        super(DOFE_Net, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')  # feature network
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')  # context network

        self.crossGuide = CrossGuide()

        self.adapter = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            nn.BatchNorm2d(2),
            nn.Tanh(),
           )
        self.dispsDownsample = nn.Sequential(nn.Conv2d(256, 2, 1, padding=0),
            nn.BatchNorm2d(2),nn.Tanh())
        self.update = UpdateModule3D()
        self.fftCorr = fftCorrelation(128, True)

    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2, 1, 0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        net, inp = net.split([128, 128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp