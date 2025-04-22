import torch
from torch import nn

class Attnc(nn.Module):

    def __init__(self, k_size=3):
        super(Attnc, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y1 = self.avg_pool(y)
        y2 = self.max_pool(y)

        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y2 = self.conv(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y1+y2)

        return x * y.expand_as(x)

import matplotlib.pyplot as plt
def visualflowFeature(dispsUncertain_mask):

    dispsUncertain_mask = dispsUncertain_mask[2].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    im1 = ax.imshow(dispsUncertain_mask.reshape(48, 64))
    ax.set_title("s_attn")
    fig.colorbar(im1, ax=ax)
    plt.show()

class Attns(nn.Module):

    def __init__(self, k_size=3):
        super(Attns, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x, y):
        avg_out = torch.mean(y,dim=1,keepdim=True)
        max_out,_= torch.max(y,dim=1,keepdim=True)

        y = torch.cat([avg_out,max_out],dim=1)
        y = self.sigmoid(self.conv1(y))
        # visualflowFeature(y)
        return x * y

class CrossGuide(nn.Module):
    def __init__(self, k_size=3):
        super(CrossGuide, self).__init__()

        self.dimAlignment = nn.Sequential(nn.Conv2d(200, 196, kernel_size=1),
                                          nn.BatchNorm2d(196)
                                          )
        self.s_attn = Attns()

        self.eacGuide_channel = Attnc(k_size)

    def forward(self, semantic_corr, depth_corr):

        depth_corr = self.dimAlignment(depth_corr)
        x = self.s_attn(semantic_corr, depth_corr)
        se_corr= self.eacGuide_channel(x, depth_corr)


        return se_corr

if __name__ =="__main__":
    corr = torch.randn((1,196,48,64))
    dcorr = torch.randn((1,200,48,64))
    cg = CrossGuide()
    se = cg(corr,dcorr)