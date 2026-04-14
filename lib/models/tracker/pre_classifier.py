import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class pre_classifier(nn.Module):
    def __init__(self, embed=768):
        super(pre_classifier, self).__init__()

        self.s_classifier = Self_classifier()
        self.c_classifier = Cross_classifier()


    def forward(self, z_r, z_i, x_r, x_i): # B, tokens, embedding

        self_score_r = self.s_classifier(x_r).unsqueeze(2)
        self_score_i = self.s_classifier(x_i).unsqueeze(2)
        # score1 = torch.softmax(torch.cat((self_score_r, self_score_i), dim=2), dim=1)

        s1, s2 = self.c_classifier(z_r, z_i, x_r, x_i)


        return self_score_r, self_score_i, s1, s2



class Self_classifier(nn.Module):
    def __init__(self, embed=768):
        super(Self_classifier, self).__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels=embed, out_channels=int(embed/4), kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(int(embed/4)),
            nn.GELU(),
            nn.Conv2d(in_channels=int(embed/4), out_channels=int(embed/8), kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(int(embed / 8)),
            nn.GELU(),
            nn.Conv2d(in_channels=int(embed / 8), out_channels=int(embed / 16), kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(int(embed / 16)),
            nn.GELU()
        )
        self.fc = nn.Linear(in_features=int(embed / 16), out_features=1)
        # self.sig = nn.Sigmoid
        self.embed = embed

    def forward(self, x):
        B, t, embed = x.shape
        x = x.permute(0, 2, 1).view(B, embed, int(math.sqrt(t)), int(math.sqrt(t)))
        x = self.fc(self.downconv(x).view(B, -1, int(embed / 16)))
        out = torch.sigmoid(x.squeeze(1))
        return out



class Cross_classifier(nn.Module):
    def __init__(self, embed=768):
        super(Cross_classifier, self).__init__()

        self.f_z = nn.Sequential(
            nn.Linear(in_features=int(embed*2), out_features=int(embed/2)),
            nn.LayerNorm(int(embed/2)),
            nn.GELU()
        )
        self.down_r = nn.Sequential(
            nn.Conv2d(in_channels=embed, out_channels=int(embed/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(embed/2)),
            nn.GELU()
        )
        self.down_i = nn.Sequential(
            nn.Conv2d(in_channels=embed, out_channels=int(embed/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(embed/2)),
            nn.GELU()
        )
        self.c = nn.Parameter(torch.ones(1) * embed/2)
        self.embed = embed
    def forward(self, z_r, z_i, x_r, x_i):
        z_f = self.f_z(torch.cat((z_r, z_i), dim=2))
        B, t, embed = x_r.shape
        _, t1, e1 = z_f.shape
        z_f = z_f.permute(0, 2, 1).view(B, e1, int(math.sqrt(t1)), int(math.sqrt(t1)))
        x_r = self.down_r(x_r.permute(0, 2, 1).view(B, embed, int(math.sqrt(t)), int(math.sqrt(t))))
        x_i = self.down_i(x_i.permute(0, 2, 1).view(B, embed, int(math.sqrt(t)), int(math.sqrt(t))))
        if int(math.sqrt(t)) == 16:
            x_r = x_r[:, :, 4:-4, 4:-4]
            x_i = x_i[:, :, 4:-4, 4:-4]
            s1 = torch.sigmoid(xcorr_fast(x_r, z_f) / self.c)
            s2 = torch.sigmoid(xcorr_fast(x_i, z_f) / self.c)
        else:
            assert 'the size of x_r should be 16'

        # s1 = nn.functional.kl_div(self.down_r(x_r.permute(0, 2, 1).view(B, embed, int(math.sqrt(t)), int(math.sqrt(t)))), z_f, log_target=True)
        # s2 = nn.functional.kl_div(self.down_i(x_i.permute(0, 2, 1).view(B, embed, int(math.sqrt(t)), int(math.sqrt(t)))), z_f, log_target=True)
        return s1, s2

def build_pre_classifier(cfg, embed):
    return pre_classifier(embed=embed)

def xcorr_depthwise(x, kernel):
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.reshape(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.reshape(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.reshape(batch, channel, out.size(2), out.size(3))
    return out

def xcorr_fast(x, kernel):
    batch = kernel.size(0)
    # channel = kernel.size(1)
    kernel = kernel.reshape(-1, x.size(1), kernel.size(2), kernel.size(3))
    x = x.reshape(1, -1, x.size(2), x.size(3))
    out = F.conv2d(x, kernel, groups=batch)
    out = out.reshape(batch, -1, out.size(2), out.size(3))
    return out