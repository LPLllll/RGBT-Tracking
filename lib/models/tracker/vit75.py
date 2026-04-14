""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

Modified by Botao Ye
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.tracker.base_backbone import BaseBackbone
from lib.models.tracker.utils import combine_tokens, recover_tokens
from lib.models.tracker.utils import token2feature, feature2token
from lib.models.tracker.pre_classifier import pre_classifier
from lib.utils import TensorDict
import numpy as np
import copy
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import gc
from PIL import Image
import cv2
def get_l2(f):
    c, h, w = f.shape
    mean = torch.mean(f, dim=0).unsqueeze(0).repeat(c, 1, 1)
    f = torch.sum(torch.pow((f - mean), 2), dim=0) / c
    f = f
    return (f - f.min()) / (f.max() - f.min())

def draw(f, name):
    plt.imshow(get_l2(f).detach().cpu().numpy())
    plt.title(name)
    plt.show()

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, return_attention=False):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         if return_attention:
#             return x, attn
#         return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, filter, eps=1e-6):
        B, N, _ = filter.size()
        B, H, N, N = attn.size()
        # filter = filter[:,:,0]
        attn_policy = filter.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        # eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        # attn_policy = attn_policy + (1.0 - attn_policy) * eye # all the tokens interact with the selected tokens

        # zero = torch.zeros((B, 1, N, N))
        attn_policy = ((attn_policy.expand(B, 1, N, N) + attn_policy.permute(0,1,3,2).expand(B, 1, N, N))>1).float()

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        # attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn1 = attn.to(torch.float32).exp() * attn_policy.to(torch.float32)
        attn2 = attn.to(torch.float32).exp() * (1 - attn_policy.to(torch.float32))
        attn1 = (attn1 + eps/N) / (attn1.sum(dim=-1, keepdim=True) + eps)
        attn2 = (attn2 + eps/N) / (attn2.sum(dim=-1, keepdim=True) + eps)
        return attn1.type_as(max_att), attn2.type_as(max_att)

    def forward(self, x, flag=False, filter=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if filter is None:
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            attn1, attn2 = self.softmax_with_policy(attn, filter)
            x = (attn1 @ v).transpose(1, 2).reshape(B, N, C) + (attn2 @ v).transpose(1, 2).reshape(B, N, C)

        # x = (attn1 @ v).transpose(1, 2).reshape(B, N, C) + (attn2 @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if flag:
            return x, attn
        return x

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_2 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/3), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/3):int(2*C/3), :, :].contiguous()
        x1 = self.conv0_1(x1)
        x2 = x[:, int(2*C / 3):, :, :].contiguous()
        x2 = self.conv0_2(x2)
        x0 = self.fovea(x0) + x1 + x2

        return self.conv1x1(x0)

class Prompt_block_init(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block_init, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        return self.conv1x1(x0)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少，因此需要一个比例系数ratio进行缩放
        """
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class TokenAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TokenAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Prompt_cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7, smooth=True):
        super(Prompt_cbam_block, self).__init__()
        self.spatialattention0 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention0 = TokenAttention(kernel_size=kernel_size)
        self.spatialattention1 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention1 = TokenAttention(kernel_size=kernel_size)
        self.spatialattention2 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention2 = TokenAttention(kernel_size=kernel_size)
        self.fovea = Fovea(smooth=smooth)

    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 3), :, :].contiguous()
        x1 = x[:, int(C / 3):int(2*C / 3), :, :].contiguous()
        x2 = x[:, int(2*C / 3):, :, :].contiguous()
        x0 = x0 * self.spatialattention0(x0)
        x0 = x0 * self.tokenattention0(x0)
        x1 = x1 * self.spatialattention1(x1)
        x1 = x1 * self.tokenattention1(x1)
        x2 = x2 * self.spatialattention1(x2)
        x2 = x2 * self.tokenattention2(x2)
        x0 = self.fovea(x0) + x1 + x2
        return x0

class Prompt_cbam_block_init(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7, smooth=True):
        super(Prompt_cbam_block_init, self).__init__()
        self.spatialattention0 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention0 = TokenAttention(kernel_size=kernel_size)
        self.spatialattention1 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention1 = TokenAttention(kernel_size=kernel_size)
        self.fovea = Fovea(smooth=smooth)

    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x0 = x0 * self.spatialattention0(x0)
        x0 = x0 * self.tokenattention0(x0)
        x1 = x1 * self.spatialattention1(x1)
        x1 = x1 * self.tokenattention1(x1)
        x0 = self.fovea(x0) + x1
        return x0



def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, remained, lens_t: int, keep_ratio: float, box_mask_z: torch.Tensor, flag=0):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = 256 #attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = torch.ceil(keep_ratio * lens_s)
    if (keep_ratio == torch.ones_like(keep_ratio).float().cuda()).sum() == bs:
        return tokens, torch.ones_like(bs), None

    attn_t = attn[:, :, :lens_t, lens_t+lens_s*(flag):lens_t+lens_s*(flag+1)]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    # topk_attn, topk_idx = sorted_attn[:, :int(lens_keep)], indices[:, :int(lens_keep)]
    # non_topk_attn, non_topk_idx = sorted_attn[:, int(lens_keep):], indices[:, int(lens_keep):]


    # topk_idx = None
    # topk_attn = None
    # non_topk_attn = None
    # non_topk_idx = None
    # topk_idx = torch.cat((a, indices[i, :int(lens_keep[i])].unsqueeze(-1)), dim=0) for i in range(bs)
    # topk_idx = torch.cat((a, indices[i, :int(lens_keep[i])].unsqueeze(-1)), dim=0) for i in range(bs)
    # topk_idx = torch.zeros_like(indices).cuda()
    # topk_attn = torch.zeros_like(sorted_attn).cuda()
    # non_topk_idx = torch.zeros_like(indices).cuda()
    # non_topk_attn = torch.zeros_like(sorted_attn).cuda()
    C = tokens.shape[2]
    for i in range(bs):
        # topk_idx[i, :int(lens_keep[i])] = indices[i, :int(lens_keep[i])].unsqueeze(0)
        topk_idx= indices[i, :int(lens_keep[i])].unsqueeze(0)
        # topk_attn[i, :int(lens_keep[i])] = sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)
        # non_topk_idx[i, :lens_s-int(lens_keep[i])] = indices[i, int(lens_keep[i]):].unsqueeze(0)
        non_topk_idx= indices[i, int(lens_keep[i]):].unsqueeze(0)
        # non_topk_attn[i, :lens_s-int(lens_keep[i])] = sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)
        pruned_lens_x = lens_s - int(lens_keep[i])
        pad_x = torch.zeros([1, pruned_lens_x, C], device=tokens.device)
        index_all = torch.cat([topk_idx, non_topk_idx], dim=1)
        x_i = torch.cat([topk_idx.unsqueeze(-1).expand(1, -1, C).to(torch.int64), pad_x], dim=1)
        # recover original token order
        # C = x_i.shape[-1]
        # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
        x_i = torch.zeros_like(x_i).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(1, -1, C).to(torch.int64),
                                             src=x_i)
        if i == 0:
            attentive = x_i
        else:
            attentive = torch.cat((attentive, x_i), dim=0)

    # for i in range(bs):
    #     if i == 0:
    #
    #         topk_idx = indices[i, :int(lens_keep[i])].unsqueeze(0)
    #         non_topk_idx = indices[i, int(lens_keep[i]):].unsqueeze(0)
    #     else:
    #         topk_idx = torch.cat((topk_idx, indices[i, :int(lens_keep[i])].unsqueeze(0)), dim=0)
    #         non_topk_idx = torch.cat((non_topk_idx, indices[i, int(lens_keep[i]):].unsqueeze(0)), dim=0)
    # for i in range(bs):
    #     if i == 0:
    #         topk_attn = sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)
    #         non_topk_attn = sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)
    #     else:
    #
    #         topk_attn = torch.cat((topk_attn, sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)), dim=0)
    #         non_topk_attn = torch.cat((non_topk_attn, sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)), dim=0)


    # keep_index = global_index.gather(dim=1, index=topk_idx)
    # removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    # B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    # attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C)) * (topk_idx.unsqueeze(-1).expand(B, -1, C) > 0)

    attentive_tokens = tokens_s * (attentive > 0) * remained
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, lens_keep, (attentive > 0) * remained

def candidate_elimination_new(attn: torch.Tensor, tokens_rgb: torch.Tensor, tokens_tir: torch.Tensor, remained_rgb, remained_tir, lens_t: int, keep_ratio, box_mask_z: torch.Tensor, flag=0, pos_embed=None):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = 256 * 2 #attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    # keep_ratio = attn[:, :, 0, (1+lens_t):]
    # keep_ratio = keep_ratio.mean(dim=2).mean(dim=1)

    # attn = attn[:, :, 1:, 1:]

    lens_keep = torch.ceil(keep_ratio * lens_s)
    if (keep_ratio == torch.ones_like(keep_ratio).float().cuda()).sum() == bs:
        return tokens_rgb, lens_keep, torch.ones_like(bs), tokens_tir, lens_keep, torch.ones_like(bs)

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    # topk_attn, topk_idx = sorted_attn[:, :int(lens_keep)], indices[:, :int(lens_keep)]
    # non_topk_attn, non_topk_idx = sorted_attn[:, int(lens_keep):], indices[:, int(lens_keep):]


    # topk_idx = None
    # topk_attn = None
    # non_topk_attn = None
    # non_topk_idx = None
    # topk_idx = torch.cat((a, indices[i, :int(lens_keep[i])].unsqueeze(-1)), dim=0) for i in range(bs)
    # topk_idx = torch.cat((a, indices[i, :int(lens_keep[i])].unsqueeze(-1)), dim=0) for i in range(bs)
    # topk_idx = torch.zeros_like(indices).cuda()
    # topk_attn = torch.zeros_like(sorted_attn).cuda()
    # non_topk_idx = torch.zeros_like(indices).cuda()
    # non_topk_attn = torch.zeros_like(sorted_attn).cuda()
    C = tokens_rgb.shape[2]
    for i in range(bs):
        # topk_idx[i, :int(lens_keep[i])] = indices[i, :int(lens_keep[i])].unsqueeze(0)
        topk_idx= indices[i, :int(lens_keep[i])].unsqueeze(0)
        # topk_attn[i, :int(lens_keep[i])] = sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)
        # non_topk_idx[i, :lens_s-int(lens_keep[i])] = indices[i, int(lens_keep[i]):].unsqueeze(0)
        non_topk_idx= indices[i, int(lens_keep[i]):].unsqueeze(0)
        # non_topk_attn[i, :lens_s-int(lens_keep[i])] = sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)
        pruned_lens_x = lens_s - int(lens_keep[i])
        pad_x = torch.zeros([1, pruned_lens_x, C], device=tokens_rgb.device)
        index_all = torch.cat([topk_idx, non_topk_idx], dim=1)
        x_i = torch.cat([topk_idx.unsqueeze(-1).expand(1, -1, C).to(torch.int64), pad_x], dim=1)
        # recover original token order
        # C = x_i.shape[-1]
        # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
        x_i = torch.zeros_like(x_i).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(1, -1, C).to(torch.int64),
                                             src=x_i)
        if i == 0:
            attentive = x_i
        else:
            attentive = torch.cat((attentive, x_i), dim=0)

    # for i in range(bs):
    #     if i == 0:
    #
    #         topk_idx = indices[i, :int(lens_keep[i])].unsqueeze(0)
    #         non_topk_idx = indices[i, int(lens_keep[i]):].unsqueeze(0)
    #     else:
    #         topk_idx = torch.cat((topk_idx, indices[i, :int(lens_keep[i])].unsqueeze(0)), dim=0)
    #         non_topk_idx = torch.cat((non_topk_idx, indices[i, int(lens_keep[i]):].unsqueeze(0)), dim=0)
    # for i in range(bs):
    #     if i == 0:
    #         topk_attn = sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)
    #         non_topk_attn = sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)
    #     else:
    #
    #         topk_attn = torch.cat((topk_attn, sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)), dim=0)
    #         non_topk_attn = torch.cat((non_topk_attn, sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)), dim=0)


    # keep_index = global_index.gather(dim=1, index=topk_idx)
    # removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t_rgb = tokens_rgb[:, :lens_t]
    tokens_t_tir = tokens_rgb[:, :lens_t]
    tokens_s_rgb = tokens_rgb[:, lens_t:]
    tokens_s_tir = tokens_rgb[:, lens_t:]

    # obtain the attentive and inattentive tokens
    # B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    # attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C)) * (topk_idx.unsqueeze(-1).expand(B, -1, C) > 0)

    attentive_tokens_rgb = tokens_s_rgb * (attentive[:,:int(lens_s/2), :] > 0) * remained_rgb + pos_embed.expand(bs,int(lens_s/2), C) * (1 - (attentive[:,:int(lens_s/2), :] > 0) * remained_rgb)
    attentive_tokens_tir = tokens_s_tir * (attentive[:, int(lens_s/2):, :] > 0) * remained_tir + pos_embed.expand(bs,int(lens_s/2), C) * (1 - (attentive[:,int(lens_s/2):, :] > 0) * remained_tir)
    tokens_new_rgb = torch.cat([tokens_t_rgb, attentive_tokens_rgb], dim=1)
    tokens_new_tir = torch.cat([tokens_t_tir, attentive_tokens_tir], dim=1)

    return tokens_new_rgb, lens_keep, (attentive[:,:int(lens_s/2), :] > 0) * remained_rgb, tokens_new_tir, lens_keep, (attentive[:, int(lens_s/2):, :] > 0) * remained_tir


def candidate_elimination_new2(attn: torch.Tensor, tokens_rgb: torch.Tensor, tokens_tir: torch.Tensor, remained_rgb, remained_tir, lens_t: int, keep_ratio, box_mask_z: torch.Tensor, flag=0, pos_embed=None):

    lens_s = 256 * 2 #attn.shape[-1] - lens_t
    bs, _, _ = attn.shape

    # keep_ratio = attn[:, :, 0, (1+lens_t):]
    # keep_ratio = keep_ratio.mean(dim=2).mean(dim=1)

    # attn = attn[:, :, 1:, 1:]

    lens_keep = torch.ceil(keep_ratio * lens_s)
    if (keep_ratio == torch.ones_like(keep_ratio).float().cuda()).sum() == bs:
        return tokens_rgb, lens_keep, torch.ones_like(bs), tokens_tir, lens_keep, torch.ones_like(bs)

    attn=attn.squeeze(-1)
    sorted_attn, indices = torch.sort(attn, dim=1, descending=True)

    C = tokens_rgb.shape[2]
    for i in range(bs):
        # topk_idx[i, :int(lens_keep[i])] = indices[i, :int(lens_keep[i])].unsqueeze(0)
        topk_idx= indices[i, :int(lens_keep[i])].unsqueeze(0)
        # topk_attn[i, :int(lens_keep[i])] = sorted_attn[i, :int(lens_keep[i])].unsqueeze(0)
        # non_topk_idx[i, :lens_s-int(lens_keep[i])] = indices[i, int(lens_keep[i]):].unsqueeze(0)
        non_topk_idx= indices[i, int(lens_keep[i]):].unsqueeze(0)
        # non_topk_attn[i, :lens_s-int(lens_keep[i])] = sorted_attn[i, int(lens_keep[i]):].unsqueeze(0)
        pruned_lens_x = lens_s - int(lens_keep[i])
        pad_x = torch.zeros([1, pruned_lens_x, C], device=tokens_rgb.device)
        index_all = torch.cat([topk_idx, non_topk_idx], dim=1)
        x_i = torch.cat([topk_idx.unsqueeze(-1).expand(1, -1, C).to(torch.int64), pad_x], dim=1)
        # recover original token order
        # C = x_i.shape[-1]
        # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
        x_i = torch.zeros_like(x_i).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(1, -1, C).to(torch.int64),
                                             src=x_i)
        if i == 0:
            attentive = x_i
        else:
            attentive = torch.cat((attentive, x_i), dim=0)

    # separate template and search tokens
    tokens_t_rgb = tokens_rgb[:, :lens_t]
    tokens_t_tir = tokens_rgb[:, :lens_t]
    tokens_s_rgb = tokens_rgb[:, lens_t:]
    tokens_s_tir = tokens_rgb[:, lens_t:]

    attentive_tokens_rgb = tokens_s_rgb * (attentive[:,:int(lens_s/2), :] > 0) * remained_rgb + pos_embed.expand(bs,int(lens_s/2), C) * (1 - (attentive[:,:int(lens_s/2), :] > 0) * remained_rgb)
    attentive_tokens_tir = tokens_s_tir * (attentive[:, int(lens_s/2):, :] > 0) * remained_tir + pos_embed.expand(bs,int(lens_s/2), C) * (1 - (attentive[:,int(lens_s/2):, :] > 0) * remained_tir)
    tokens_new_rgb = torch.cat([tokens_t_rgb, attentive_tokens_rgb], dim=1)
    tokens_new_tir = torch.cat([tokens_t_tir, attentive_tokens_tir], dim=1)

    return tokens_new_rgb, lens_keep, (attentive[:,:int(lens_s/2), :] > 0) * remained_rgb, tokens_new_tir, lens_keep, (attentive[:, int(lens_s/2):, :] > 0) * remained_tir


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, filter=None):
        if return_attention:
            if filter is None:
                feat, attn = self.attn(self.norm1(x), True)
            else:
                feat, attn = self.attn(self.norm1(x), True, filter)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            if filter is None:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), filter=filter))# * filter.expand(x.size(0), x.size(1), x.size(2))
                x = x + self.drop_path(self.mlp(self.norm2(x)))# * filter.expand(x.size(0), x.size(1), x.size(2))
            return x

class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class VisionTransformerMPLT(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 mplt_loc=None, mplt_drop_path=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = 64
        self.num_patches_search = 256

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        prompt_blocks = []
        prompt_blocks_rev = []
        block_nums = depth
        for i in range(block_nums):
            prompt_blocks.append(Prompt_cbam_block(channel=embed_dim, ratio=8, kernel_size=7, smooth=True))
            prompt_blocks_rev.append(Prompt_cbam_block(channel=embed_dim, ratio=8, kernel_size=7, smooth=True))
        self.prompt_blocks = nn.Sequential(*prompt_blocks)
        self.prompt_blocks_rev = nn.Sequential(*prompt_blocks_rev)
        self.prompt_blocks_init = Prompt_cbam_block_init(channel=embed_dim, ratio=8, kernel_size=7, smooth=True)
        self.prompt_blocks_init_rev = Prompt_cbam_block_init(channel=embed_dim, ratio=8, kernel_size=7, smooth=True)

        prompt_norms = []
        prompt_norms_rev = []
        for i in range(block_nums):
            prompt_norms.append(norm_layer(embed_dim))
            prompt_norms_rev.append(norm_layer(embed_dim))
        self.prompt_norms = nn.Sequential(*prompt_norms)
        self.prompt_norms_rev = nn.Sequential(*prompt_norms_rev)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x):
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_v = self.patch_embed(x[0])
        z_v = self.patch_embed(z[0])
        x_i = self.patch_embed(x[1])
        z_i = self.patch_embed(z[1])
        z_feat_rgb = token2feature(self.prompt_norms[0](z_v))
        x_feat_rgb = token2feature(self.prompt_norms[0](x_v))
        z_dte_feat = token2feature(self.prompt_norms[0](z_i))
        x_dte_feat = token2feature(self.prompt_norms[0](x_i))
        z_feat = torch.cat([z_feat_rgb, z_dte_feat], dim=1)
        x_feat = torch.cat([x_feat_rgb, x_dte_feat], dim=1)
        z_feat_rev = torch.cat([z_dte_feat, z_feat_rgb], dim=1)
        x_feat_rev = torch.cat([x_dte_feat, x_feat_rgb], dim=1)
        z_feat = self.prompt_blocks_init(z_feat)
        x_feat = self.prompt_blocks_init(x_feat)
        x_feat_rev = self.prompt_blocks_init_rev(x_feat_rev)
        z_feat_rev = self.prompt_blocks_init_rev(z_feat_rev)
        z_tokens = feature2token(z_feat)
        x_tokens = feature2token(x_feat)
        z_tokens_rev = feature2token(z_feat_rev)
        x_tokens_rev = feature2token(x_feat_rev)
        z_prompted, x_prompted = z_tokens, x_tokens
        z_prompted_rev, x_prompted_rev = z_tokens_rev, x_tokens_rev
        z_v = z_v + z_tokens
        x_v = x_v + x_tokens
        z_i = z_i + z_tokens_rev
        x_i = x_i + x_tokens_rev
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x

        z_i += self.pos_embed_z
        x_i += self.pos_embed_x


        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x_v = self.pos_drop(x_v)
        x_i = self.pos_drop(x_i)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        # lens_z = 64
        # lens_x = 256

        mplt_index = 0
        for i, blk in enumerate(self.blocks):
            if i >= 1:
                # prompt
                x_ori = x_v
                x_ori_rev = x_i
                x_v = self.prompt_norms[i - 1](x_v)  # todo
                x_i = self.prompt_norms_rev[i - 1](x_i)  # todo
                z_tokens = x_v[:, :lens_z, :]
                x_tokens = x_v[:, lens_z:, :]
                z_tokens_rev = x_i[:, :lens_z, :]
                x_tokens_rev = x_i[:, lens_z:, :]
                z_feat_rgb = token2feature(z_tokens)
                x_feat_rgb = token2feature(x_tokens)
                z_feat_rev = token2feature(z_tokens_rev)
                x_feat_rev = token2feature(x_tokens_rev)

                z_prompted = self.prompt_norms[i](z_prompted)
                x_prompted = self.prompt_norms[i](x_prompted)
                z_prompted_rev = self.prompt_norms_rev[i](z_prompted_rev)
                x_prompted_rev = self.prompt_norms_rev[i](x_prompted_rev)
                z_prompt_feat = token2feature(z_prompted)
                x_prompt_feat = token2feature(x_prompted)
                z_prompt_feat_rev = token2feature(z_prompted_rev)
                x_prompt_feat_rev = token2feature(x_prompted_rev)

                z_feat = torch.cat([z_feat_rgb, z_prompt_feat, z_feat_rev], dim=1)
                x_feat = torch.cat([x_feat_rgb, x_prompt_feat, x_feat_rev], dim=1)
                z_feat_rev = torch.cat([z_feat_rev, z_prompt_feat_rev, z_feat_rgb], dim=1)
                x_feat_rev = torch.cat([x_feat_rev, x_prompt_feat_rev, x_feat_rgb], dim=1)
                z_feat = self.prompt_blocks[i](z_feat)
                x_feat = self.prompt_blocks[i](x_feat)
                z_feat_rev = self.prompt_blocks_rev[i](z_feat_rev)
                x_feat_rev = self.prompt_blocks_rev[i](x_feat_rev)

                z_v = feature2token(z_feat)
                x_v = feature2token(x_feat)
                z_i = feature2token(z_feat_rev)
                x_i = feature2token(x_feat_rev)
                z_prompted, x_prompted = z_v, x_v
                z_prompted_rev, x_prompted_rev = z_i, x_i
                x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
                x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)
                x_v = x_ori + x_v
                x_i = x_ori_rev + x_i
            x_v = blk(x_v)
            x_i = blk(x_i)

        x_v = recover_tokens(x_v, lens_z, lens_x, mode=self.cat_mode)
        x_i = recover_tokens(x_i, lens_z, lens_x, mode=self.cat_mode)
        x = torch.cat([x_v, x_i], dim=1)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


class VisionTransformerBase(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 mplt_loc=None, mplt_drop_path=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = 64
        self.num_patches_search = 256

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # prompt_blocks = []
        # blocks = []
        # prompt_blocks_rev = []
        # block_nums = depth
        # for i in range(block_nums):
        #     blocks.append(Block(channel=embed_dim, ratio=8, kernel_size=7, smooth=True))
        #     # prompt_blocks_rev.append(Prompt_cbam_block(channel=embed_dim, ratio=8, kernel_size=7, smooth=True))
        # self.blocks = nn.Sequential(*blocks)
        # self.prompt_blocks_rev = nn.Sequential(*prompt_blocks_rev)
        self.base_blocks_pre_rgb_z = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_tir_z = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_rgb_x = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_tir_x = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        # self.prompt_blocks_init_rev = Prompt_cbam_block_init(channel=embed_dim, ratio=8, kernel_size=7, smooth=True)

        # prompt_norms = []
        # prompt_norms_rev = []
        # for i in range(block_nums):
        #     prompt_norms.append(norm_layer(embed_dim))
        #     prompt_norms_rev.append(norm_layer(embed_dim))
        # self.prompt_norms = nn.Sequential(*prompt_norms)
        # self.prompt_norms_rev = nn.Sequential(*prompt_norms_rev)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.init_weights(weight_init)
        self.base_pre_norm_rgb = norm_layer(embed_dim)
        self.base_pre_norm_tir = norm_layer(embed_dim)

        self.base_fuse_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer, act_layer=act_layer)
        # self.base_down_mlp = nn.Linear()

    def forward_features(self, z, x):

        with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
            f.writelines('start forward\n')
            f.close()
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_v = self.patch_embed(x[0])
        z_v = self.patch_embed(z[0])
        x_i = self.patch_embed(x[1])
        z_i = self.patch_embed(z[1])
        # z_feat_rgb = token2feature(self.base_pre_norm_rgb(z_v))
        # x_feat_rgb = token2feature(self.base_pre_norm_rgb(x_v))
        # z_dte_feat = token2feature(self.base_pre_norm_tir(z_i))
        # x_dte_feat = token2feature(self.base_pre_norm_tir(x_i))
        z_feat_rgb = self.base_pre_norm_rgb(z_v)
        x_feat_rgb = self.base_pre_norm_rgb(x_v)
        z_dte_feat = self.base_pre_norm_tir(z_i)
        x_dte_feat = self.base_pre_norm_tir(x_i)
        # z_feat = torch.cat([z_feat_rgb, z_dte_feat], dim=1)
        # x_feat = torch.cat([x_feat_rgb, x_dte_feat], dim=1)
        # z_feat_rev = torch.cat([z_dte_feat, z_feat_rgb], dim=1)
        # x_feat_rev = torch.cat([x_dte_feat, x_feat_rgb], dim=1)
        z_feat = self.base_blocks_pre_rgb_z(z_feat_rgb)
        x_feat = self.base_blocks_pre_rgb_x(x_feat_rgb)
        z_feat_dte = self.base_blocks_pre_tir_z(z_dte_feat)
        x_feat_dte = self.base_blocks_pre_tir_x(x_dte_feat)
        # z_tokens = feature2token(z_feat)
        # z_tokens_dte = feature2token(z_feat_dte)
        # x_tokens = feature2token(x_feat)
        # x_tokens_dte = feature2token(x_feat_dte)
        # z_tokens_rev = feature2token(z_feat_rev)
        # x_tokens_rev = feature2token(x_feat_rev)
        # z_prompted, x_prompted = z_tokens, x_tokens
        # z_prompted_rev, x_prompted_rev = z_tokens_rev, x_tokens_rev
        z_v = z_v + z_feat
        x_v = x_v + x_feat
        z_i = z_i + z_feat_dte
        x_i = x_i + x_feat_dte


        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x
        z_i += self.pos_embed_z
        x_i += self.pos_embed_x

        # if self.add_sep_seg:
        #     x += self.search_segment_pos_embed
        #     z += self.template_segment_pos_embed

        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)

        # if self.add_cls_token:
        #     x = torch.cat([cls_tokens, x], dim=1)

        # x_v = self.pos_drop(x_v)
        # z_v = self.pos_drop(z_v)
        # x_i = self.pos_drop(x_i)
        # z_i = self.pos_drop(z_i)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        # lens_z = 64
        # lens_x = 256

        mplt_index = 0
        for i, blk in enumerate(self.blocks):
            x_v, att_r = blk(x_v, True)
            x_i, att_i = blk(x_i, True)
            # if i >= 1:
                # prompt
                # x_ori = x_v
                # x_ori_rev = x_i
                # x_v = self.prompt_norms[i - 1](x_v)  # todo
                # x_i = self.prompt_norms_rev[i - 1](x_i)  # todo
                # z_tokens = x_v[:, :lens_z, :]
                # x_tokens = x_v[:, lens_z:, :]
                # z_tokens_rev = x_i[:, :lens_z, :]
                # x_tokens_rev = x_i[:, lens_z:, :]
                # z_feat_rgb = x_v[:, :lens_z, :]
                # x_feat_rgb = x_v[:, lens_z:, :]
                # z_feat_dte = x_i[:, :lens_z, :]
                # x_feat_dte = x_i[:, lens_z:, :]

                # z_prompted = self.prompt_norms[i](z_prompted)
                # x_prompted = self.prompt_norms[i](x_prompted)
                # z_prompted_rev = self.prompt_norms_rev[i](z_prompted_rev)
                # x_prompted_rev = self.prompt_norms_rev[i](x_prompted_rev)
                # z_prompt_feat = token2feature(z_prompted)
                # x_prompt_feat = token2feature(x_prompted)
                # z_prompt_feat_rev = token2feature(z_prompted_rev)
                # x_prompt_feat_rev = token2feature(x_prompted_rev)

                # v_feat = torch.cat([z_feat_rgb, x_feat_rgb], dim=1)
                # i_feat = torch.cat([z_feat_dte, x_feat_dte], dim=1)
                # v_feat, v_att = self.blocks[i](v_feat, True)
                # i_feat, i_att = self.blocks[i](i_feat, True)


                # z_v = feature2token(z_feat)
                # x_v = feature2token(x_feat)
                # z_i = feature2token(z_feat_rev)
                # x_i = feature2token(x_feat_rev)
                # z_prompted, x_prompted = z_v, x_v
                # z_prompted_rev, x_prompted_rev = z_i, x_i
                # x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
                # x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)
                # x_v = x_ori + x_v
                # x_i = x_ori_rev + x_i
            # x_v, v_att = blk(v_feat, True)
            # x_v = blk(x_v)
            # # x_i, i_att = blk(i_feat, True)
            # x_i = blk(x_i)

        x_v = recover_tokens(x_v, lens_z, lens_x, mode=self.cat_mode)
        x_i = recover_tokens(x_i, lens_z, lens_x, mode=self.cat_mode)
        # x = torch.cat([x_v, x_i], dim=1)

        x = self.base_fuse_block(x_v + x_i)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


class VisionTransformerPre(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 mplt_loc=None, mplt_drop_path=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = 64
        self.num_patches_search = 256

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.base_blocks_pre_rgb_z = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_tir_z = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_rgb_x = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_tir_x = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.init_weights(weight_init)
        self.base_pre_norm_rgb = norm_layer(embed_dim)
        self.base_pre_norm_tir = norm_layer(embed_dim)

        self.base_fuse_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer, act_layer=act_layer)

        self.pre_classifier = pre_classifier(embed=embed_dim)

    def forward_features(self, z, x, kwards):

        # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
        #     f.writelines('start forward\n')
        #     f.close()
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_v = self.patch_embed(x[0])
        z_v = self.patch_embed(z[0])
        x_i = self.patch_embed(x[1])
        z_i = self.patch_embed(z[1])
        if (kwards['change_v'] is not None) and (kwards['change_i'] is not None):
            ssr, ssi, csr, csi = self.pre_classifier(z_v, z_i, x_v, x_i)
            return [], TensorDict({'ssr': ssr, 'ssi': ssi, 'csr': csr, 'csi': csi})
        # z_feat_rgb = token2feature(self.base_pre_norm_rgb(z_v))
        # x_feat_rgb = token2feature(self.base_pre_norm_rgb(x_v))
        # z_dte_feat = token2feature(self.base_pre_norm_tir(z_i))
        # x_dte_feat = token2feature(self.base_pre_norm_tir(x_i))
        z_feat_rgb = self.base_pre_norm_rgb(z_v)
        x_feat_rgb = self.base_pre_norm_rgb(x_v)
        z_dte_feat = self.base_pre_norm_tir(z_i)
        x_dte_feat = self.base_pre_norm_tir(x_i)
        # z_feat = torch.cat([z_feat_rgb, z_dte_feat], dim=1)
        # x_feat = torch.cat([x_feat_rgb, x_dte_feat], dim=1)
        # z_feat_rev = torch.cat([z_dte_feat, z_feat_rgb], dim=1)
        # x_feat_rev = torch.cat([x_dte_feat, x_feat_rgb], dim=1)
        z_feat = self.base_blocks_pre_rgb_z(z_feat_rgb)
        x_feat = self.base_blocks_pre_rgb_x(x_feat_rgb)
        z_feat_dte = self.base_blocks_pre_tir_z(z_dte_feat)
        x_feat_dte = self.base_blocks_pre_tir_x(x_dte_feat)
        # z_tokens = feature2token(z_feat)
        # z_tokens_dte = feature2token(z_feat_dte)
        # x_tokens = feature2token(x_feat)
        # x_tokens_dte = feature2token(x_feat_dte)
        # z_tokens_rev = feature2token(z_feat_rev)
        # x_tokens_rev = feature2token(x_feat_rev)
        # z_prompted, x_prompted = z_tokens, x_tokens
        # z_prompted_rev, x_prompted_rev = z_tokens_rev, x_tokens_rev
        z_v = z_v + z_feat
        x_v = x_v + x_feat
        z_i = z_i + z_feat_dte
        x_i = x_i + x_feat_dte


        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x
        z_i += self.pos_embed_z
        x_i += self.pos_embed_x

        # if self.add_sep_seg:
        #     x += self.search_segment_pos_embed
        #     z += self.template_segment_pos_embed

        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)

        # if self.add_cls_token:
        #     x = torch.cat([cls_tokens, x], dim=1)

        # x_v = self.pos_drop(x_v)
        # z_v = self.pos_drop(z_v)
        # x_i = self.pos_drop(x_i)
        # z_i = self.pos_drop(z_i)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        # lens_z = 64
        # lens_x = 256

        mplt_index = 0
        for i, blk in enumerate(self.blocks):
            x_v = blk(x_v)
            x_i = blk(x_i)
            # if i >= 1:
                # prompt
                # x_ori = x_v
                # x_ori_rev = x_i
                # x_v = self.prompt_norms[i - 1](x_v)  # todo
                # x_i = self.prompt_norms_rev[i - 1](x_i)  # todo
                # z_tokens = x_v[:, :lens_z, :]
                # x_tokens = x_v[:, lens_z:, :]
                # z_tokens_rev = x_i[:, :lens_z, :]
                # x_tokens_rev = x_i[:, lens_z:, :]
                # z_feat_rgb = x_v[:, :lens_z, :]
                # x_feat_rgb = x_v[:, lens_z:, :]
                # z_feat_dte = x_i[:, :lens_z, :]
                # x_feat_dte = x_i[:, lens_z:, :]

                # z_prompted = self.prompt_norms[i](z_prompted)
                # x_prompted = self.prompt_norms[i](x_prompted)
                # z_prompted_rev = self.prompt_norms_rev[i](z_prompted_rev)
                # x_prompted_rev = self.prompt_norms_rev[i](x_prompted_rev)
                # z_prompt_feat = token2feature(z_prompted)
                # x_prompt_feat = token2feature(x_prompted)
                # z_prompt_feat_rev = token2feature(z_prompted_rev)
                # x_prompt_feat_rev = token2feature(x_prompted_rev)

                # v_feat = torch.cat([z_feat_rgb, x_feat_rgb], dim=1)
                # i_feat = torch.cat([z_feat_dte, x_feat_dte], dim=1)
                # v_feat, v_att = self.blocks[i](v_feat, True)
                # i_feat, i_att = self.blocks[i](i_feat, True)


                # z_v = feature2token(z_feat)
                # x_v = feature2token(x_feat)
                # z_i = feature2token(z_feat_rev)
                # x_i = feature2token(x_feat_rev)
                # z_prompted, x_prompted = z_v, x_v
                # z_prompted_rev, x_prompted_rev = z_i, x_i
                # x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
                # x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)
                # x_v = x_ori + x_v
                # x_i = x_ori_rev + x_i
            # x_v, v_att = blk(v_feat, True)
            # x_v = blk(x_v)
            # # x_i, i_att = blk(i_feat, True)
            # x_i = blk(x_i)

        x_v = recover_tokens(x_v, lens_z, lens_x, mode=self.cat_mode)
        x_i = recover_tokens(x_i, lens_z, lens_x, mode=self.cat_mode)
        # x = torch.cat([x_v, x_i], dim=1)

        x = self.base_fuse_block(x_v + x_i)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


class P_module(nn.Module):
    def __init__(self, embed=768):
        super(P_module, self).__init__()
        self.nn = nn.Parameter(torch.zeros(1, 1, embed))
    def get(self):
        return self.nn

class Predictor(nn.Module):
    def __init__(self, embed=712, outbed=2):
        super(Predictor, self).__init__()

        self.mlp =nn.Sequential(
            nn.LayerNorm(embed),
            nn.Linear(embed, int(embed/4)),
            nn.LayerNorm(int(embed/4)),
            nn.Linear(int(embed/4), outbed)
        )
        # self.iner = 512
        self.tr_rgb = nn.Sequential(
           nn.LayerNorm(embed),
           nn.Linear(embed, embed)
        )
        self.tr_tir = nn.Sequential(
           nn.LayerNorm(embed),
           nn.Linear(embed, embed)
        )
    def forward(self, rgb, tir):
        # x = self.mlp(x)
        x = self.mlp(torch.cat((self.tr_rgb(rgb), self.tr_tir(tir)), dim=1))
        return x
        # x = x[:, 64:, :]#.permute(0, 2, 1)
        # return x, self.impor(x.permute(0, 2, 1))


    # def forward(self):
class VisionTransformerIner(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 mplt_loc=None, mplt_drop_path=None, flag=True, epoch=20):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.ce_start = epoch
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = 64
        self.num_patches_search = 256

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.base_blocks_pre_rgb_z = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_tir_z = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_rgb_x = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.base_blocks_pre_tir_x = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.init_weights(weight_init)
        self.base_pre_norm_rgb = norm_layer(embed_dim)
        self.base_pre_norm_tir = norm_layer(embed_dim)

        self.base_fuse_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer, act_layer=act_layer)

        # self.pre_classifier = pre_classifier(embed=embed_dim)

        # self.selection_ratio = torch.ones((3,))

        # self.iner_ada_selection = nn.Sequential(
        #     nn.Linear(in_features=int(embed_dim), out_features=int(embed_dim/2)),
        #     nn.Linear(in_features=int(embed_dim/2), out_features=int(embed_dim/8)),
        #     nn.Linear(in_features=int(embed_dim/8), out_features=int(embed_dim/16)),
        #     nn.Linear(in_features=int(embed_dim/16), out_features=1),
        # )
        #
        #
        # self.iner_ada_selection2 = nn.Sequential(
        #     nn.Linear(in_features=int(640), out_features=int(320)),
        #     nn.Linear(in_features=int(320), out_features=int(160)),
        #     nn.Linear(in_features=int(160), out_features=int(80)),
        #     nn.Linear(in_features=int(80), out_features=1),
        # )

        self.selection_places = mplt_loc
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.selection_ratio_tokens = nn.Sequential(*[
        #         nn.Parameter(torch.zeros(1, 1, embed_dim))
        #     for i in range(len(mplt_loc))])

        # self.selection_ratio_tokens = []
        for i in range(len(mplt_loc)):
            # self.selection_ratio_tokens.append(nn.Parameter(torch.zeros(1, 1, embed_dim)))
            # name = 'selection_ratio_tokens_' + str(i)
            # # self.add_module(name, nn.Parameter(torch.zeros(1, 1, embed_dim)))
            # self.add_module(name, P_module(embed_dim))
            name = 'selection_ratio_predictor_' + str(i)
            # self.add_module(name, nn.Parameter(torch.z
            self.add_module(name, Predictor(embed_dim))
        # self.selection_att = Block(
        #         dim=embed_dim, num_heads=int(num_heads/2), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        #         attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer, act_layer=act_layer)
        # self.pos_embed_f = nn.Parameter(torch.zeros(1, (self.num_patches_search+self.num_patches_template), embed_dim))

        self.progressive = True

    def draw_remained(self, remained, img):

        m = np.zeros((256,256, 3))
        remained = (remained[:, :, 0]>0).view(16, 16).detach().cpu().numpy()
        for i in range(16):
            for j in range(16):
                if remained[i, j] == 1:
                    m[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16, :] = 1
        img = img[0,:,:,:].permute(1, 2, 0).detach().cpu().numpy()
        img = ((img-img.min())/(img.max()-img.min()))
        img_add = cv2.addWeighted(img, 1, 1-m, 1.0, 0, dtype=cv2.CV_32F)
        cv2.imshow('a', img_add)
        cv2.waitKey(0)
        # plt.imshow(img)
        # plt.imshow(remained, alpha=0.5)
        # plt.show()
    def draw_remained_z(self, remained, img):
        s=8
        m = np.zeros((s*16,s*16, 3))
        remained = (remained[:, :, 0]>0).view(s, s).detach().cpu().numpy()
        for i in range(s):
            for j in range(s):
                if remained[i, j] == 1:
                    m[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16, :] = 1
        img = img[0,:,:,:].permute(1, 2, 0).detach().cpu().numpy()
        img = ((img-img.min())/(img.max()-img.min()))
        img_add = cv2.addWeighted(img, 1, 1-m, 1.0, 0, dtype=cv2.CV_32F)
        cv2.imshow('a', img_add)
        cv2.waitKey(0)



    def forward_features(self, z, x, kwards):

        # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
        #     f.writelines('start forward\n')
        #     f.close()
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_v = self.patch_embed(x[0])
        z_v = self.patch_embed(z[0])
        x_i = self.patch_embed(x[1])
        z_i = self.patch_embed(z[1])
        # if (kwards['change_v'] is not None) and (kwards['change_i'] is not None):
        #     ssr, ssi, csr, csi = self.pre_classifier(z_v, z_i, x_v, x_i)
        #     return [], TensorDict({'ssr': ssr, 'ssi': ssi, 'csr': csr, 'csi': csi})

        z_feat_rgb = self.base_pre_norm_rgb(z_v)
        x_feat_rgb = self.base_pre_norm_rgb(x_v)
        z_dte_feat = self.base_pre_norm_tir(z_i)
        x_dte_feat = self.base_pre_norm_tir(x_i)
        z_feat = self.base_blocks_pre_rgb_z(z_feat_rgb)
        x_feat = self.base_blocks_pre_rgb_x(x_feat_rgb)
        z_feat_dte = self.base_blocks_pre_tir_z(z_dte_feat)
        x_feat_dte = self.base_blocks_pre_tir_x(x_dte_feat)
        z_v = z_v + z_feat
        x_v = x_v + x_feat
        z_i = z_i + z_feat_dte
        x_i = x_i + x_feat_dte


        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x
        z_i += self.pos_embed_z
        x_i += self.pos_embed_x

        # if self.add_sep_seg:
        #     x += self.search_segment_pos_embed
        #     z += self.template_segment_pos_embed

        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode).to(torch.float32)
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode).to(torch.float32)

        # if self.add_cls_token:
        #     x = torch.cat([cls_tokens, x], dim=1)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        lens_x_new_rgb = lens_x
        lens_x_new_tir = lens_x
        # lens_z = 64
        # lens_x = 256

        mplt_index = 0

        # removed_indexes_rgb = []
        # removed_indexes_tir = []
        #
        global_index = torch.linspace(0, lens_x - 1, lens_x).to(x_v.device)
        global_index = global_index.repeat(B, 1)
        # remained_indexes_rgb = global_index
        # remained_indexes_tir = global_index
        keep_ratio_rgb = []
        keep_ratio_tir = []
        remained_rgb = torch.ones_like(global_index.unsqueeze(-1).repeat(1,1,x_v.size(-1))).cuda()
        remained_tir = torch.ones_like(global_index.unsqueeze(-1).repeat(1,1,x_v.size(-1))).cuda()
        remained_rgb_tokens = []
        remained_tir_tokens = []
        keep_ratio_all = []
        ts = torch.ones(B, lens_z, 1).cuda()
        pre_mask = torch.ones(B, lens_x*2, 1).cuda()
        pruned_rgb = torch.zeros(B, lens_x, self.embed_dim).cuda()
        pruned_tir = torch.zeros(B, lens_x, self.embed_dim).cuda()
        pruned_z = torch.zeros(B, lens_z, self.embed_dim).cuda()
        original_mask_rgb = []
        original_mask_tir = []
        for i, blk in enumerate(self.blocks):
            token_selection_count = 0
            if i in self.selection_places:
                # x_v = blk(x_v, False)
                # x_i = blk(x_i, False)

                # x_v_copy = copy.copy(x_v).detach()
                # x_v_copy = copy.copy(x_v).detach()
                # x_i_copy = copy.copy(x_i).detach()

                # z_v = x_v[:, :lens_z, :]
                # z_i = x_i[:, :lens_z, :]
                # xv = x_v[:, lens_z:, :]
                # xi = x_i[:, lens_z:, :]

                # z_f = c
                # ratio_token = getattr(self, 'selection_ratio_tokens_'+str(token_selection_count)).get().expand(B, -1, -1)
                # ratio_predictor = getattr(self, 'selection_ratio_predictor_'+str(token_selection_count))
                # f = torch.cat(((z_v + z_i) / 2, xv, xi), dim=1)
                # token_selection_count += 1
                # f, att = blk(f, True, torch.cat((ts, (remained_rgb[:, :, 0]>0).unsqueeze(-1), (remained_tir[:, :, 0]>0).unsqueeze(-1)), dim=1))
                # f, att = blk(f, True)

                # s = self.iner_ada_selection(torch.cat((x_v, x_i), dim=1))
                # s = self.iner_ada_selection2(s.permute(0, 2, 1))
                # keep_ratio = torch.sigmoid(ratio_predictor(f))
                # weighting, keep_ratio = ratio_predictor(f)
                # z_weight = torch.ones_like(weighting[:, :lens_z, :]).cuda()
                # weight = torch.cat((z_weight, weighting), dim=1)
                # x_v = torch.cat((z_v, xv * torch.cat((z_weight, weighting[:, :lens_x, :]), dim=1).repeat(1, 1, self.embed_dim)), dim=1)
                # x_i = torch.cat((z_i, xi * torch.cat((z_weight, weighting[:, lens_x, :]), dim=1).repeat(1, 1, self.embed_dim)), dim=1)
                # keep_ratio = torch.sigmoid(keep_ratio)
                # weighting = torch.sigmoid(weighting)
                # if kwards['epoch'] > self.ce_start:
                #
                #     x_v, keep_ratio_v, remained_rgb, x_i, keep_ratio_i, remained_tir = candidate_elimination_new2(weighting,x_v,x_i,remained_rgb,remained_tir,lens_z,keep_ratio, box_mask_z=kwards['ce_template_mask'],flag=0, pos_embed=self.pos_embed_x)
                #
                #     keep_ratio_rgb.append(keep_ratio_v)
                #     keep_ratio_tir.append(keep_ratio_i)
                #     remained_rgb_tokens.append((remained_rgb[:,:,0]>0).sum(dim=1))
                #     remained_tir_tokens.append((remained_tir[:,:,0]>0).sum(dim=1))
                # else:
                #     x_v = x_v * torch.cat((z_weight, weighting[:, :lens_x, :]), dim=1).repeat(1, 1, self.embed_dim)
                #     x_i = x_i * torch.cat((z_weight, weighting[:, lens_x:, :]), dim=1).repeat(1, 1, self.embed_dim)

                ratio_predictor = getattr(self, 'selection_ratio_predictor_' + str(token_selection_count))
                keep_mask = ratio_predictor(x_v, x_i)

                hard_keep = F.gumbel_softmax(keep_mask, hard=True, dim=-1)[:,:,0]
                mask_search_rgb = hard_keep[:, lens_z:lens_x+lens_z].unsqueeze(-1)
                mask_z_rgb = hard_keep[:, :lens_z].unsqueeze(-1)
                mask_search_tir = hard_keep[:, -lens_x:].unsqueeze(-1)
                mask_z_tir = hard_keep[:, lens_x+lens_z:lens_x+lens_z*2].unsqueeze(-1)
                this_mask = torch.cat((mask_search_rgb, mask_search_tir), dim=1)

                if self.progressive:
                    used_mask = this_mask * pre_mask
                    mask_rgb = used_mask[:, :lens_x]
                    mask_tir = used_mask[:, -lens_x:]
                else:
                    used_mask = this_mask

                filter_rgb = torch.cat((ts, used_mask[:,:lens_x, :]), dim=1)
                # filter_rgb = torch.cat((mask_z_rgb, used_mask[:,:lens_x, :]), dim=1)
                filter_tir = torch.cat((ts, used_mask[:,lens_x:, :]), dim=1)
                # filter_tir = torch.cat((mask_z_tir, used_mask[:,lens_x:, :]), dim=1)
                # new_mask = ((this_mask - pre_mask)<0).float()   # no reactivate
                # pruned_rgb += x_v[:, lens_z:,:] * (new_mask[:,:lens_x, :]).expand(B, lens_x, self.embed_dim)
                # pruned_tir += x_i[:, lens_z:,:] * (new_mask[:,lens_x:, :]).expand(B, lens_x, self.embed_dim)
                # pruned_rgb += self.pos_embed_x.expand(B, lens_x, self.embed_dim) * (new_mask[:, :lens_x, :]).expand(B, lens_x, self.embed_dim)
                # pruned_tir += self.pos_embed_x.expand(B, lens_x, self.embed_dim) * (new_mask[:, lens_x:, :]).expand(B, lens_x, self.embed_dim)
                x_v = blk(x_v, filter=filter_rgb)
                # x_v = blk(x_v*filter_rgb.expand(B, lens_z+lens_x, self.embed_dim), filter=filter_rgb)
                # x_v, att_r = blk(x_v*filter_rgb.expand(B, lens_z+lens_x, self.embed_dim), True, filter=filter_rgb)
                # x_i = blk(x_i*filter_tir.expand(B, lens_z+lens_x, self.embed_dim), filter=filter_tir)
                x_i = blk(x_i, filter=filter_tir)
                # x_i, att_i = blk(x_i*filter_tir.expand(B, lens_z+lens_x, self.embed_dim), True, filter=filter_tir)

                pre_mask = used_mask
                ratio_rgb = mask_rgb.sum(1)/lens_x
                ratio_tir = mask_tir.sum(1)/lens_x
                keep_ratio_rgb.append(ratio_rgb)
                keep_ratio_tir.append(ratio_tir)
                keep_ratio_all.append(used_mask)


                # original_mask_rgb.append(att_r[:, :, lens_z:, lens_z:].mean(2).mean(1))
                # original_mask_tir.append(att_i[:, :, lens_z:, lens_z:].mean(2).mean(1))
            else:
                # x_v = blk(x_v, False, torch.cat((ts, pre_mask[:,:lens_x, :]), dim=1))
                x_v = blk(x_v, False)
                # x_v, att_r = blk(x_v, True)
                # x_i = blk(x_i, False, torch.cat((ts, pre_mask[:,lens_x:, :]), dim=1))
                x_i = blk(x_i, False)
                # x_i, att_t = blk(x_i, True)



        # remained = ((mask_rgb + mask_tir) > 0).float()
        # remained_z = torch.ones_like(remained[:, :lens_z, :]).cuda()
        # abadon_z = torch.ones_like(remained[:, :lens_z, :]).cuda()
        # filter = torch.cat((remained_z, remained_rgb, remained_z, remained_tir), dim=1)[:,:,0].unsqueeze(-1).float()
        # x_f = self.base_fuse_block(torch.cat((x_v * torch.cat((remained_z, remained_rgb), dim=1),x_i * torch.cat((remained_z, remained_tir), dim=1)), dim=1), filter=filter)


        # x_f = self.base_fuse_block(x_v * torch.cat((remained_z, remained_rgb), dim=1) + x_i * torch.cat((remained_z, remained_tir), dim=1), filter=torch.cat((remained_z[:,:,0], remained[:,:,0]), dim=1).unsqueeze(-1)).to(torch.float32) * torch.cat((remained_z, remained), dim=1)
        # input = x_v * torch.cat((remained_z, remained_rgb), dim=1) + x_i * torch.cat((remained_z, remained_tir), dim=1)
        # and_remained=remained_rgb * remained_tir
        # or_remained = ((remained_rgb - and_remained) > 0).float() + ((remained_tir - and_remained) > 0).float()
        # input = input + input * torch.cat((abadon_z, or_remained), dim=1)
        # x_f = self.base_fuse_block(x_v+x_i, filter=torch.cat((remained_z, remained), dim=1)[:,:,0].unsqueeze(-1).float()).to(torch.float32)# * torch.cat((remained_z, remained), dim=1)
        # x_f = self.base_fuse_block(x_v+x_i, filter=torch.cat((ts, remained), dim=1)).to(torch.float32)# * torch.cat((remained_z, remained), dim=1)

        # x_f = self.base_fuse_block(torch.cat((x_v[:,:lens_z, :], x_v[:,lens_z:, :] + pruned_rgb), dim=1)+torch.cat((x_i[:,:lens_z, :], x_i[:,lens_z:, :] + pruned_tir), dim=1)).to(torch.float32)# * torch.cat((remained_z, remained), dim=1)
        x_f = self.base_fuse_block(x_v+x_i).to(torch.float32)# * torch.cat((remained_z, remained), dim=1)
        aux_dict = {"attn": None, "mask": keep_ratio_all, "remained_rgb":remained_rgb, "remained_tir":remained_tir}#, 'mask_rgb':original_mask_rgb, 'mask_tir':original_mask_tir
        return self.norm(x_f), aux_dict
        # return self.norm(x_f) * torch.cat((remained_z, and_remained), dim=1), aux_dict


    def tokens_selection(self, x_v, x_i, att_r=None, att_t=None, remained_rgb=None, remained_tir=None, lens_z=64, z_mask=None):

        # Iner_t4
        # s = self.iner_ada_selection(torch.cat((x_v, x_i), dim=1))
        # v = s[:, lens_z:(lens_z+lens_x_new_rgb), :].mean(1)
        # ir = s[:, (lens_z+lens_x_new_rgb) + lens_z:, :].mean(1)
        # v = [max(torch.sigmoid(v)[b], torch.ones(1).cuda() * 0.05) for b in range(v.size(0))]
        # ir = [max(torch.sigmoid(ir)[b], torch.ones(1).cuda() * 0.05) for b in range(ir.size(0))]
        # v = torch.Tensor(v).cuda()

        # Iner_t5
        # s = self.iner_ada_selection(torch.cat((x_v, x_i), dim=1))
        # # v = s[:, lens_z:(lens_z+lens_x_new_rgb), :]
        # # ir = s[:, (lens_z+lens_x_new_rgb) + lens_z:, :]
        # s = self.iner_ada_selection2(s.permute(0,2,1))
        # # v = [max(torch.sigmoid(v)[b], torch.ones(1).cuda() * 0.05) for b in range(v.size(0))]
        # # ir = [max(torch.sigmoid(ir)[b], torch.ones(1).cuda() * 0.05) for b in range(ir.size(0))]
        # # v = torch.Tensor(v).cuda()

        # Iner_t10
        z_v = x_v[:, :lens_z, :]
        z_i = x_i[:, :lens_z, :]
        xv = x_v[:, lens_z:, :]
        xi = x_i[:, lens_z:, :]

        z_f = (z_v + z_i) / 2

        s = self.iner_ada_selection(torch.cat((x_v, x_i), dim=1))
        s = self.iner_ada_selection2(s.permute(0,2,1))


        f = torch.cat((z_f, xv, xi), dim=1)

        _, att = self.selection_att(f, True)
        # s_i = (s_i > 0).sum(dim=1).items()

        # x_v, keep_ratio_v, remained_rgb = candidate_elimination(att, x_v, remained_rgb, lens_z, torch.sigmoid(s[:,:,0]), z_mask, flag=0)
        x_v, keep_ratio_v, remained_rgb, x_i, keep_ratio_i, remained_tir= candidate_elimination_new(att, x_v, x_i, remained_rgb, remained_tir, lens_z, torch.sigmoid(s[:,:,0]), z_mask, flag=0)
        # x_v, keep_ratio_v, remained_rgb = candidate_elimination(att_r, x_v, remained_rgb, lens_z, s_v, None)
        # x_i, keep_ratio_i, remained_tir = candidate_elimination(att, x_i, remained_tir, lens_z, torch.sigmoid(s[:,:,0]), z_mask, flag=1)
        # x_i, keep_ratio_i, remained_tir = candidate_elimination(att_t, x_i, remained_tir, lens_z, s_i, None)

        return x_v, keep_ratio_v, remained_rgb, x_i, keep_ratio_i, remained_tir

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformerMPLT, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformerMPLT(**kwargs)
    # model = VisionTransformerBase(**kwargs)
    # tensor = ([torch.rand(1, 3, 128, 128),torch.rand(1, 3, 128, 128)],
    #           [torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)])
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def _create_vision_transformerBase(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # model = VisionTransformerMPLT(**kwargs)
    model = VisionTransformerBase(**kwargs)
    # tensor = ([torch.rand(1, 3, 128, 128),torch.rand(1, 3, 128, 128)],
    #           [torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)])
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def _create_vision_transformerPre(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # model = VisionTransformerMPLT(**kwargs)
    model = VisionTransformerPre(**kwargs)
    # tensor = ([torch.rand(1, 3, 128, 128),torch.rand(1, 3, 128, 128)],
    #           [torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)])
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model

def _create_vision_transformerIner(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # model = VisionTransformerMPLT(**kwargs)
    model = VisionTransformerIner(**kwargs)
    # tensor = ([torch.rand(1, 3, 128, 128),torch.rand(1, 3, 128, 128)],
    #           [torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)])
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model

def vit_base_patch16_224_mplt(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

def vit_base_patch16_224_Base(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformerBase('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

def vit_small_patch16_224_mplt(pretrained=False, **kwargs):
    """
    ViT-Small model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def vit_base_patch16_224_Pre(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformerPre('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224_Iner(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformerIner('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

def vit_tiny_patch16_224_mplt(pretrained=False, **kwargs):
    """
    ViT-Tiny model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
