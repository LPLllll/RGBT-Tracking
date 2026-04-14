"""
MPLT_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
# from lib.models.tracker.vit_mplt_care import vit_base_patch16_224_Base
from lib.models.tracker.vit_mplt_care import vit_base_patch16_224_Iner
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils import TensorDict
import torch.nn.functional as F
import  matplotlib.pyplot as plt
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


def draw_img(x):
    x = x.permute(1, 2, 0)
    x = (x - x.min()) / (x.max() - x.min())
    plt.imshow(x.detach().cpu().numpy())
    plt.show()

class InerTrack(nn.Module):
    """ This is the base class for MPLTTrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, box_head_rgb, box_head_tir, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.mplt_fuse_search = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.mplt_fuse_search_rgb = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.mplt_fuse_search_tir = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head
        self.box_head_rgb = box_head_rgb
        self.box_head_tir = box_head_tir

        self.aux_loss = aux_loss
        self.head_type = head_type

        self.base_weights = nn.Parameter(torch.ones(2))

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                change_v=None,
                change_i=None,
                epoch=100,
                seq_name=None
                ):
        # if (change_v is not None) and (change_i is not None):
        #     _, aux_dict = self.backbone(z=template, x=search,
        #                                 ce_template_mask=ce_template_mask,
        #                                 ce_keep_rate=ce_keep_rate,
        #                                 return_last_attn=return_last_attn, change_v=change_v, change_i=change_i)
        #     return aux_dict
        # else:
        x, xr, xi, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, epoch=epoch)
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        b, n, e = x.shape
        # hard_keep = F.gumbel_softmax(aux_dict['score'], hard=True, dim=-1)[:,:,0]
        # s = torch.cat(
        #     ((aux_dict['score_rgb'] - aux_dict['score_tir'])>=0, (aux_dict['score_tir'] - aux_dict['score_rgb'])>0), dim=1)
        # t = hard_keep.unsqueeze(-1).expand(b, 1, e).expand(b, n, e) * x_r + (1 - hard_keep).unsqueeze(-1).expand(b, 1, e).expand(b, n, e) * x_i

        # out = self.forward_head(feat_last, t, None)
        # out = self.forward_head(feat_last, None)
        out = self.forward_head_triple(feat_last, xr, xi, aux_dict['mask'], None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

        #     # if opt_feat.size(0) > 1:
        #     #     if index_rgb.size(0) > 0 and index_tir.size(0) > 0:
        #     #         batch_rgb = torch.index_select(opt_feat, dim=0, index=index_rgb)
        #     #         batch_tir = torch.index_select(opt_feat, dim=0, index=index_tir)
        #     #         # keep_r = ((keep_rgb>=0.5) * keep_rgb).sum() / index_rgb.size(0)
        #     #         # keep_t = ((keep_rgb<0.5) * keep_tir).sum() / index_tir.size(0)
        #     #         # para1, para2 = self.meta_head(keep_r, keep_t, self.box_head.state_dict(), self.box_head_tir.state_dict())
        #     #         # self.box_head.load_state_dict(para1)
        #     #         # self.box_head_tir.load_state_dict(para2)
    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        # enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.mplt_fuse_search(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
    def forward_head_triple(self, cat_feature, xv, xi, s, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        # enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat_all = self.mplt_fuse_search(opt_feat)

        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = xv[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = xv[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        # enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat_rgb = self.mplt_fuse_search_rgb(opt_feat)

        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = xi[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = xi[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        # enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat_tir = self.mplt_fuse_search_tir(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr_all, bbox_all, size_map_all, offset_map_all, max_score_all = self.box_head(opt_feat_all, gt_score_map)
            score_map_ctr_rgb, bbox_rgb, size_map_rgb, offset_map_rgb, max_score_rgb = self.box_head_rgb(opt_feat_rgb, gt_score_map)
            score_map_ctr_tir, bbox_tir, size_map_tir, offset_map_tir, max_score_tir = self.box_head_tir(opt_feat_tir, gt_score_map)
            if s is None:
                if bbox_tir.size(0) != 1:
                    # triple output
                    # bbox = [bbox_tir, bbox_rgb, bbox_all]
                    # score_map_ctr = [score_map_ctr_tir, score_map_ctr_rgb, score_map_ctr_all]
                    # size_map = [size_map_tir, size_map_rgb, size_map_all]
                    # offset_map = [offset_map_tir, offset_map_rgb, offset_map_all]

                    #learnable weights   the corresponding loss is also changed
                    w = F.softmax(self.base_weights, 0)
                    # w = [1/3, 1/3, 1/3]
                    # bbox = [bbox_tir * w[0]+  bbox_rgb* w[1]+ bbox_all* w[2]]
                    # score_map_ctr = [score_map_ctr_tir* w[0]+ score_map_ctr_rgb* w[1]+score_map_ctr_all* w[2]]
                    # size_map = [size_map_tir* w[0]+ size_map_rgb* w[1]+ size_map_all*w[2]]
                    # offset_map = [offset_map_tir* w[0]+ offset_map_rgb* w[1]+ offset_map_all*w[2]]
                    bbox = [bbox_tir * w[0]+  bbox_rgb* w[1]]
                    score_map_ctr = [score_map_ctr_tir* w[0]+ score_map_ctr_rgb* w[1]]
                    size_map = [size_map_tir* w[0]+ size_map_rgb* w[1]]
                    offset_map = [offset_map_tir* w[0]+ offset_map_rgb* w[1]]
                else:
                    #learnable weights   the corresponding loss is also changed
                    w = F.softmax(self.base_weights, 0)
                    # w = [1/3, 1/3, 1/3]
                    # bbox = [bbox_tir * w[0]+  bbox_rgb* w[1]+ bbox_all* w[2]]
                    # score_map_ctr = [score_map_ctr_tir* w[0]+ score_map_ctr_rgb* w[1]+score_map_ctr_all* w[2]]
                    # size_map = [size_map_tir* w[0]+ size_map_rgb* w[1]+ size_map_all*w[2]]
                    # offset_map = [offset_map_tir* w[0]+ offset_map_rgb* w[1]+ offset_map_all*w[2]]
                    bbox = bbox_tir * w[0]+  bbox_rgb* w[1]
                    score_map_ctr = score_map_ctr_tir* w[0]+ score_map_ctr_rgb* w[1]
                    size_map = size_map_tir* w[0]+ size_map_rgb* w[1]
                    offset_map = offset_map_tir* w[0]+ offset_map_rgb* w[1]

                    # if max_score_rgb > max_score_all and max_score_rgb > max_score_tir:
                    #     bbox = bbox_rgb
                    #     score_map_ctr = score_map_ctr_rgb
                    #     size_map = size_map_rgb
                    #     offset_map = offset_map_rgb
                    # elif max_score_all > max_score_rgb and max_score_all > max_score_tir:
                    # if (max_score_tir + max_score_rgb) > 1.2 * max_score_all:
                    #     bbox = bbox_all
                    #     score_map_ctr = score_map_ctr_all
                    #     size_map = size_map_all
                    #     offset_map = offset_map_all
                    # else:
                    #     if max_score_rgb >= max_score_tir:
                    #         bbox = bbox_rgb
                    #         score_map_ctr = score_map_ctr_rgb
                    #         size_map = size_map_rgb
                    #         offset_map = offset_map_rgb
                    #     else:
                    #         bbox = bbox_tir
                    #         score_map_ctr = score_map_ctr_tir
                    #         size_map = size_map_tir
                    #         offset_map = offset_map_tir
                    # elif max_score_tir > max_score_rgb and max_score_tir > max_score_all:
                    # bbox = bbox_tir
                    # score_map_ctr = score_map_ctr_tir
                    # size_map = size_map_tir
                    # offset_map = offset_map_tir
            else:

                ss=F.gumbel_softmax(s.squeeze(-1), hard=True, dim=1)[:, 0].unsqueeze(-1)


                bboxs = ss * bbox_rgb + (1 - ss) * bbox_tir
                score_map_ctrs = ss.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3]) * score_map_ctr_rgb + \
                                 (1 - ss.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3])) * score_map_ctr_tir
                size_maps = ss.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], size_map_all.shape[1], size_map_all.shape[2], size_map_all.shape[3]) * size_map_rgb + \
                            (1 - ss.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], size_map_all.shape[1], size_map_all.shape[2], size_map_all.shape[3])) * size_map_tir
                offset_maps = ss.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], offset_map_all.shape[1], offset_map_all.shape[2], offset_map_all.shape[3]) * offset_map_rgb + \
                              (1 - ss.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], offset_map_all.shape[1], offset_map_all.shape[2], offset_map_all.shape[3])) * offset_map_tir
                max_scores = ss * max_score_rgb + (1 - ss) * max_score_tir
                # bbox = (bboxs + bbox_all) / 2.0
                # score_map_ctr =(score_map_ctrs + score_map_ctr_all) / 2.0
                # size_map =(size_maps + size_map_all) / 2.0
                # offset_map =(offset_maps + offset_map_all) / 2.0
                # max_score =(max_scores + max_score_all) / 2.0
                m1 = max_scores / (max_scores + max_score_all)
                m2 = max_score_all / (max_scores + max_score_all)
                # print('%f, %f, %f'%(ss, m1, m2))

                bbox = bboxs * m1 + bbox_all * m2
                score_map_ctr =score_map_ctrs * m1.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3]) \
                               + score_map_ctr_all * m2.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3])
                size_map =size_maps * m1.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3])\
                          + size_map_all * m2.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3])
                offset_map =offset_maps * m1.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3])\
                            + offset_map_all * m2.unsqueeze(-1).unsqueeze(-1).expand(ss.shape[0], score_map_ctr_all.shape[1], score_map_ctr_all.shape[2], score_map_ctr_all.shape[3])
                max_score =max_scores * m1 + max_score_all * m2
            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': bbox,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
            # return 1
        else:
            raise NotImplementedError

def build_iner_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../models')
    if cfg.MODEL.PRETRAIN_FILE and ('InerTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_iner':
        backbone = vit_base_patch16_224_Iner(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            mplt_loc=cfg.MODEL.BACKBONE.MPLT_LOC,
                                            mplt_drop_path=cfg.TRAIN.MPLT_DROP_PATH, epoch=cfg.MODEL.BACKBONE.CE_START
                                            )
    else:
        raise NotImplementedError

    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('----backbone build!!\n')
    #     f.close()

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    box_head_rgb = build_box_head(cfg, hidden_dim)
    box_head_tir = build_box_head(cfg, hidden_dim)

    model = InerTrack(
        backbone,
        box_head,
        box_head_rgb,
        box_head_tir,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('----trtacker build!!\n')
    #     f.close()
    # if 'MPLTTrack' in cfg.MODEL.PRETRAIN_FILE and training:
    # if 'BaseTrack' in cfg.MODEL.PRETRAIN_FILE and training:
    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        if cfg.TRAIN.TYPE == 'Iner':
            model.box_head_tir.load_state_dict(model.box_head.state_dict())
            model.box_head_rgb.load_state_dict(model.box_head.state_dict())
            model.mplt_fuse_search_rgb.load_state_dict(model.mplt_fuse_search.state_dict())
            model.mplt_fuse_search_tir.load_state_dict(model.mplt_fuse_search.state_dict())
            model.backbone.base_fuse_block_rgb.load_state_dict(model.backbone.base_fuse_block.state_dict())
            model.backbone.base_fuse_block_tir.load_state_dict(model.backbone.base_fuse_block.state_dict())
            model.backbone.norm_rgb.load_state_dict(model.backbone.norm.state_dict())
            model.backbone.norm_tir.load_state_dict(model.backbone.norm.state_dict())
    # if 'InerTrack' in cfg.MODEL.PRETRAIN_FILE and training:
    #     pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    #     checkpoint = torch.load(pretrained_file, map_location="cpu")
    #     missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    #     print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    # with open('/mnt/fast/nosbackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('----load pretrained!!\n')
    #     f.close()
    return model
