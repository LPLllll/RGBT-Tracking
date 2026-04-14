import os

# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.tracker import build_mplt_track
from lib.models.tracker import build_base_track
# forward propagation related
from lib.train.actors import MPLTTrackActor
from lib.train.actors import BaseTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for Base RGB-T Tracker'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    if not os.path.exists(settings.save_dir):
        os.makedirs(settings.save_dir)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)
    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('data loader finished!!!\n')
    #     f.close()
    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "Base":
        # net = build_mplt_track(cfg)
        net = build_base_track(cfg)
    else:
        raise ValueError("illegal script name")
    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines(settings.script_name + '----tracker build!!\n')
    #     f.close()
    # wrap networks to distributed one
    net.cuda()
    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('1\n')
    #     f.close()
    # if settings.local_rank != -1:
    #     # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
    #     net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
    #     settings.device = torch.device("cuda:%d" % settings.local_rank)
    # else:
    #     settings.device = torch.device("cuda:0")
    net = torch.nn.DataParallel(net, [0, 1]).cuda()
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "Base":
        # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
        #
        #     f.writelines('loss ???!!\n')
        #     f.close()
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
        #
        #     f.writelines('loss ok!!\n')
        #     f.close()
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        # actor = MPLTTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        actor = BaseTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
        #
        #     f.writelines(settings.script_name  + 'loss ok!!\n')
        #     f.close()
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('actor prepared!!\n')
    #     f.close()
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    # with open('/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-mymain/out.txt', 'a') as f:
    #
    #     f.writelines('prepared!!\n')
    #     f.close()
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
