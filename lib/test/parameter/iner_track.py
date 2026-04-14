from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.Iner.config import cfg, update_config_from_file


def parameters(yaml_name: str, verson:str, epoch:int):
    params = TrackerParams()
    save_dir = env_settings().save_dir
    # check_dir = env_settings().save_dir
    # update default config from yaml file
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    yaml_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % ('Iner', yaml_name))

    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = False
    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "%s/checkpoints/train/%s/%s/InerTrack_ep%04d.pth.tar" %
                                     (verson, 'Iner', yaml_name, epoch))
    # params.checkpoint = '/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-main/models/MPLTTrack_ep0015.pth.tar'
    # params.checkpoint = '/scratch/zhangyong-zt00315/Trackers/MPLT-main/models/MPLTTrack_ep0015.pth.tar'
    # params.checkpoint = '/data/Disk_A/zhangyong/MPLT-mymain-inerselection/models/BaseTrack_ep0050.pth.tar'

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
