from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mplt_track.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    save_dir = env_settings().save_dir
    check_dir = env_settings().check_dir
    # update default config from yaml file
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    yaml_file = os.path.join(prj_dir, 'experiments/tracker/%s.yaml' % yaml_name)
    
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
    # params.checkpoint = os.path.join(check_dir, "%s/checkpoints/train/tracker/%s/MPLTTrack_ep%04d.pth.tar" %
    #                                  (yaml_name, yaml_name, cfg.TEST.EPOCH))
    # params.checkpoint = '/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-main/models/MPLTTrack_ep0015.pth.tar'
    params.checkpoint = '/scratch/zhangyong-zt00315/Trackers/MPLT-main/models/MPLTTrack_ep0015.pth.tar'

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
