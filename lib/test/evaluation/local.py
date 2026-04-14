from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.davis_dir = ''
    settings.network_path = ''    # Where tracking networks are stored.
    settings.gtot_dir = ''
    # settings.lasher_path = '/vol/research/facer2vm_occ/people/zhangyong/datasets/Lasher'
    settings.lasher_path = '/mnt/fast/nobackup/scratch4weeks/zt00315/Lasher'
    settings.UAV_RGBT_dir = ''
    settings.rgbt234_dir = ''
    settings.got10k_path = ''
    settings.rgbt210_dir = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.network_path = ''  # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/scratch/zhangyong-zt00315/Trackers/MPLT-main'
    settings.result_plot_path = ''
    settings.results_path = '/scratch/zhangyong-zt00315/Trackers/MPLT-main/results'
    # settings.results_path = '/mnt/fast/nobackup/users/zt00315/Trackers/MPLT-main'
    settings.save_dir = '/mnt/fast/nobackup/scratch4weeks/zt00315/Trackers/MPLT'
    settings.save_dir = '/data/Disk_D/zhangyong_space/snapshot/Selective'
    # settings.mydataset_dir = '/vol/research/facer2vm_occ/people/zhangyong/datasets/Mydataset'
    settings.mydataset_dir = '/mnt/fast/nobackup/scratch4weeks/zt00315/datasets/Mydataset'
    return settings

