class EnvironmentSettings:
    def __init__(self):
        # self.workspace_dir = '/scratch/zhangyong-zt00315/Trackers/MPLT-mymain-inerselection'  # Base directory for saving network checkpoints.
        self.workspace_dir = '/data/Disk_A/zhangyong/MPLT-mymain-inerselection-train'  # Base directory for saving network checkpoints.
        # self.workspace_dir = '/mnt/fast/nobackup/users/zt00315/Tracker/MPLT-main-preselection'  # Base directory for saving network checkpoints.

        self.tensorboard_dir = ''  # Directory for tensorboard files.
        self.pretrained_networks = '/home/young/Code/MPLT-main-prompt/pretrained_networks'
        self.got10k_val_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k/val'
        self.lasot_lmdb_dir = '/media/young/TiPlus/Datasets/LaSOT/zip'
        self.got10k_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/coco_lmdb'
        self.coco_dir = '/media/young/Data-SSD/Datasets/COCO2017'
        self.lasot_dir = '/media/young/TiPlus/Datasets/LaSOT/zip'
        self.got10k_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k/train'
        self.trackingnet_dir = '/media/young/TiPlus/Datasets/TrackingNet'
        self.depthtrack_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/depthtrack/train'
        # self.lasher_train_dir = '/media/young/Data-SSD/Datasets/LasHeR-Divided/TrainingSet/Trainingset'
        # self.lasher_test_dir = '/media/young/Data-SSD/Datasets/LasHeR-Divided/TestingSet/testingset'

        # self.lasher_dir = '/mnt/fast/nobackup/scratch4weeks/zt00315/Lasher'
        # self.lasher_dir = '/vol/research/facer2vm_occ/people/zhangyong/datasets/Lasher'
        self.lasher_dir = '/data/Disk_A/zhangyong/LasHeR3'
        self.vtuavst_dir = '/data/Disk_A/zhangyong/VTUAV/VTUAV/VTUAV-ST'
        self.vtuavsttest_dir = '/data/Disk_A/zhangyong/VTUAV/VTUAV/test-st'

        self.visevent_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/visevent/train'
