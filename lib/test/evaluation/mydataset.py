import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class MyDataset(BaseDataset):
    """ RGBT234 dataset for RGB-T tracking.

    Publication:
        RGBT234:RGB-T Object Tracking: Benchmark and Baseline
        Chenglong Li, Xinyan Liang, Yijuan Lu, Nan Zhao, and Jin Tang
        https://arxiv.org/pdf/1805.08982.pdf
    Download dataset from: https://pan.baidu.com/share/init?surl=weaiBh0_yH2BQni5eTxHgg
    """

    def __init__(self):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)

        # self.base_path = os.path.join(self.env_settings.rgbt234_dir)
        # self.base_path = '/mnt/fast/nobackup/scratch4weeks/zt00315/Mydataset'
        self.base_path = self.env_settings.mydataset_dir
        self.sequence_list = self._get_sequence_list()
        # self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def pre(self, path):

        # res_txt = os.path.join(res_root, name+'.txt')
        # # res_time_txt = os.path.join(res_root, name+'time.txt')
        # if os.path.exists(res_txt):
        #     return

        if os.path.exists(os.path.join(path, 'flag.txt')):
            flag = 'common'
            # dicts = ['visible', 'infrared']
            with open(os.path.join(path, 'flag.txt')) as f:
                a = f.readlines()
                f.close()
            print('%s' % (a[0]))
            if a[0] == 'rgb':
                if os.path.exists(os.path.join(path, 'rgb_ref')):
                    dicts = ['color', 'rgb_ref']
                else:
                    dicts = ['color', 'infrared']
            elif a[0] == 'tir':
                if os.path.exists(os.path.join(path, 'tir_ref')):
                    dicts = ['tir_ref', 'infrared']
                else:
                    dicts = ['color', 'infrared']
                # dicts = ['tir_ref', 'infrared']
            else:
                # print('Wrong flag! ---- %s\n'%(name))
                # print('%s'%a)
                # print('111')
                return
        else:
            flag = 'attacked'
            dicts = ['visible', 'infrared']

        return dicts, flag

    def _construct_sequence(self, sequence_name):
        video_path = '{}/{}'.format(self.base_path, sequence_name)
        dicts, flag = self.pre(video_path)
        if flag == 'common':
            gt_rgb = sequence_name + '_1.txt'
            gt_tir = sequence_name + '_1.txt'
        else:
            gt_rgb = 'visible.txt'
            gt_tir = 'infrared.txt'

        RGB_img_list = sorted([video_path + '/' + dicts[0] + '/' + p for p in os.listdir(video_path + '/' + dicts[0]) if p.endswith(".jpg") or p.endswith(".png")])
        T_img_list = sorted([video_path + '/' + dicts[1] + '/' + p for p in os.listdir(video_path + '/' + dicts[1]) if p.endswith(".jpg") or p.endswith(".png")])

        RGB_gt = np.loadtxt(video_path + '/' + gt_rgb, delimiter=',')
        # T_gt = np.loadtxt(video_path + '/' + gt_tir, delimiter=',')

        # anno_path = '{}/{}/{}_1.txt'.format(self.base_path, sequence_name, sequence_name)
        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # frames_path_i = '{}/{}/infrared'.format(self.base_path, sequence_name)
        # frames_path_v = '{}/{}/visible'.format(self.base_path, sequence_name)
        # frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".jpg")]
        # frame_list_i.sort(key=lambda f: int(f[1:-5]))
        # frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".jpg")]
        # frame_list_v.sort(key=lambda f: int(f[1:-5]))
        # frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        # frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [RGB_img_list, T_img_list]
        return Sequence(sequence_name, frames_list, 'mydataset', RGB_gt)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
            f.close()
        # if split == 'ltrval':
        #     with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
        #         seq_ids = f.read().splitlines()
        #
        #     sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
