import numpy as np
import random
import pathlib
import os
import torch
import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, phase,aug=False):
        self.phase = phase
        self.aug = aug
        # self.transform = {'train': train_transform, 'val': val_transform}
        self.list_case_id = {'train': r'D:\ddpm-dose\data\2\train',
                             'val': r'D:\ddpm-dose\data\2\val',
                             'test': r'D:\ddpm-dose\data\2\test'}[phase]
        self.list_datapath = []
        for case_id in os.listdir(self.list_case_id):
            path=os.path.join(self.list_case_id,case_id)
            list_fn = pathlib.Path(path + '/slice_image/').glob("*_structure_image.npy")
            for fn in list_fn:
                n_slice = str(fn).split('/')[-1][0:-20]
                imagepath = n_slice + '_structure_image.npy'
                dosepath = n_slice + '_dose.npy'
                self.list_datapath.append([imagepath, dosepath])

        random.shuffle(self.list_datapath)
        self.sum_case = len(self.list_datapath)

    def __getitem__(self, index_):
        # 按索引获取数据集中的一个样本
        if index_ <= self.sum_case - 1:
            datapath = self.list_datapath[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            datapath = self.list_datapath[new_index_]

        npimage = np.load(datapath[0]).transpose((2, 0, 1))
        npdose = np.load(datapath[1])
        image_ = torch.from_numpy(npimage.copy()).float()
        dose_ = torch.from_numpy(npdose.copy()).float()
        return image_,  dose_,datapath[0],datapath[1]

    def __len__(self):
        return self.sum_case

