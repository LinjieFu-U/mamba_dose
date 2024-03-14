import numpy as np
import random
import pathlib
import os
import torch
import torch.utils.data as data
from PIL import Image
class MyDataset(data.Dataset):
    def __init__(self, phase,aug=False):
        self.phase = phase
        self.aug = aug
        # self.transform = {'train': train_transform, 'val': val_transform}
        self.list_case_id = {'train': r'/mnt/d/ddpm-dose/dosenewdata/train',
                             'val': r'/mnt/d/ddpm-dose/dosenewdata/val',
                             'test': r'/mnt/d/ddpm-dose/dosenewdata/test'}[phase]
        self.list_datapath = []

        for case_id in os.listdir(self.list_case_id):
            path=os.path.join(self.list_case_id,case_id)
            list_fn = pathlib.Path(path + '/slice_image/').glob("*_structure_image.npy")
            for fn in list_fn:
                dir_path, file_name = os.path.split(fn)
                base_name = os.path.splitext(file_name)[0]
                base_name = base_name.replace('_structure_image', '')
                n_slice = os.path.join(dir_path, base_name)

                imagepath = n_slice + '_structure_image.npy'
                dosepath = n_slice + '_dose.npy'
                maskpath=n_slice+'_possible_mask.npy'
                self.list_datapath.append([imagepath, dosepath,maskpath])

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
        npmask=np.load(datapath[2])
        image_ = torch.from_numpy(npimage.copy()).float()
        dose_ = torch.from_numpy(npdose.copy()).float()
        mask_=torch.from_numpy(npmask.copy()).float()
        return image_,  dose_,mask_,datapath[0],datapath[1]

    def __len__(self):
        return self.sum_case

