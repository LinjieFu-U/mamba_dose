import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms

def random_rot(img1,img2):
    k = np.random.randint(0, 3)
    img1 = np.rot90(img1, k+1)
    img2 = np.rot90(img2, k+1)
    return img1,img2

def random_flip(img1,img2):
    axis = np.random.randint(0, 2)
    img1 = np.flip(img1, axis=axis).copy()
    img2 = np.flip(img2, axis=axis).copy()
    return img1,img2

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']

        if random.random() > 0.5:
            lr, hr = random_rot(lr, hr)
        if random.random() > 0.5:
            lr, hr = random_flip(lr, hr)
        sample = {'lr': lr,'hr': hr}
        return sample



class Train_Data(Dataset):
    def __init__(self):
        path = r'J:\code\DDPM\data\zenodo\data_train_zenodoCT.hdf5'
        f = h5py.File(path,'r')
        load_data = f['images']
        cbct_path=f['names']
        self.lr = load_data
        self.cbctpath=cbct_path
        path = r'J:\code\DDPM\data\zenodo\data_train_zenodoT2.hdf5'
        f = h5py.File(path,'r')
        load_data = f['images']
        ct_path=f['names']
        self.hr = load_data
        self.ctpath = ct_path
        c, self.h, self.w = self.lr.shape

        self.len = c
        self.transform=transforms.Compose([RandomGenerator(output_size=[self.h, self.w])])

    def __getitem__(self, index):
        x = self.lr[index, :, :]
        y = self.hr[index, :, :]

        x = self.norm(x)
        y = self.norm(y)

        sample = {'lr': x,'hr': y}
        if self.transform:
            sample = self.transform(sample)

        x, y = sample['lr'], sample['hr']

        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))

        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()

        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)

        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)

        # ct_path=str(self.ctpath[index].decode('utf-8'))
        # cbct_path=str(self.cbctpath[index].decode('utf-8'))
        return xx, yy
    def __len__(self):
        return self.len

    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x


class Valid_Data(Dataset):
    def __init__(self):
        path = r'J:\code\DDPM\data\zenodo\data_val_zenodoCT.hdf5'
        f = h5py.File(path,'r')
        load_data = f['images']
        cbct_path=f['names']
        self.lr = load_data
        self.cbctpath=cbct_path
        path = r'J:\code\DDPM\data\zenodo\data_val_zenodoT2.hdf5'
        f = h5py.File(path,'r')
        load_data = f['images']
        ct_path=f['names']
        self.hr = load_data
        self.ctpath = ct_path
        c, self.h, self.w = self.lr.shape

        self.len = 27

    def __getitem__(self, index):
        x = self.lr[index, :, :]
        y = self.hr[index, :, :]

        x = self.norm(x)
        y = self.norm(y)

        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))

        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()

        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)

        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)

        # ct_path=str(self.ctpath[index].decode('utf-8'))
        # cbct_path=str(self.cbctpath[index].decode('utf-8'))
        return xx, yy

    def __len__(self):
        return self.len

    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x

class Test_Data(Dataset):
    def __init__(self):
        path = r'J:\code\DDPM\data\syl\data_test_zenodoCT.hdf5'
        f = h5py.File(path,'r')
        load_data = f['images']
        cbct_path=f['names']
        self.lr = load_data
        self.cbctpath=cbct_path
        path = r'J:\code\DDPM\data\syl\data_test_zenodoT2.hdf5'
        f = h5py.File(path,'r')
        load_data = f['images']
        ct_path=f['names']
        self.hr = load_data
        self.ctpath = ct_path
        c, self.h, self.w = self.lr.shape

        self.len = 10000

    def __getitem__(self, index):
        x = self.lr[index, :, :]
        y = self.hr[index, :, :]

        x = self.norm(x)
        y = self.norm(y)

        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))

        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()

        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)

        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)
        ct_path=str(self.ctpath[index].decode('utf-8'))
        cbct_path=str(self.cbctpath[index].decode('utf-8'))
        return xx, yy,cbct_path,ct_path

    def __len__(self):
        return self.len

    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x