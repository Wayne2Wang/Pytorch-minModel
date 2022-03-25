import numpy as np
import scipy.io as io
from os import listdir
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.utils.data.dataset import Dataset

import torchio.transforms as transforms


###################################################################
## code credit: https://github.com/xchhuang/simple-pytorch-3dgan ##
###################################################################
def getVoxelFromMat(path, cube_len=64):
    if cube_len == 32:
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    else:
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


#########################################################################
## code adapted from: https://github.com/xchhuang/simple-pytorch-3dgan ##
#########################################################################
def SavePloat_Voxels(voxels, path, title, disp_num, show=False):
    voxels = voxels[:disp_num].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, disp_num//2)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if show:
        plt.show()
        plt.close()
    else:
        plt.savefig(path + '/{}.png'.format(title), bbox_inches='tight')
        plt.close()


#########################################################################
## code adapted from: https://github.com/xchhuang/simple-pytorch-3dgan ##
#########################################################################
class ShapeNetDataset(Dataset):

    def __init__(self, root='../../datasets/ShapeNet_Chair/'):
        
        
        self.root = root
        self.listdir_ = listdir(self.root)

        data_size = len(self.listdir_)
        self.listdir_ = self.listdir_[0:int(data_size)]
        
        print ('data_size =', len(self.listdir_))

    def __getitem__(self, index):
        transformations = transforms.Compose({
            transforms.RandomFlip(flip_probability=0.2) : 0.5,
            transforms.RandomAffine(scales=0.1, degrees=10, default_pad_value=0, translation=1) : 0.5, # causing to be real valued
        })
        with open(self.root + self.listdir_[index], "rb") as f:
            volume = np.asarray(getVoxelFromMat(f, 64), dtype=np.float32)
        x = torch.FloatTensor(volume).unsqueeze(0)
        reg_x = transformations(x)

        return x, reg_x

    def __len__(self):
        return len(self.listdir_)



