import glob
import json
import math
import os
import random
import cv2
import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataset.gaussian_aug import draw_gaussian
from lcnn.config import M


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx].replace("_label.npz", ".png")
        image = io.imread(iname).astype(float)[:, :, :3]
        # image = (image - M.image.mean) / M.image.stddev # Not right   !!!!!!!!!!!
        if self.split=="train":
            image = brightness_augment(image)

        image = np.rollaxis(image, 2).copy()

        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["corner", "center", "corner_offset","corner_bin_offset"]
            }
        if self.split=="train":
            image,target = random_rotate(image,target,factor=0.2) # 0.2 of all data will be applied random rotation

        # target["center"]=draw_gaussian(target["center"],radius=1) # apply gaussian augmentation

        return torch.from_numpy(image).float(), target, iname

def random_rotate(img,target,factor):
    img: [3,512,512]
    if np.random.uniform()>factor:
        return img,target
    image_size=img.shape[1:]
    angle=np.random.normal()*90
    matrix = cv2.getRotationMatrix2D(tuple([image_size[0] / 2, image_size[1] / 2]), -angle, 1)
    image = cv2.warpAffine(img.transpose(1,2,0), matrix, (image_size[1], image_size[0])).transpose(2,0,1)

    featuremap_size=target['corner'].shape[1:] # [128,128]
    # matrix = cv2.getRotationMatrix2D(tuple([featuremap_size[0] / 2, featuremap_size[1] / 2]), -angle, 1)
    for i in target: # target[i] [1,128,128]
        if len(target[i].shape)==4: # offset and corner_bin_offset
            new_map = torch.zeros_like(target[i])
            for j in range(2):
                for point in torch.nonzero(target[i][:, j, :, :].squeeze(), as_tuple=False):
                    new_point= rotate2DPoint(point, featuremap_size, angle).int()
                    if featuremap_size[0]>new_point[0]>=0 and featuremap_size[1]>new_point[1]>=0:
                        tmp_point_value = [target[i].squeeze()[0, point[0], point[1]]+point[0],
                                     target[i].squeeze()[1, point[0], point[1]]+point[1]]
                        tmp_point_value=rotate2DPoint(tmp_point_value, featuremap_size, angle)
                        tmp_point_value=[tmp_point_value[0]-new_point[0],tmp_point_value[1]-new_point[1]]
                        new_map.squeeze()[j,new_point[0],new_point[1]]=tmp_point_value[j]
            target[i] = new_map
        else:
            new_map = torch.zeros_like(target[i])
            for point in torch.nonzero(target[i].squeeze(), as_tuple=False):
                new_point = rotate2DPoint(point, featuremap_size, angle).int()
                if featuremap_size[0] > new_point[0] >= 0 and featuremap_size[1] > new_point[1] >= 0:
                    new_map.squeeze()[new_point[0], new_point[1]] = 1
            target[i] = new_map
    return image,target

def brightness_augment(img, factor=0.5):
    # random brightness
    img=img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb.astype(float)

def rotate2DPoint(Point, image_shape, angle):
    center=[image_shape[0]/2,image_shape[1]/2]
    matrix=cv2.getRotationMatrix2D(tuple(center), angle, 1)
    rotation=matrix[:,0:2:1]
    bias=matrix[:,2]
    return torch.from_numpy(np.dot(rotation,Point)+bias)

def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        # [b[1] for b in batch],
        default_collate([b[1] for b in batch]),
        default_collate([b[2].split('/')[-1] for b in batch])
    )
