import os
from torch.utils.data import Dataset
import torch
import math
import random
from skimage import io
from torchvision import transforms
import numpy as np
from collections import defaultdict
from utils.data_augmentation import RandomFlip


class ReadDataset(Dataset):
    def __init__(self, path, GaussianRadius:int=0,PercentageOfSamplesForTrain:int=0.7, randomflip=False):
        '''
        init
        :param path: path of dataset
        '''
        if path[-1]!='/':
            path+='/'
        self.path = path
        self.GaussianRadius=GaussianRadius
        self.PercentageOfSamplesForTrain=PercentageOfSamplesForTrain
        self.pairedDict,self.filelist = read_images(self.path)
        self.flip=randomflip
        self.flipfunc=None
        if self.flip:
            self.flipfunc = RandomFlip(h=True, v=True, p=0.5)

    def fetchImageAndLabel(self, imgp, labelp):
        '''
        a funtion that fetches image and label information from dataset
        :param imgp: path of image
        :param labelp: path of image's label
        :return: image tensor and label
        '''
        image = io.imread(imgp) # range [0,255], type numpy.ndarray, ImageSize[0]*ImageSize[1]*3
        r=self.GaussianRadius # radius for gaussian augmentation
        keypoints = np.zeros((image.shape[0], image.shape[1]),dtype=np.float32)
        xvector = np.zeros((image.shape[0], image.shape[1]),dtype=np.float32)
        yvector = np.zeros((image.shape[0], image.shape[1]),dtype=np.float32)
        centermap = np.zeros((image.shape[0], image.shape[1]),dtype=np.float32)
        with open(labelp) as f:
            tmp = f.readline()
            while tmp != "":
                vertices=eval(tmp)
                vertices=[[i[1],i[0]] for i in vertices]
                center=np.array(vertices) # center of a polygon
                center=np.sum(center,0)/len(center)
                center=[int(i) for i in np.round(center).tolist()]
                self.draw_gaussian(centermap, [center[0], center[1]], radius=r, k=1)
                for i in vertices:
                    self.draw_gaussian(keypoints,[i[0],i[1]],radius=r,k=1)
                    self.draw_gaussian(xvector, [i[0], i[1]], radius=r, k=center[0]-i[0],decreasing=False)
                    self.draw_gaussian(yvector, [i[0], i[1]], radius=r, k=center[1]-i[1],decreasing=False)
                    # b[i[0], i[1]] = 1
                    # xvector[i[0],i[1]]=center[0]-i[0]
                    # yvector[i[0],i[1]]=center[1]-i[1]
                tmp = f.readline()
        keypoints=torch.from_numpy(keypoints)
        xvector=torch.from_numpy(xvector)
        yvector=torch.from_numpy(yvector)
        centermap=torch.from_numpy(centermap)

        keypoints = keypoints.unsqueeze(0)
        xvector = xvector.unsqueeze(0)
        yvector = yvector.unsqueeze(0)
        centermap = centermap.unsqueeze(0)

        transform = transforms.ToTensor()
        image_tensor = transform(image)

        assert keypoints.shape==xvector.shape==yvector.shape==centermap.shape
        assert image_tensor.dtype==keypoints.dtype==xvector.dtype==yvector.dtype==torch.float32==centermap.dtype
        return image_tensor, keypoints, xvector, yvector, centermap # image_tensor,keypoints,xvector,yvector:FloatTensor in -1*image.shape[0]*image.shape[1]

    def gaussian2D(self,shape, sigma=1):
        '''
        Generate 2D gaussian
        '''
        m, n = [(ss - 1.) / 2. for ss in shape]
        x, y = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian(self,heatmap, center, radius, k=1, decreasing=True):
        '''
        Apply 2D gaussian
        :param heatmap: feature map on which gaussian will be applied
        :param center: center of gaussian
        :param radius: radius of gaussian
        :param k: value
        '''
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = center
        height, width = heatmap.shape[0:2]

        top, bottom = min(x, radius), min(height - x, radius + 1)
        left, right = min(y, radius), min(width - y, radius + 1)
        masked_heatmap = heatmap[x - top:x + bottom,y - left:y + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        assert masked_heatmap.shape==masked_gaussian.shape,'mask shape in draw_gaussian are not equal'
        if decreasing:
            if k>0:
                np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            else:
                np.minimum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        else:
            if k>0:
                np.maximum(masked_heatmap,np.ones_like(masked_heatmap)*k,out=masked_heatmap)
            else:
                np.minimum(masked_heatmap,np.ones_like(masked_heatmap)*k,out=masked_heatmap)

    def __getitem__(self, index):
        '''
        if self.train:
        return filename in string format; image and label in torch.FloatTensor type(torch.float32). Label consists of location, xvector and y vector
        '''
        filename=self.filelist[index]
        img = self.pairedDict[filename][0]
        label = self.pairedDict[filename][1]
        img, location, xvector, yvector, centermap = self.fetchImageAndLabel(img, label)
        label = {'location':location, 'xvector':xvector, 'yvector':yvector, 'center':centermap}

        if self.flip:
            img,label=self.flipfunc(img,label)

        return {'filename':filename,'image':img, 'label':label}

    def __len__(self):
        return len(self.filelist)


def read_images(path):
    '''
    a function returns paths of files in training/test dataset
    :param path: path of data pieces
    :return: a list of filenames and a dictionary with filenames as key, file paths as value
    '''
    filelist=[f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    paireddict=defaultdict(lambda : 'Nan')
    for i in filelist:
        key,type=i.split('.')
        if key=='' or type=='':
            continue
        if paireddict[key]=='Nan':
            paireddict[key]=[None]*2
        if type=='txt':
            paireddict[key][1]=os.path.join(path,i)
        else:
            paireddict[key][0]=os.path.join(path,i)
    filelist=sorted(paireddict.keys())
    return paireddict,filelist

# def recursive_to(input, device):
#     if isinstance(input, torch.Tensor):
#         return input.to(device)
#     if isinstance(input, dict):
#         for name in input:
#             if isinstance(input[name], torch.Tensor):
#                 input[name] = input[name].to(device)
#         return input
#     if isinstance(input, list):
#         for i, item in enumerate(input):
#             input[i] = recursive_to(item, device)
#         return input
#     assert False