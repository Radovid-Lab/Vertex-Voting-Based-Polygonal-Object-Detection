from typing import List
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch import nn
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from conf.config import Config


def drawDot(image, point: List[int], color: List[int]):
    '''
    draw dot on image
    :param image: image. should be n*n*3
    :param point: coordinates of a point.
    :param color: color of the dot
    '''
    xminimum = int(max(point[0] - 1, 0))
    xmaximum = int(min(point[0] + 1, image.shape[0] - 1))
    yminimum = int(max(point[1] - 1, 0))
    ymaximum = int(min(point[1] + 1, image.shape[1] - 1))
    for x in range(xminimum, xmaximum + 1):
        for y in range(yminimum, ymaximum):
            image[x, y] = color
    return


def drawLine(image, point1: List[int], point2: List[int], color: List[int]):
    '''
    draw a line between point1 and point2 on given image
    :param image: image. n*n*3
    :param point1: point1
    :param point2: point2
    :param color: color of line
    '''
    k = (point2[1] - point1[1]) / (point2[0] - point1[0] + 1e-6)
    if abs(point2[0] - point1[0]) > abs(point2[1] - point1[1]):
        if point2[0] - point1[0] < 0:
            step = -1
        else:
            step = 1
        for i in range(0, int(point2[0] - point1[0]), step):
            temppoint = [int(i + point1[0]), int(k * i + point1[1])]
            image[temppoint[0], temppoint[1]] = color
    else:
        if point2[1] - point1[1] < 0:
            step = -1
        else:
            step = 1
        for i in range(0, int(point2[1] - point1[1]), step):
            temppoint = [int(i / k + point1[0]), int(i + point1[1])]
            image[temppoint[0], temppoint[1]] = color
    return

def getPredictedCenter(corner,xoffset_map,yoffset_map):
    '''
    for each corner, an object center is predicted by combining xoffset_map and yoffset_map and corner coordinates.
    :param corners: corner coordinates
    :param xoffset_map: feature map for x offset
    :param yoffset_map: feature map for y offset
    :return: predicted object center, x&y coordinates are floatTensors
    '''
    shape=xoffset_map.shape
    zero=torch.tensor(0.)
    predictedCenter = [corner[0] + xoffset_map[corner[0], corner[1]], corner[1] + yoffset_map[corner[0], corner[1]]]
    predictedCenter[0] = max(predictedCenter[0], zero)
    predictedCenter[0] = min(predictedCenter[0], torch.from_numpy(np.array(shape[0]-1).astype(np.float32)))
    predictedCenter[1] = max(predictedCenter[1], zero)
    predictedCenter[1] = min(predictedCenter[1], torch.from_numpy(np.array(shape[1]-1).astype(np.float32)))
    return predictedCenter


def visualize(picture, labels,config:Config,NMS=False):
    '''
    Visualize prediction
    :param picture: picture in n*n*3 ndarray
    :param labels: labels: dict. values are FloatTensor in 1*n*n form
    :param should be true when visualizing predicted results, NMS will be applied
    :return return the image in n*n*3 ndarray
    '''

    if type(picture)!=np.ndarray:
        image=picture.squeeze().numpy().copy()
    shape=image.shape
    location = labels["location"].squeeze().view(shape[0],shape[1],-1)
    xvector = labels["xvector"].squeeze().view(shape[0],shape[1],-1)
    yvector = labels["yvector"].squeeze().view(shape[0],shape[1],-1)
    assert image.shape[0] == location.shape[0] == xvector.shape[0] == yvector.shape[0]
    assert image.shape[1] == location.shape[1] == xvector.shape[1] == yvector.shape[1]
    if not NMS:
        corners = location.nonzero()
    else:
        corners=poolingNMS(location,extractionNum=config.EXTRACTIONNUM)
    colorOfCorners = [255, 0.0, 0.0]
    colorOfCenter = [255, 255, 255]
    colorOfLines = [0, 255, 0]
    for i in corners:
        predictedCenter=getPredictedCenter(i,xvector,yvector)
        drawDot(image, i, colorOfCorners)
        drawDot(image, predictedCenter, colorOfCenter)
        drawLine(image, i, predictedCenter, colorOfLines)
    return image

def poolingNMS(featureMap: torch.Tensor,extractionNum:int):
    '''
    NMS
    :param featureMap:
    :return: indices of the extracted corners
    '''

    with torch.no_grad():
        if len(featureMap.shape)==3:
            featureMap=featureMap.unsqueeze(dim=0)
            featureMap=featureMap.permute(0,3,1,2)

        pooling = nn.MaxPool2d(3, stride=3, return_indices=True)
        shape=featureMap.shape
        featureMap,indices=pooling(featureMap)
        unpooling = nn.MaxUnpool2d(3, stride=3)
        featureMap = unpooling(featureMap, indices, output_size=shape)

    featureMap=featureMap.squeeze()
    assert len(featureMap.shape)==2

    sorted,_=torch.sort(torch.flatten(featureMap))
    threshold=sorted[-extractionNum]
    result=torch.nonzero((featureMap>threshold).float())
    assert len(result)<extractionNum
    return result

def showGroupingResult(image,result_dict):
    image = image.squeeze().numpy().copy()
    colors = plt.cm.Spectral(np.linspace(0, 1, len(result_dict.keys())))
    for i,col in zip(result_dict.keys(),colors):
        if i==-1:
            color=[0,0,255]
        else:
            color=[ int(255*i) for i in col[:3]]
        for j in range(len(result_dict[i])):
            drawDot(image, result_dict[i][j], color)
            drawLine(image,result_dict[i][j],result_dict[i][(j+1)%len(result_dict[i])],color)
    return image

def showGTExample(image_path,gt_path):
    image=io.imread(image_path)
    gt=[]
    with open(gt_path) as f:
        tmp=f.readline()
        while tmp:
            gt.append([[i[1],i[0]] for i in eval(tmp.strip())])

            tmp=f.readline()
    colors = plt.cm.Spectral(np.linspace(0, 1, len(gt)))
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            drawDot(image,gt[i][j],colors[i][:3]*255)
            drawLine(image,gt[i][j],gt[i][(j+1)%len(gt[i])],colors[i][:3]*255)
    return image

def colorbar(axs):
    return inset_axes(axs,
                        width="5%",  # width = 5% of parent_bbox width
                        height="100%",  # height : 50%
                        loc='lower left',
                        bbox_to_anchor=(1.05, 0, 1.0, 1.0),
                        bbox_transform=axs.transAxes,
                        borderpad=0,
                        )


