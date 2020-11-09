import shutil
from typing import List
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch import nn
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from collections import defaultdict
from functools import reduce
from shapely.geometry import Polygon
import os

EXTRACTIONNUM_CENTER=100
EXTRACTIONNUM=100
EPSILON=10


def getPredictedCenter(corner,xoffset_map,yoffset_map):
    '''
    for each corner, an object center is predicted by combining xoffset_map and yoffset_map and corner coordinates.
    :param corners: corner coordinates
    :param xoffset_map: feature map for x offset
    :param yoffset_map: feature map for y offset
    :return: predicted object center, x&y coordinates are floatTensors
    '''
    shape=xoffset_map.shape
    predictedCenter = [corner[0] + xoffset_map[corner[0], corner[1]], corner[1] + yoffset_map[corner[0], corner[1]]]
    predictedCenter[0] = max(predictedCenter[0], 0)
    predictedCenter[0] = min(predictedCenter[0], np.float32(shape[0]))
    predictedCenter[1] = max(predictedCenter[1], 0)
    predictedCenter[1] = min(predictedCenter[1], np.float32(shape[1]-1))
    return predictedCenter

def postprocess(im_info, feature_maps, outdir, NMS=False, plot=False):
    '''
    Visualize prediction
    when applied on gt, set NMS to be False.
    when applied on prediction, set NMS to be True
    '''
    image, image_name=im_info
    center, corner, corner_offset, corner_bin_offset=feature_maps
    shape=center.shape
    # results are (1, 128, 128) (1, 128, 128) (1, 2, 128, 128) (1, 2, 128, 128)
    corner=corner.squeeze()
    center=center.squeeze()
    xvector = corner_offset.squeeze()[0]
    yvector = corner_offset.squeeze()[1]
    assert center.shape[0] == xvector.shape[0] == yvector.shape[0]
    assert center.shape[1] == xvector.shape[1] == yvector.shape[1]
    if not NMS:
        corners = list(zip(*corner.nonzero()))
        centers = list(zip(*center.nonzero()))
    else:
        corners= poolingNMS(corner,extractionNum=EXTRACTIONNUM)
        # get predicted centers for ploting
        centers =poolingNMS(center,extractionNum=EXTRACTIONNUM_CENTER)
        clustered = DBSC(centers, epsilon=EPSILON, minimumSamples=1)
        centers = []
        for i in clustered.keys():
            max = clustered[i][0]
            for j in clustered[i]:
                if center[j[0], j[1]] > center[max[0], max[1]]:
                    max = j
            centers.append(max)

    if plot:
        image_resize=resize(image,(shape[1],shape[2]),anti_aliasing=True) # 128,128,3
        plt.imshow(image_resize.astype(int))
        for i in corners:
            predictedCenter=getPredictedCenter(i,xvector,yvector)
            plt.scatter(i[1], i[0], s=20, color='red')
            plt.plot(*list(zip(i[::-1], predictedCenter[::-1])), color='springgreen', linewidth=2)
            plt.scatter(predictedCenter[1], predictedCenter[0], s=20, color='blue')
        for i in centers:
            plt.scatter(i[1], i[0], s=20, color='yellow')
        print('saved as',os.path.join(outdir,image_name))
        plt.savefig(os.path.join(outdir,image_name))
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

    grouped_corners=center_grouping(corner,xvector,yvector,center,NMS=NMS)
    grouped_corners=remap(grouped_corners,corner_bin_offset,image.shape) # remap grouped points to original image space
    if plot:
        image_name=image_name.split('.')[0]+'_poly.'+image_name.split('.')[1]
        showGroupingResult(image, grouped_corners,os.path.join(outdir,image_name))

    return grouped_corners

def remap(grouped_corners,corner_bin_offset,im_shape):
    corner_bin_offset=corner_bin_offset.squeeze()
    for i in grouped_corners:
        for j in range(len(grouped_corners[i])):
            tmp0 =grouped_corners[i][j][0]+corner_bin_offset[0][grouped_corners[i][j][0],grouped_corners[i][j][1]]+ 0.5
            tmp1 = grouped_corners[i][j][1] + corner_bin_offset[1][grouped_corners[i][j][0], grouped_corners[i][j][1]] + 0.5
            grouped_corners[i][j][0] = int(tmp0*im_shape[0] / 128)
            grouped_corners[i][j][1] = int(tmp1*im_shape[1] / 128)
    return grouped_corners

def poolingNMS(featureMap ,extractionNum:int):
    '''
    NMS
    :param featureMap:[128,128] ndarray
    :return: indices of the extracted corners
    '''
    row,col=featureMap.shape
    with torch.no_grad():
        featureMap=torch.from_numpy(featureMap).reshape(1,1,row,col)

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
    return result.tolist()

def showGroupingResult(image,result_dict,savepath=None):
    plt.imshow(image.astype(int))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(result_dict.keys())))
    for i,col in zip(result_dict.keys(),colors):
        for j in range(len(result_dict[i])):
            plt.scatter(result_dict[i][j][1],result_dict[i][j][0],color=col)
            plt.plot(*list(zip(result_dict[i][j][::-1],result_dict[i][(j+1)%len(result_dict[i])][::-1])),color=col)
    if savepath:
        print('saved as',savepath)
        plt.savefig(savepath)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    return

def DBSC(data,epsilon = 1,minimumSamples = 1):
    '''
    apply DBSCAN on dataset
    :param data: predicted object centers
    :param epsilon: DBSCAN parameter
    :param minimumSamples: DBSCAN parameter
    :return: clustered points in dictionay form
    '''
    cluster = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(data)
    labels = cluster.labels_
    # remove outliers
    core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
    core_samples_mask[cluster.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f'number of clusters {n_clusters_}, number of noise {n_noise_}')
    dict=defaultdict(lambda :'None')
    for i in range(len(labels)):
        if core_samples_mask[i]==True:
            if dict[labels[i]]=='None':
                dict[labels[i]]=[data[i]]
            else:
                dict[labels[i]].append(data[i])
    return dict

def center_grouping(cornermap,xoffset,yoffset,centermap,NMS=False):
    """
    group corners together if their predicted centers are close
    :param cornermap:
    :param xoffset:
    :param yoffset:
    :param centermap:
    :param NMS: whether use NMS to select corners&centers or not. set False when using gt
    :return: dict{center:[corner0,corner1...]}
    """
    if NMS:
        center =  poolingNMS(centermap,extractionNum=EXTRACTIONNUM_CENTER)
        location_indices = poolingNMS(cornermap, extractionNum=EXTRACTIONNUM)
    else:
        center = list(zip(*centermap.nonzero()))
        location_indices = list(zip(*cornermap.nonzero()))

    dict = {}

    for i in location_indices:
        predictedCenter = getPredictedCenter(i, xoffset, yoffset)
        predictedCenter = [np.round(j) for j in predictedCenter]
        predictedCenter = tuple(predictedCenter)
        if predictedCenter in dict:
            dict[predictedCenter].append(i)
        else:
            dict[predictedCenter] = [i]

    # obtain centers
    clustered = DBSC(center,epsilon=EPSILON,minimumSamples=1)

    centers=[]
    for i in clustered.keys():
        max=clustered[i][0]
        for j in clustered[i]:
            if centermap[j[0],j[1]]>centermap[max[0],max[1]]:
                max=j
        centers.append(max)

    predictedCenters = np.array(list(dict.keys()))
    result = {}
    for i in centers:
        result[tuple(i)]=[]
    for i in predictedCenters:
        tmp=None
        min=float('inf')
        for k in dict[tuple(i)]:
            for j in centers:
                a = dotproduct(subtract(i, k), subtract(j, k))
                b=length(subtract(j,k))
                projection_length=a/(b+1e-6)
                # filter out the outliers
                if projection_length*2>length(subtract(j,k)) and length(subtract(i,j))<min:
                    tmp=j
                    min=length(subtract(i,j))
            if tmp is not None:
                result[tuple(tmp)].append(list(k))

    new_result={}
    for i in result.keys():
        # if number of points is less than 3, polygon cannot be formed
        if len(result[i])<3:
            continue

        # if all points are on the same line, they cannot form a polygon
        check_line = result[i]
        if len(set([p[0] for p in check_line])) == 1 or len(set([p[1] for p in check_line])) == 1:
            continue
        else:
            new_result[i] = graham(result[i])
            # # a combination of clockwise linking and graham scan.
            # # Graham scan is used if clockwise linking cannot get valid polygon
            # new_result[i] = clockwise_sort(list(i),result[i])
            # if not Polygon(new_result[i]).is_valid:
            #     new_result[i]=graham(result[i])

    return new_result

def graham(points):
    '''
    Returns points on convex hull in CCW order according to Graham's scan algorithm.
    By Tom Switzer <thomas.switzer@gmail.com>.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) == TURN_RIGHT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    points=[list(map(int,i)) for i in points] # convert from numpy.int64 to int
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l

def clockwise_sort(centerpoint,cornerpoints):
    v1=[1,0]
    dict={}
    for i in cornerpoints:
        v2=subtract(i,centerpoint)
        angle=angle_between(v1,v2)
        if angle not in dict:
            dict[angle]=i
        elif length(v2)>length(subtract(v1,dict[angle])):
                dict[angle]=i
    result=[]
    for i in sorted(dict.items()):
        result.append(i[1])

    return result

# Math functions
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def scalarproduct(a,v):
    return [a*i for i in v]

def length(v):
    return math.sqrt(dotproduct(v, v))

def subtract(v1,v2):
    return [v1[0]-v2[0],v1[1]-v2[1]]

def add_vec(v1,v2):
    return [v1[0]+v2[0],v1[1]+v2[1]]

def angle_radian(v1,v2):
    eps=1/float('inf')
    return math.acos(dotproduct(v1,v2)/(length(v1)*length(v2)+eps))

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))



