from task.visualize import poolingNMS,getPredictedCenter
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from functools import reduce
from utils.misc_math import *
from conf.config import Config
from shapely.geometry import Polygon


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

def grouping(test_result):
    '''
    group corners based on their predicted object center.
    Corners with geometrically close centers will be attributed to the same object.
    :param test_result: dictionary type {'location': location, 'xvector': xvector, 'yvector': yvector}
    :return: grouped corners in dictionary form
    '''
    location = test_result['location'].detach()
    xoffset = test_result['xvector'].squeeze().detach()
    yoffset = test_result['yvector'].squeeze().detach()
    dict={}

    location_indices=[i.numpy() for i in poolingNMS(location)]
    for i in location_indices:
        predictedCenter=getPredictedCenter(i,xoffset,yoffset)
        predictedCenter=[np.round(j.numpy()) for j in predictedCenter]
        predictedCenter=tuple(predictedCenter)
        dict[predictedCenter]=i

    data=np.array(list(dict.keys()))
    clustered=DBSC(data)
    count=0
    for i in clustered.values():
        count+=len(i)
    result={}
    for i in clustered.keys():
        temp_cluster=clustered[i]
        temp_cornerlist=[]
        for j in temp_cluster:
            temp_cornerlist.append(dict[tuple(j)])
        result[i]=temp_cornerlist
    for i in result.keys():
        result[i]=graham([i.tolist() for i in result[i]])#clockwise_sort(list(i),result[i]) #using graham scan
    return result

def center_grouping(test_result,config:Config):
    location = test_result['location'].detach()
    xoffset = test_result['xvector'].squeeze().detach()
    yoffset = test_result['yvector'].squeeze().detach()
    centermap=test_result['center'].squeeze().detach()
    center = [i.numpy() for i in poolingNMS(test_result['center'].detach(),extractionNum=config.EXTRACTIONNUM_CENTER)]

    dict = {}

    location_indices = [i.numpy() for i in poolingNMS(location,extractionNum=config.EXTRACTIONNUM)]
    for i in location_indices:
        predictedCenter = getPredictedCenter(i, xoffset, yoffset)
        predictedCenter = [np.round(j.numpy()) for j in predictedCenter]
        predictedCenter = tuple(predictedCenter)
        dict[predictedCenter] = i

    # obtain centers
    clustered = DBSC(center,epsilon=10,minimumSamples=1)

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
        for j in centers:
            a=dotproduct(subtract(i, dict[tuple(i)]), subtract(j, dict[tuple(i)]))
            b=length(subtract(j,dict[tuple(i)]))
            projection_length=a/(b+1e-6)
            if projection_length*2>length(subtract(j,dict[tuple(i)])) and length(subtract(i,j))<min:
                tmp=j
                min=length(subtract(i,j))
        if tmp is not None:
            result[tuple(tmp)].append(dict[tuple(i)].tolist())

    new_result={}
    for i in result.keys():
        if len(result[i])<3:
            continue
        # line whether points are on a line
        check_line = result[i]
        if len(set([p[0] for p in check_line])) == 1 or len(set([p[1] for p in check_line])) == 1:
            continue

        else:
            # a combination of clockwise linking and graham scan.
            # Graham scan is used if clockwise linking cannot get valid polygon
            new_result[i] = clockwise_sort(list(i),result[i])
            if not Polygon(new_result[i]).is_valid:
                new_result[i]=graham(result[i])

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
