#!/usr/bin/env python
"""Process gate dataset (recalculate corners & centers when original corners & centers are out of image space)
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )
Examples:
    python3 dataset/wireframe_recal.py /datadir/wireframe data/wireframe
Arguments:
    <src>                Original data directory
    <dst>                Directory of the output
    <yaml-config>        Path to the yaml hyper-parameter file
Options:
   -h --help             Show this screen.
"""

import os
import xml.etree.ElementTree as ET
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcnn.postprocess import clockwise_sort
import json
import shapely
from shapely.geometry import LineString, Point
from itertools import combinations
import glob
from skimage import io
import copy
import cv2
import re
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import zoom
from shapely.geometry import Point, Polygon

# hyperparameters:
IM_RESCALE = (512, 512) # resize input image to 512*512
HEATMAP_SCALE = (128, 128) # gt_heatmap size is set to be


try:
    sys.path.append(".")
    sys.path.append("..")
    from lcnn.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))

def convert(point,image_shape):
    return [image_shape[0] - float(point[1]), float(point[0])]

def save_gt_txt(prefix, annotation, child_class, image_shape):
    dst=f"{prefix}.txt"
    fx, fy = IM_RESCALE[0] / image_shape[0], IM_RESCALE[1] / image_shape[1]

    with open(dst, 'w') as f:
        for i in annotation:
            i=annotation[i]
            i = [[int(j[0] * fx), int(j[1] * fy)] for j in i]
            l = []
            for j in i:
                l.append(j)
            l = str(l)
            f.write(f'{child_class}|{l}\n')

def save_heatmap(prefix, image, input_annotation):
    annotation = copy.deepcopy(input_annotation)
    im_rescale=IM_RESCALE
    heatmap_scale = HEATMAP_SCALE
    fx, fy = heatmap_scale[0] / image.shape[0], heatmap_scale[1] / image.shape[1]
    center = np.zeros((1,) + heatmap_scale, dtype=np.float32)  # [1,128,128]
    corner = np.zeros((1,) + heatmap_scale, dtype=np.float32) # [1,128,128]
    corner_offset = np.zeros((1, 2) + heatmap_scale, dtype=np.float32) # [1,2,128,128]
    corner_bin_offset=np.zeros((1, 2) + heatmap_scale, dtype=np.float32) # [1,2,128,128]

    for i in annotation:
        annotation[i]=[[j[0]*fx,j[1]*fy] for j in annotation[i]] #[[np.clip(j[0]*fx,0,heatmap_scale[0] - 1e-4),np.clip(j[1] * fy, 0, heatmap_scale[1] - 1e-4)] for j in annotation[i]]
        center_on_heatmap=[i[0] * fx,i[1] * fy]
        center[0,int(center_on_heatmap[0]),int(center_on_heatmap[1])] = 1
        for j in annotation[i]:
            # if int(j[0])==heatmap_scale[0]:
            #     j[0]=heatmap_scale[0]-1
            # if int(j[1])==heatmap_scale[1]:
            #     j[1]=heatmap_scale[1]-1
            corner[0,int(j[0]),int(j[1])]=1
            corner_offset[0,:,int(j[0]),int(j[1])]=int(center_on_heatmap[0])-int(j[0]),int(center_on_heatmap[1])-int(j[1])
            corner_bin_offset[0, :,int(j[0]),int(j[1])] = j[0]-int(j[0])-0.5, j[1]-int(j[1])-0.5

    image = cv2.resize(image, im_rescale)

    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0], # sth useless
        corner=corner,  # [C, H, W]    Corner heat map
        center=center,  # [C, H, W]    Center heat map
        corner_offset=corner_offset, # [C, 2, H, W] corner's offset to object centroid
        corner_bin_offset=corner_bin_offset # [C, 2, H, W] corner's location offset in bins
    )

    cv2.imwrite(f"{prefix}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return True

def recalculate(corners:dict,image_shape):

    def check_im_corner(poly,image_shape):
        """
        if image corner is within the polygon formed by given corners, it should be counted as a new corner
        :param poly: polygon formed by gate corners
        :param image_shape: shape of image space
        :return:
        """
        im_top_left=Point(0,0)
        im_top_right=Point(0,image_shape[1])
        im_bottom_right=Point(image_shape[0],image_shape[1])
        im_bottom_left=Point(image_shape[0],0)
        result=set()
        for i in [im_top_left,im_top_right,im_bottom_right,im_bottom_left]:
            if i.within(poly):
                result.add(i.coords[0])
        return result

    def cal_intersction(line_corner,line_image):
        result=set()
        int_pt = line_corner.intersection(line_image)
        if int_pt:
            if type(int_pt) == LineString:
                start, end = int_pt.boundary
                start = start.coords[0]
                end = end.coords[0]
                result.add(start)
                result.add(end)
            else:
                point_of_intersection = int_pt.x, int_pt.y
                result.add(point_of_intersection)
            return result
        else:
            return None

    def check_in_image(point,image_shape):
        if 0 <= point[0] < image_shape[0] and 0 <= point[1] < image_shape[1]:
            return True
        else:
            return False

    def cal_intersction_im_line(line,image_shape):
        """
        :param line: border of gate
        :param image_shape: border of image space
        :return: tuple list
        """
        result=set()
        # borders of image
        image_edge_up=LineString([[0,0],[0,image_shape[1]]])
        image_edge_right = LineString([[0, image_shape[1]], [image_shape[0], image_shape[1]]])
        image_edge_down = LineString([[image_shape[0],image_shape[1]],[image_shape[0],0]])
        image_edge_left = LineString([[image_shape[0],0], [0, 0]])
        for i in [image_edge_up,image_edge_right,image_edge_down,image_edge_left]:
            if cal_intersction(line, i):
                result=result.union(cal_intersction(line, i)) # tuple

        return result

    image_shape=[i-1 for i in image_shape] #[128,128]->[127,127] to avoid index out of range error
    new_corner = set() # list to hold recalculated corners
    toggle=True
    for i in [corners['top_left'],corners['top_right'],corners['bottom_right'],corners['bottom_left']]:
        if check_in_image(i,image_shape):
            new_corner.add(tuple(i))
        else:
            toggle=False
    if toggle: # no need to recalculate if all points are in image space
        new_corner = clockwise_sort(corners['center'], new_corner)
        return tuple(corners['center']), new_corner

    tmp=[tuple(i) for i in [corners['top_left'], corners['top_right'], corners['bottom_right'], corners['bottom_left']]]
    poly=Polygon(tmp)
    new_corner=new_corner.union(check_im_corner(poly,image_shape))

    top_line=LineString([corners['top_left'], corners['top_right']])
    right_line=LineString([corners['top_right'],corners['bottom_right']])
    bottom_line=LineString([corners['bottom_right'],corners['bottom_left']])
    left_line=LineString([corners['bottom_left'],corners['top_left']])
    for i in [top_line,right_line,bottom_line,left_line]:
        tmp=cal_intersction_im_line(i, image_shape)
        if len(tmp)!=0:
            new_corner=new_corner.union(tmp)

    assert len(new_corner)>=3,'wrong number of corners'
    new_corner=[list(i) for i in list(new_corner)]
    tmp=[0,0]
    for i in new_corner:
        tmp[0]+=i[0]
        tmp[1]+=i[1]
    tmp[0]/=len(new_corner)
    tmp[1]/=len(new_corner)
    new_center=tmp
    new_corner=clockwise_sort(new_center,new_corner)
    new_center=tuple(new_center)

    return new_center,new_corner

def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    if not os.path.exists(os.path.join(data_root, 'train')):
        print('train set not exists')
    if not os.path.exists(os.path.join(data_root, 'valid')):
        print('valid set not exists')
    for batch in ["train", "valid"]:
        filelist = glob.glob(f"{data_root}/{batch}/*.xml")
        filelist.sort()
        def handle(xmlname):
            iname = xmlname.replace("xml", "jpg")
            image = io.imread(iname).astype(np.float32)[:, :, :3]
            image_size = image.shape # [416,416]
            prefix = xmlname.split(".")[-2].split('/')[-1]
            os.makedirs(os.path.join(data_output, batch), exist_ok=True)
            path = os.path.join(data_output, batch, prefix)
            try:
                tree = ET.parse(xmlname)
                root = tree.getroot()
            except:
                with open(xmlname) as f:
                    xml=f.read()
                root = ET.fromstring("<root>" + xml + "</root>")
            annotation={}
            for child_class_name in root.iter(tag='object'):

                child_class=child_class_name.find('name').text
                if child_class not in annotation:
                    annotation[child_class]={}
                for child_of_root in child_class_name.iter(tag='gate_corners'):

                    original_annotation = {}
                    top_left=child_of_root.find('top_left').text.split(',')
                    assert len(top_left)==2
                    # convert to following coordinate system
                    # |
                    # x
                    # ↓---y--→
                    original_annotation['top_left']=convert(top_left,image_size)

                    top_right = child_of_root.find('top_right').text.split(',')
                    assert len(top_right) == 2
                    original_annotation['top_right'] = convert(top_right,image_size)

                    bottom_right = child_of_root.find('bottom_right').text.split(',')
                    assert len(bottom_right) == 2
                    original_annotation['bottom_right'] = convert(bottom_right,image_size)

                    bottom_left = child_of_root.find('bottom_left').text.split(',')
                    assert len(bottom_left) == 2
                    original_annotation['bottom_left'] = convert(bottom_left,image_size)

                    center = child_of_root.find('center').text.split(',')
                    assert len(center) == 2
                    original_annotation['center'] = convert(center,image_size)

                    newcenter,newcorner=recalculate(original_annotation,image_size)
                    annotation[child_class][newcenter]=newcorner

            assert len(annotation.keys())<=1 # only one class right now
            if len(annotation.keys())==0:
                obj_class=None
                annotation=[]
            else:
                obj_class=list(annotation.keys())[0]
                annotation=annotation[obj_class]

            image_size=[i-1 for i in image_size] # to avoid index out of range error

            save_heatmap(f"{path}_0", image[::, ::], annotation)
            save_gt_txt(f"{path}_0", annotation,obj_class, image.shape)
            if batch != "valid":
                annotation1={}
                for i in annotation:
                    annotation1[i[0],image_size[1]-i[1]]=[[j[0],image_size[1]-j[1]] for j in annotation[i]]
                if not save_heatmap(f"{path}_1", image[::, ::-1], annotation1):
                    return
                else:
                    save_gt_txt(f"{path}_1", annotation1, obj_class, image.shape)

                annotation2={}
                for i in annotation:
                    annotation2[image_size[0]-i[0], i[1]]=[[image_size[0]-j[0], j[1]] for j in annotation[i]]
                if not save_heatmap(f"{path}_2", image[::-1, ::], annotation2):
                    return
                else:
                    save_gt_txt(f"{path}_2", annotation2, obj_class, image.shape)

                annotation3 ={}
                for i in annotation:
                    annotation3[image_size[0]-i[0], image_size[1]-i[1]]=[[image_size[0]-j[0], image_size[1]-j[1]] for j in annotation[i]]
                if not save_heatmap(f"{path}_3", image[::-1, ::-1], annotation3):
                    return
                else:
                    save_gt_txt(f"{path}_3", annotation3, obj_class, image.shape)
            print("Finishing", os.path.join(data_output, batch, prefix))

        parmap(handle, filelist, 1)

if __name__ == "__main__":
    main()