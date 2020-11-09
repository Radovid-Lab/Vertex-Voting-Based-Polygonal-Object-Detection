#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )
Examples:
    python3 dataset/wireframe.py /datadir/wireframe data/wireframe
Arguments:
    <src>                Original data directory
    <dst>                Directory of the output
Options:
   -h --help             Show this screen.
"""

import os
import xml.etree.ElementTree as ET
import sys
import json
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


def save_heatmap(prefix, image, input_annotation):
    annotation = copy.deepcopy(input_annotation)
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)
    fx, fy = heatmap_scale[0] / image.shape[0], heatmap_scale[1] / image.shape[1]
    center = np.zeros((1,) + heatmap_scale, dtype=np.float32)  # [1,128,128]
    corner = np.zeros((1,) + heatmap_scale, dtype=np.float32) # [1,128,128]
    corner_offset = np.zeros((1, 2) + heatmap_scale, dtype=np.float32) # [1,2,128,128]
    corner_bin_offset=np.zeros((1, 2) + heatmap_scale, dtype=np.float32) # [1,2,128,128]

    for i in annotation:
        annotation[i]=[[j[0]*fx,j[1]*fy] for j in annotation[i]] #[[np.clip(j[0]*fx,0,heatmap_scale[0] - 1e-4),np.clip(j[1] * fy, 0, heatmap_scale[1] - 1e-4)] for j in annotation[i]]
        center_on_heatmap=[i[0] * fx,i[1] * fy]
        if 0<=int(center_on_heatmap[0])<heatmap_scale[0] and 0<=int(center_on_heatmap[1])<heatmap_scale[1]:
            center[0,int(center_on_heatmap[0]),int(center_on_heatmap[1])] = 1
        for j in annotation[i]:
            if int(j[0])==heatmap_scale[0]:
                j[0]=heatmap_scale[0]-1
            if int(j[1])==heatmap_scale[1]:
                j[1]=heatmap_scale[1]-1
            corner[0,int(j[0]),int(j[1])]=1
            corner_offset[0,:,int(j[0]),int(j[1])]=np.round(center_on_heatmap[0]-j[0]),np.round(center_on_heatmap[1]-j[1])
            corner_bin_offset[0, :,int(j[0]),int(j[1])] = j[0]-int(j[0])-0.5, j[1]-int(j[1])-0.5

    image = cv2.resize(image, im_rescale)

    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        corner=corner,  # [C, H, W]    Corner heat map
        center=center,  # [C, H, W]    Center heat map
        corner_offset=corner_offset, # [C, 2, H, W] corner's offset to object centroid
        corner_bin_offset=corner_bin_offset # [C, 2, H, W] corner's location offset in bins
    )

    cv2.imwrite(f"{prefix}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return True


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in ["train", "valid"]:
        filelist = glob.glob(f"{data_root}/{batch}/*.xml")
        filelist.sort()
        def handle(xmlname):
            iname = xmlname.replace("xml", "jpg")
            image = io.imread(iname).astype(np.float32)[:, :, :3]
            image_size = image.shape
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
            for child_of_root in root.iter(tag='gate_corners'):
                corners = []
                tmp=child_of_root.find('top_left').text.split(',')
                assert len(tmp)==2
                tmp=[image_size[0]-float(tmp[1]),float(tmp[0])]
                if image_size[0]>tmp[0]>=0 and image_size[1]>tmp[1]>=0:
                    corners.append(tmp)

                tmp = child_of_root.find('top_right').text.split(',')
                assert len(tmp) == 2
                tmp = [image_size[0] - float(tmp[1]), float(tmp[0])]
                if image_size[0]>tmp[0]>=0 and image_size[1]>tmp[1]>=0:
                    corners.append(tmp)

                tmp = child_of_root.find('bottom_right').text.split(',')
                assert len(tmp) == 2
                tmp = [image_size[0] - float(tmp[1]), float(tmp[0])]
                if image_size[0]>tmp[0]>=0 and image_size[1]>tmp[1]>=0:
                    corners.append(tmp)

                tmp = child_of_root.find('bottom_left').text.split(',')
                assert len(tmp) == 2
                tmp = [image_size[0] - float(tmp[1]), float(tmp[0])]
                if image_size[0]>tmp[0]>=0 and image_size[1]>tmp[1]>=0:
                    corners.append(tmp)

                tmp = child_of_root.find('center').text.split(',')
                assert len(tmp) == 2
                tmp = [image_size[0] - float(tmp[1]), float(tmp[0])]
                annotation[tuple(tmp)]=corners

            save_heatmap(f"{path}_0", image[::, ::], annotation)
            if batch != "valid":
                annotation1={}
                for i in annotation:
                    annotation1[i[0],image_size[1]-i[1]]=[[j[0],image_size[1]-j[1]] for j in annotation[i]]
                if not save_heatmap(f"{path}_1", image[::, ::-1], annotation1):
                    return

                annotation2={}
                for i in annotation:
                    annotation2[image_size[0]-i[0], i[1]]=[[image_size[0]-j[0], j[1]] for j in annotation[i]]
                if not save_heatmap(f"{path}_2", image[::-1, ::], annotation2):
                    return

                annotation3 ={}
                for i in annotation:
                    annotation3[image_size[0]-i[0], image_size[1]-i[1]]=[[image_size[0]-j[0], image_size[1]-j[1]] for j in annotation[i]]
                if not save_heatmap(f"{path}_3", image[::-1, ::-1], annotation3):
                    return
            print("Finishing", os.path.join(data_output, batch, prefix))

        parmap(handle, filelist, 1)

if __name__ == "__main__":
    main()