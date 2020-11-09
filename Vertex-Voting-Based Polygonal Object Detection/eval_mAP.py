#!/usr/bin/env python3
"""Evaluate dataset using mAP
Usage:
    process.py [options] <yaml-config> <checkpoint> <image-dir> <output-dir>
    process.py (-h | --help )

Examples:
    python3 eval_mAP.py config/wireframe.yaml ./checkpoint_best.pth data/wireframe/ logs/test/ --plot
Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <image-dir>                   Path to the directory containing processed images
   <output-dir>                  Path to the output directory

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   --plot                        Plot the result
"""

import os
import shutil
import sys
import shlex
import pprint
import random
import os.path as osp
import threading
import subprocess
from mAP.evaluate import evalCOCO
import yaml
import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import matplotlib.pyplot as plt
from docopt import docopt
import time
import lcnn
from lcnn.utils import recursive_to
from lcnn.config import C, M
# from lcnn.postprocess import postprocess
from lcnn.postprocess_no_center import postprocess
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner

def move_to_mAP(img_des,img_name,output_des):
    npz_name = img_name.replace(".png",".txt")
    npz_name=os.path.join(img_des,npz_name)
    shutil.copy(npz_name, output_des)
    return

def write_pd_to_mAP(results,img_name,output_des):
    npz_name = img_name.replace(".png",".txt")
    with open(os.path.join(output_des, npz_name), 'w') as f:
        for i in results:
            l = []
            for j in results[i]:
                l.append(j)
            l = str(l)
            f.write(f'gate|{i}|{l}\n')  # only one class right now
    return

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if M.backbone == "stacked_hourglass":
        model = lcnn.models.hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
    else:
        raise NotImplementedError

    model = MultitaskLearner(model)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        checkpoint = torch.load(args["<checkpoint>"])
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        checkpoint = torch.load(args["<checkpoint>"],map_location=torch.device('cpu'))
        print("CUDA is not available")
    device = torch.device(device_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        WireframeDataset(args["<image-dir>"], split="valid"),
        shuffle=False,
        batch_size=M.batch_size_eval,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )
    os.path.join(args["<output-dir>"], 'test_result')
    if os.path.exists(args["<output-dir>"]):
        shutil.rmtree(args["<output-dir>"])
    os.makedirs(args["<output-dir>"], exist_ok=False)
    outdir = os.path.join(args["<output-dir>"],'test_result')
    os.mkdir(outdir)

    # clean previous files in mAP folders
    for mAP_folder in [os.path.join(C.io.mAP,'detection-results'),os.path.join(C.io.mAP,'ground-truth')]:
        if os.path.exists(mAP_folder):
            shutil.rmtree(mAP_folder)
        os.makedirs(mAP_folder, exist_ok=False)

    total_inference_time=0
    time_cost_by_network=0
    time_cost_by_post = 0
    for batch_idx, (image, target, iname) in enumerate(loader):
        with torch.no_grad():
            # predict given image
            input_target={
                    "center": torch.zeros_like(target['center']),
                    "corner": torch.zeros_like(target['corner']),
                    "corner_offset": torch.zeros_like(target['corner_offset']),
                    "corner_bin_offset": torch.zeros_like(target['corner_bin_offset'])
            }
            input_dict = {
                "image": recursive_to(image, device),
                "target": recursive_to(input_target, device),
                "mode": "validation",
            }
            # time cost by network
            timer_begin = time.time()
            H = model(input_dict)["preds"]
            timer_end = time.time()
            time_cost_by_network += timer_end - timer_begin
            total_inference_time += timer_end - timer_begin

            # plot prediction
            for i in range(len(iname)): #M.batch_size
                im = image[i].cpu().numpy().transpose(1, 2, 0) # [512,512,3]

                # move gt files to mAP folder for evaluation
                move_to_mAP(os.path.join(args["<image-dir>"],'valid'),iname[i],os.path.join(C.io.mAP,'ground-truth'))

                # plot&process pd
                pd_im_info= [im, iname[i].split('.')[0]+'_pd.'+iname[i].split('.')[1]]
                pd_center = H["center"][i].cpu().numpy()
                pd_corner = H["corner"][i].cpu().numpy()
                pd_corner_offset = H["corner_offset"][i].cpu().numpy()
                pd_corner_bin_offset= H["corner_bin_offset"][i].cpu().numpy()
                feature_maps = [pd_center, pd_corner, pd_corner_offset, pd_corner_bin_offset]
                ## post processing with center prediction
                # grouped_corners=postprocess(pd_im_info, feature_maps, outdir, NMS=True,plot=args['--plot'])
                ## post processing without center prediction
                timer_begin=time.time()
                grouped_corners=postprocess(pd_im_info, feature_maps, outdir, maxDet=10, NMS=True,plot=args['--plot'])
                timer_end=time.time()
                time_cost_by_post+=timer_end-timer_begin
                total_inference_time+=timer_end-timer_begin
                write_pd_to_mAP(grouped_corners, iname[i], os.path.join(C.io.mAP,'detection-results'))
                # print(f'prediction of {iname[i]} finished')

            # Evaluation:
    evalCOCO() # TBD
    print("inference time is", total_inference_time/len(loader.dataset), "s / img")
    print(f"time cost by network is {time_cost_by_network/len(loader.dataset)}, time cost by post-processing is {time_cost_by_post/len(loader.dataset)}")
    print('-----finished-----')
    return

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


if __name__ == "__main__":
    main()

