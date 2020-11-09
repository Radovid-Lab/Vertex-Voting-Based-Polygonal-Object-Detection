#!/usr/bin/env python3
"""Process a dataset with the trained neural network
Usage:
    process.py [options] <yaml-config> <checkpoint> <image-dir> <output-dir>
    process.py (-h | --help )

Examples:
    python3 process.py config/wireframe.yaml ./checkpoint_best.pth data/wireframe/ logs/test/ --plot
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
from time import time
import yaml
import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import matplotlib.pyplot as plt
from docopt import docopt

import lcnn
from lcnn.utils import recursive_to
from lcnn.config import C, M
from lcnn.postprocess import postprocess
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner


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

    print(f'evaluation batch size {M.batch_size_eval}')
    loader = torch.utils.data.DataLoader(
        WireframeDataset(args["<image-dir>"], split="valid"),
        shuffle=False,
        batch_size=M.batch_size_eval,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )
    if os.path.exists(args["<output-dir>"]):
        shutil.rmtree(args["<output-dir>"])
    os.makedirs(args["<output-dir>"], exist_ok=False)
    outdir = os.path.join(args["<output-dir>"],'test_result')
    os.mkdir(outdir)

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
            network_start_time=time()
            H = model(input_dict)["preds"]
            network_end_time=time()
            # plot gt & prediction
            for i in range(len(iname)): #M.batch_size
                if not args["--plot"]:
                    continue
                im = image[i].cpu().numpy().transpose(1, 2, 0) # [512,512,3]
                # im = im * M.image.stddev + M.image.mean

                # plot&process gt
                gt_im_info= [im, iname[i].split('.')[0]+'_gt.'+iname[i].split('.')[1]]
                gt_center = target["center"][i].cpu().numpy()
                gt_corner = target["corner"][i].cpu().numpy()
                gt_corner_offset = target["corner_offset"][i].cpu().numpy()
                gt_corner_bin_offset= target["corner_bin_offset"][i].cpu().numpy()
                feature_maps = [gt_center, gt_corner, gt_corner_offset, gt_corner_bin_offset]
                postprocess(gt_im_info, feature_maps, outdir, NMS=False,plot=True)
                # plot&process pd
                pd_im_info= [im, iname[i].split('.')[0]+'_pd.'+iname[i].split('.')[1]]
                pd_center = H["center"][i].cpu().numpy()
                pd_corner = H["corner"][i].cpu().numpy()
                pd_corner_offset = H["corner_offset"][i].cpu().numpy()
                pd_corner_bin_offset= H["corner_bin_offset"][i].cpu().numpy()
                feature_maps = [pd_center, pd_corner, pd_corner_offset, pd_corner_bin_offset]
                postprocess_start_time=time()
                grouped_corners=postprocess(pd_im_info, feature_maps, outdir, NMS=True,plot=True)
                postprocess_end_time=time()
                print(f'inference time is {postprocess_end_time-postprocess_start_time+network_end_time-network_start_time}, network cost:{network_end_time-network_start_time}, postprocessing cost:{postprocess_end_time-postprocess_start_time}')

            # Evaluation:
            # eval() # TBD
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
