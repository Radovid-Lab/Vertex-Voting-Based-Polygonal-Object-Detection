import datetime
import glob
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from torchsummary import summary
import numpy as np
import torch
import yaml
from docopt import docopt
import matplotlib.pyplot as plt
import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner

def main():
    device=torch.device('cpu')
    C.update(C.from_yaml(filename="config/wireframe.yaml"))
    M.update(C.model)


    if M.backbone == "stacked_hourglass":
        model = lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
    else:
        raise NotImplementedError

    model = MultitaskLearner(model)

    x=torch.load(('./checkpoint_best.pth'),map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(x)
    model = model.to(torch.device('cpu'))
    model.eval()
    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }

    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid"),
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )
    for batch_idx, (image, target,_) in enumerate(val_loader):
        plt.imshow(image[0].permute(1,2,0).detach().int())
        plt.show()

        with torch.no_grad():
            input_dict = {
                "image": image.to(device),
                "target": {
                    "corner": torch.zeros([1, 1, 128, 128]).to(device),
                    "center": torch.zeros([1, 1, 128, 128]).to(device),
                    "corner_offset": torch.zeros([1, 1, 2, 128, 128]).to(device),
                    "corner_bin_offset": torch.zeros([1, 1, 2, 128, 128]).to(device),
                },
                "mode": "testing",
            }
            H = model(input_dict)["preds"]

            plt.imshow(H['corner'][0].squeeze())
            plt.colorbar()
            plt.title('corner')
            plt.show()

            plt.imshow(H['center'][0].squeeze())
            plt.colorbar()
            plt.title('center')
            plt.show()


            plt.imshow(H['corner_offset'][0][0][0].squeeze())
            plt.colorbar()
            plt.title('corner_offset')
            plt.show()
            plt.imshow(H['corner_offset'][0][0][1].squeeze())
            plt.colorbar()
            plt.title('corner_offset')
            plt.show()

            plt.imshow(H['corner_bin_offset'][0][0][0].squeeze())
            plt.colorbar()
            plt.title('corner_bin_offset')
            plt.show()
            plt.imshow(H['corner_bin_offset'][0][0][1].squeeze())
            plt.colorbar()
            plt.title('corner_bin_offset')
            plt.show()


        # print(result)
        input()


if __name__ == '__main__':
    main()
