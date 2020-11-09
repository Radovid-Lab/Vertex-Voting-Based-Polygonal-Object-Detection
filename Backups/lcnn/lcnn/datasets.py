import glob
import json
import math
import os
import random

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

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

        image = np.rollaxis(image, 2).copy()

        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["corner", "center", "corner_offset","corner_bin_offset"]
            }
        return torch.from_numpy(image).float(), target, iname


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        # [b[1] for b in batch],
        default_collate([b[1] for b in batch]),
        default_collate([b[2].split('/')[-1] for b in batch])
    )