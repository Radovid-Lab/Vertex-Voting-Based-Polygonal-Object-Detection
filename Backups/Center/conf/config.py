import random
import numpy as np
import torch
import sys
class Config():
    def __init__(self):
        self.batchsize=None
        self.epoches=None
        self.lr=None
        self.weight_decay=None
        self.path=None
        self.seed=None
        self.image=None
        self.summary=None
        self.tolerance=None
        self.resume = False
        self.EXTRACTIONNUM=300
        self.EXTRACTIONNUM_CENTER=300
        self.use_gpu=torch.cuda.is_available()
        self.device=torch.device("cuda" if self.use_gpu else "cpu")
        print('device', self.use_gpu, self.device)
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        # to do: image size
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_gpu:
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False # Greatly slow down speed
            print('__CUDA VERSION')
            print('__CUDNN VERSION:', torch.backends.cudnn.version())
            print('__Number CUDA Devices:', torch.cuda.device_count())
            print('__Devices')
            print('Active CUDA Device: GPU', torch.cuda.current_device())
            print('Available devices ', torch.cuda.device_count())
            print('Current cuda device ', torch.cuda.current_device())