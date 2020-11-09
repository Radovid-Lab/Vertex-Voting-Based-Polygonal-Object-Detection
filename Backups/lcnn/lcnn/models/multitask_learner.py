from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict["image"]

        outputs, feature = self.backbone(image)
        result = {"feature": feature}

        batch, channel, row, col = outputs[0].shape

        T = input_dict["target"].copy()
        n_jtyp = T["corner"].shape[1] # type of corners/centers

        # import matplotlib.pyplot as plt
        # asd=torch.max_pool2d(image[0],kernel_size=2,stride=2)
        # asd = torch.max_pool2d(asd, kernel_size=2, stride=2)
        # plt.imshow(asd.permute(1,2,0).detach().int())
        # plt.show()
        # plt.imshow(T['center'][0].permute(1, 2, 0).squeeze().detach().int())
        # plt.show()
        # input()

        # switch to CNHW
        for task in ["corner","center"]:
            T[task] = T[task].permute(1, 0, 2, 3) # 1(type),6,128,128
        for task in ["corner_bin_offset",'corner_offset']:
            T[task] = T[task].permute(1, 2, 0, 3, 4) # 1(type),2(x and y),6,128,128


        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        # center,corner,corner_offset,corner_bin_offset
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous() # 8,6,128,128
            center = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col) # 1, 2, 6, 128, 128
            corner = output[offset[0]: offset[1]].reshape(n_jtyp, 2, batch, row, col)  # 1, 2, 6, 128, 128
            tmp_corner_offset = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col).tanh() # 1, 2, 6, 128, 128
            corner_offset=torch.zeros_like(tmp_corner_offset) # 1, 2, 6, 128, 128
            for t in range(n_jtyp):
                corner_offset[t][0] = tmp_corner_offset[t][0]*row
                corner_offset[t][1] = tmp_corner_offset[t][1]*col
            corner_bin_offset = output[offset[2]: offset[3]].reshape(n_jtyp, 2, batch, row, col)  # 1, 2, 6, 128, 128
            if stack == 0:
                result["preds"] = {
                    "center": center.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1], # 6, 1, 128, 128
                    "corner": corner.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1], # 6, 1, 128, 128
                    "corner_offset": corner_offset.permute(2, 0, 1, 3, 4), # 6, 1, 2, 128, 128
                    "corner_bin_offset": corner_bin_offset.permute(2, 0, 1, 3, 4).sigmoid() - 0.5 # 6, 1, 2, 128, 128
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()

            L["center"] = sum(
                cross_entropy_loss(center[i], T["center"][i]) for i in range(n_jtyp)
            )
            L["corner"] = sum(
                cross_entropy_loss(corner[i], T["corner"][i]) for i in range(n_jtyp)
            )
            L["corner_offset"] = sum(
                heatmaplossl1(corner_offset[i,j], T["corner_offset"][i,j])
                for i in range(n_jtyp)
                for j in range(2)
            )
            L["corner_bin_offset"] = sum(
                sigmoid_l1_loss(corner_bin_offset[i, j], T["corner_bin_offset"][i, j], -0.5, T["corner"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                    L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)

        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    #logits [2,6,128,128]; positive [6,128,128]
    nlogp = -F.log_softmax(logits, dim=0)

    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean(2).mean(1)

def heatmaplossl1(input, target):
    '''
    ratio is to control the contribution of positive and negative loss.
    eg. ratio=0.1 means the contribution of negative is 0.1 times of positive
    '''
    ratio=0.0005
    SmoothL1 = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='none')

    maskpos = (target != 0).float()
    maskneg = (target == 0).float() * ratio
    loss = maskpos * SmoothL1(input, target) + maskneg * SmoothL1(input, target) # [ch,128,128]

    return loss.mean(2).mean(1) # [ch,1]