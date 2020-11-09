import random
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
        print(f'multiplier for the loss of different branches: {M.loss_weight}')

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

            center = output[0 : offset[0]].reshape(n_jtyp, 1, batch, row, col) # 1, 1, 6, 128, 128
            corner = output[offset[0]: offset[1]].reshape(n_jtyp, 2, batch, row, col)  # 1, 2, 6, 128, 128
            tmp_corner_offset = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col).tanh() # 1, 2, 6, 128, 128
            corner_offset=torch.zeros_like(tmp_corner_offset) # 1, 2, 6, 128, 128
            for t in range(n_jtyp):
                corner_offset[t][0] = tmp_corner_offset[t][0]*row
                corner_offset[t][1] = tmp_corner_offset[t][1]*col
            corner_bin_offset = output[offset[2]: offset[3]].reshape(n_jtyp, 2, batch, row, col)  # 1, 2, 6, 128, 128
            if stack == 0:
                result["preds"] = {
                    "center": center.permute(2, 0, 1, 3, 4)[:, :, 0], # 6, 1, 128, 128
                    "corner": corner.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1], # 6, 1, 128, 128
                    "corner_offset": corner_offset.permute(2, 0, 1, 3, 4), # 6, 1, 2, 128, 128
                    "corner_bin_offset": corner_bin_offset.permute(2, 0, 1, 3, 4).sigmoid() - 0.5 # 6, 1, 2, 128, 128
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()

            L["center"] = sum(
                # cross_entropy_loss(center[i], T["center"][i]) for i in range(n_jtyp)
                triplet_loss(center[i],T["corner"][i],T["corner_offset"][i]) for i in range(n_jtyp)
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
                    # print(L["corner_offset"].type())
                    # print(L["corner_offset"])
                    # print(L["center"].type())
                    # print(L["center"])
                    # input()
                    L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)

        result["losses"] = losses
        return result

def triplet_loss(pd,corner_map,offset_map):
    # pd [1,batch_size,128,128]
    # corner_map [batch_size,128,128]
    # offset_map [x_y_map,batch_size,128,128]
    MARGIN=10

    shape=pd[0][0].shape # 128,128
    batch=pd[0].shape[0]

    batch_loss=torch.zeros(1,requires_grad=True).to(pd.device)
    for i in range(batch):
        batch_pd=pd[0][i]
        voting_map=offset_map[0][i]+offset_map[1][i]
        coord_corners=torch.nonzero(corner_map[i],as_tuple =False)
        for j in coord_corners:
            voting_map[j[0],j[1]]+=j[0]+j[1]
        centers=torch.unique(voting_map)
        per_img_loss=torch.zeros(1,requires_grad=True).to(centers.device)
        for j in centers:
            # triplet loss here
            # L(a, p, n) = max(0, D(a, p) — D(a, n) + margin)
            if j==0 and len(centers)>1:
                anchor,positive=random.sample(torch.nonzero(voting_map==j,as_tuple=False).tolist(),2)
                negative=random.choice(torch.nonzero(voting_map!=j,as_tuple=False).tolist())
                Dap= (batch_pd[anchor[0],anchor[1]]-batch_pd[positive[0],positive[1]])**2
                Dan= (batch_pd[anchor[0],anchor[1]]-batch_pd[negative[0],negative[1]])**2
                cur_triplet_loss = max(Dap - Dan + MARGIN, torch.zeros_like(Dap - Dan + MARGIN))
                per_img_loss+=cur_triplet_loss
                continue

            positive_corners=torch.nonzero(voting_map==j,as_tuple=False)
            # print('positive corners',positive_corners)
            cur_positives=positive_corners.tolist()
            if len(cur_positives)>1:
                anchor,positive=random.sample(cur_positives,2)
            else:
                anchor=positive = random.choice(cur_positives)
            # triplet loss with non corners
            negative=random.choice(torch.nonzero(voting_map==0,as_tuple=False).tolist())
            # print('anchor',anchor,'positive',positive,'negative',negative)
            Dap= (batch_pd[anchor[0],anchor[1]]-batch_pd[positive[0],positive[1]])**2
            Dan= (batch_pd[anchor[0],anchor[1]]-batch_pd[negative[0],negative[1]])**2
            cur_triplet_loss = max(Dap - Dan + MARGIN, torch.zeros_like(Dap - Dan + MARGIN))
            if len(centers)>2: # including 0
                # triplet loss with other corners
                negative_corners = torch.nonzero((voting_map != 0) * (voting_map != j),as_tuple=False)
                # print('negative corners', negative_corners)
                negative=random.choice(negative_corners.tolist())
                Dan= (batch_pd[anchor[0],anchor[1]]-batch_pd[negative[0],negative[1]])**2
                cur_triplet_loss = cur_triplet_loss + max(Dap-Dan+MARGIN,torch.zeros_like(Dap-Dan+MARGIN))
                cur_triplet_loss = cur_triplet_loss / 2 # average
                # print('anchor', anchor, 'positive', positive, 'negative', negative)
            per_img_loss+=cur_triplet_loss
        per_img_loss=per_img_loss/len(centers)
        batch_loss+=per_img_loss
    return batch_loss/batch


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    #logits [2,6,128,128]; positive [6,128,128]
    ratio=0.0001
    nlogp = -F.log_softmax(logits, dim=0)

    return (positive * nlogp[1]*(1/ratio) + (1 - positive) * nlogp[0]).mean(2).mean(1)


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
    ratio=0.0001
    SmoothL1 = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='none')

    maskpos = (target != 0).float() * (1/ratio)
    maskneg = (target == 0).float() 
    loss = maskpos * SmoothL1(input, target) + maskneg * SmoothL1(input, target) # [ch,128,128]

    return loss.mean(2).mean(1) # [ch,1]
