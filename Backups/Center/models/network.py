from torch import nn
from models.layers import *
from models.loss import HeatmapLossSL1,FocalLoss,HeatmapLossMSE
from conf.config import Config
import torch


class PolyNet(nn.Module):
    def __init__(self, config: Config):
        super(PolyNet, self).__init__()
        self.config = config
        self.increase=nn.Conv2d(3,10,kernel_size=3,stride=1,padding=1)
        self.pre = Hourglass(4, 10, increase=10)# downsample 5 times
        self.focalLoss=FocalLoss()
        # self.heatmapLoss = HeatmapLossMSE(weight=100, config=self.config)
        self.heatmapLoss = HeatmapLossSL1()
        # self.criterion = nn.BCELoss()  # standard BCEloss
        self.location = nn.Sequential(
            Conv(kernel_size=3,inp_dim=10,out_dim=30,relu=True),
            nn.Conv2d(in_channels=30,out_channels=1, kernel_size=1, stride=1),
        )

        self.center=nn.Sequential(
            Conv(kernel_size=3, inp_dim=10, out_dim=30, relu=True),
            nn.Conv2d(in_channels=30, out_channels=1, kernel_size=1, stride=1),
        )

        self.xyoffsets =nn.Sequential( #2 channels for x and y
            Conv(kernel_size=3, inp_dim=10, out_dim=30, relu=True),
            nn.Conv2d(in_channels=30, out_channels=2, kernel_size=1, stride=1),
        )
        # self.conv=Conv(inp_dim=3, out_dim=1, kernel_size=1, stride=1, bn=False, relu=False)

        self.location2 = nn.Sequential(
            Conv(kernel_size=3,inp_dim=14,out_dim=30,relu=True),
            nn.Conv2d(in_channels=30,out_channels=1, kernel_size=1, stride=1),
        )

        self.center2 =nn.Sequential(
            Conv(kernel_size=3, inp_dim=14, out_dim=30, relu=True),
            nn.Conv2d(in_channels=30, out_channels=1, kernel_size=1, stride=1),
        )

        self.xyoffsets2 =nn.Sequential( #2 channels for x and y
            Conv(kernel_size=3, inp_dim=14, out_dim=30, relu=True),
            nn.Conv2d(in_channels=30, out_channels=2, kernel_size=1, stride=1),
        )

    def forward(self, img):
        x=self.increase(img)
        afterpre = self.pre(x)
        location = self.location(afterpre)
        location = torch.sigmoid(location)
        center=self.center(afterpre)
        center=torch.sigmoid(center)
        xyoffset=self.xyoffsets(afterpre)
        xoffset,yoffset=torch.split(xyoffset, 1, dim=1)

        secondstage=torch.cat([afterpre,location,center,xoffset,yoffset],1)

        center2=self.center2(secondstage)
        center2=torch.sigmoid(center2)
        location2=self.location2(secondstage)
        location2=torch.sigmoid(location2)
        xyoffset2=self.xyoffsets2(secondstage)
        xoffset2,yoffset2=torch.split(xyoffset2, 1, dim=1)

        stage1={"location": location, "xvector": xoffset, "yvector": yoffset,"center":center}
        stage2={"location": location2, "xvector": xoffset2, "yvector": yoffset2,"center":center2}
        return [stage1,stage2]

    def cal_loss(self, predict, label, mean=True, alpha=100):
        '''
        calculate the loss. location uses BCE loss while offsets use heatmap(mse) loss.
        :param predict: predicted result
        :param label: ground truth
        :param mean: represents whether to average loss among all pixels
        :param alpha: control the contribution of first branch. total loss=alpha*loss1+loss2+loss3
        :return: loss
        '''
        loss=0
        for stage in predict:
            # loss1 = self.criterion(predict['location'], label['location'])
            loss1=self.focalLoss(stage['location'],label['location'],mean=mean)
            loss2=self.heatmapLoss(stage["xvector"],label["xvector"],mean=mean)
            loss3=self.heatmapLoss(stage["yvector"],label["yvector"],mean=mean)
            loss4=self.focalLoss(stage['center'],label['center'],mean=mean)
            stageloss=alpha*loss1+loss2+loss3+alpha*loss4
            loss+=stageloss

        return loss