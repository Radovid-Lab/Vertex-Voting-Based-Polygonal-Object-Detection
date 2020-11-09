from torch import nn
from models.layers import Hourglass,Conv
from models.loss import HeatmapLoss
import torch

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        self.pre=Hourglass(3,1,increase=10) #downsample the feature map for 5 times
        self.location=Hourglass(5,1,increase=15)
        self.xVector=Hourglass(5,3,increase=10) #predict the offset from corners to center along x axis
        self.yVector = Hourglass(5,3,increase=10)  # predict the offset from corners to center along y axis
        self.conv= Conv(3, 1, kernel_size=1, stride=1)
        self.conv2 = Conv(3, 1, kernel_size=1, stride=1,relu=False)
        self.conv3 = Conv(3, 1, kernel_size=1, stride=1,relu=False)


    def forward(self,img):
        # x=torch.sigmoid(self.pre(img))
        x=self.conv(img)
        x=self.pre(x)
        fBranch1=torch.sigmoid(self.location(x))
        # fBranch2=self.conv2(self.xVector(x))
        # fBranch3=self.conv3(self.yVector(x))
        # return {"location":fBranch1,"xvector":fBranch2,"yvector":fBranch3}
        return {"location":fBranch1,"xvector":fBranch1,"yvector":fBranch1}


    def cal_loss(self,predict,label,mean=True):
        # calculate the loss. mean represents whether to average loss among all pixels
        # heatmapLoss=HeatmapLoss(weight=10,mean=True)
        # loss1=heatmapLoss(predict["location"],label["location"])
        # loss2=heatmapLoss(predict["xvector"],label["xvector"])
        # loss3=heatmapLoss(predict["yvector"],label["yvector"])

        criterion = nn.BCELoss()  # standard BCEloss
        loss1 = criterion(predict['location'], label['location'])
        return loss1#+loss2+loss3

##############
# below are useless
import torch.nn.functional as F
#
# nn.Sequential(
#             Conv(3, 64, 7, 2, bn=True, relu=True),
#             Residual(64, 128),
#             Pool(2, 2),
#             Residual(128, 128),
#             Residual(128, inp_dim)
#         )

class TestCornerNet(nn.Module):
    def __init__(self):
        super(TestCornerNet, self).__init__()
        self.pre=nn.Sequential(
            Conv(3, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            Conv(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            Conv(16, 3, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.location =nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1),
        )
        self.xvector =nn.Sequential( #2 channels for x and y
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.yvector =nn.Sequential( #2 channels for x and y
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x=self.pre(x)
        branch1=torch.sigmoid(self.location(x))


        # print('b',x.shape)
        # print(torch.max(x.detach()))
        # plt.imshow(x[0].squeeze().permute(1,2,0).detach())
        # plt.title('wenti')
        # plt.show()
        # input()

        branch2=self.xvector(x)*50 #image size
        branch3 = self.yvector(x) * 50  # image size

        return {"location":branch1,"xvector":branch2,"yvector":branch3}

    def cal_loss(self,predict,label,mean=True):
        # calculate the loss. mean represents whether to average loss among all pixels
        heatmapLoss=HeatmapLoss(weight=10,mean=True)
        loss1=heatmapLoss(predict['location'],label["location"])
        loss2=heatmapLoss(predict["xvector"],label["xvector"])
        loss3=heatmapLoss(predict["yvector"],label["yvector"])
        return loss1*200+loss2+loss3


import torch.nn.functional as F
#
# nn.Sequential(
#             Conv(3, 64, 7, 2, bn=True, relu=True),
#             Residual(64, 128),
#             Pool(2, 2),
#             Residual(128, 128),
#             Residual(128, inp_dim)
#         )

class TestCornerNet2(nn.Module):
    def __init__(self):
        super(TestCornerNet2, self).__init__()
        self.pre=nn.Sequential(
            Conv(3, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            Conv(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            Conv(16, 3, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.location =nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1),
        )
        self.xyoffsets =nn.Sequential( #2 channels for x and y
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
        )
        from conf.config import Config
        self.config=Config()
        self.config.image=128

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x=self.pre(x)
        branch1=torch.sigmoid(self.location(x))


        # print('b',x.shape)
        # print(torch.max(x.detach()))
        # plt.imshow(x[0].squeeze().permute(1,2,0).detach())
        # plt.title('wenti')
        # plt.show()
        # input()


        branch2=self.xyoffsets(x)*128 #image size
        return {"location":branch1,"xvector":branch2[:,0,:,:].unsqueeze(dim=1),"yvector":branch2[:,1,:,:].unsqueeze(dim=1)}

    def cal_loss(self,predict,label,mean=True):
        # calculate the loss. mean represents whether to average loss among all pixels


        heatmapLoss=HeatmapLoss(weight=1000,mean=True,config=self.config)
        #loss1=#heatmapLoss(predict['location'],label["location"])
        loss2=heatmapLoss(predict["xvector"],label["xvector"])
        #loss3=heatmapLoss(predict["yvector"],label["yvector"])
        return loss2#loss1*200+loss2+loss3



class TestCornerNet3(nn.Module):
    def __init__(self):
        super(TestCornerNet3, self).__init__()
        self.skip=nn.Sequential(nn.Conv2d(3,3,kernel_size=1,stride=1),
                                nn.ReLU())
        self.stage1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7,stride=1,padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        self.pooling=nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.unpool=nn.MaxUnpool2d(kernel_size=3,stride=3)
        self.stage2=nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=7,stride=1,padding=3),
            nn.ReLU()
        )
        self.end=nn.Conv2d(3,1,kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        s=self.skip(x)
        x=self.stage1(x)
        shape=x.shape
        x,indices=self.pooling(x)
        x=self.unpool(x,indices,output_size=shape)
        x=self.stage2(x)
        x=x+s
        x=self.end(x)
        x=torch.sigmoid(x)
        return {"location":x,"xvector":x,"yvector":x}


    def cal_loss(self,predict,label,mean=True):
        # calculate the loss. mean represents whether to average loss among all pixels
        criterion = nn.BCELoss()  # standard BCEloss
        # criterion= HeatmapLoss(weight=1000,mean=False)
        import matplotlib.pyplot as plt
        # print(predict['location'].size(),label['location'].size())

        # input()
        loss = criterion(predict['location'], label['location'])
        return loss