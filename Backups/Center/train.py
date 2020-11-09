import argparse
import copy
import math
import time
import os

import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from models.network import *
from task.DataReader import ReadDataset


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch size',metavar='int')
    parser.add_argument('-e', '--epoch', type=int, default=100,help='number of epoches to train the model',metavar='int')
    parser.add_argument('-l','--lr',type=float,default=5e-4,help='learning rate',metavar='float')
    parser.add_argument('-w', '--weight_decay', type=float, default=0.0, help='weight decay', metavar='float')
    parser.add_argument('-p', '--path', type=str, default='./Data/', help='path of data',metavar='str')
    parser.add_argument('-s', '--seed', type=int, default=666, help='seed for randomization',metavar='int')
    parser.add_argument('-i','--image',type=int,default=256,help='size(h&w) of image',metavar='int')
    parser.add_argument('-t', '--tolerance', type=int, default=1, help='used for gaussian-aug & PR calculation', metavar='int')
    parser.add_argument('--summary',action='store_true',help='whether output summary')
    parser.add_argument('-r','--resume', action='store_true', help='whether resume from trained model')
    args = parser.parse_args()
    return args

def init()->Config:
    '''
    init config parameters from args
    '''
    config=Config()
    args=parse_command_line()
    config.batchsize=args.batch
    config.epoches=args.epoch
    config.lr=args.lr
    config.weight_decay=args.weight_decay
    config.path=args.path
    config.seed=args.seed
    config.tolerance=args.tolerance
    config.image=args.image
    config.summary=args.summary
    config.resume=args.resume
    return config

def train(model, train_loader: DataLoader, val_loader:DataLoader, config: Config, scheduler, optimizer):
    '''
    model training function
    :param model: network model
    :param train_loader: loader for training set
    :param val_loader: loader for validation set
    :param config: settings
    :param scheduler: scheduler
    :param optimizer: optimizer
    :return: trained model
    '''
    model.train()
    lossRecord = []
    validationLossRecord=[]
    bestLoss=float('inf')
    bestModel=None
    overall_start_time=time.time()
    for epoch in range(config.epoches):
        start_time_epoch = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0): # data = {'original':original,'image':img, 'label':label}
            optimizer.zero_grad() # Set optimizer gradient to zeros

            image=data['image'].to(config.device)
            temp_location=data['label']['location'].to(config.device)
            temp_xvector = data['label']['xvector'].to(config.device)
            temp_yvector = data['label']['yvector'].to(config.device)
            temp_center = data['label']['center'].to(config.device)

            label={'location':temp_location, 'xvector':temp_xvector, 'yvector':temp_yvector, 'center':temp_center}
            assert len(list(image.shape)) == 4 # dimension check
            outputs = model(image)

            loss = model.cal_loss(outputs,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*outputs[-1]['location'].size(0) # add non-averaged loss
        lossRecord.append(running_loss/len(train_loader.dataset))
        end_time_epoch=time.time()
        scheduler.step()
        # calculate validation loss for early stopping
        val_loss=validate(model,val_loader,config)
        validationLossRecord.append(val_loss)
        if val_loss<bestLoss:
            bestLoss=val_loss
            bestModel=copy.deepcopy(model.state_dict())
            if os.path.isfile('./trained_models/trainedModel.pth'):
                os.remove('./trained_models/trainedModel.pth')
            torch.save(bestModel, './trained_models/trainedModel.pth')
        print(f'Epoch.{epoch} loss:{lossRecord[-1]}, time spent:{end_time_epoch-start_time_epoch}, validation loss: {val_loss}')
    overall_end_time=time.time()
    print(f'Training Finished, total time spent {overall_end_time-overall_start_time}')
    xAxis = list(range(len(lossRecord)))
    img, trainingLRPlot = plt.subplots()
    trainingLRPlot.set_title('training loss')
    trainingLRPlot.plot(xAxis, lossRecord, label='training loss', color='r')
    trainingLRPlot.plot(xAxis,validationLossRecord,label='validation loss',color='b')
    plt.legend(loc="upper left")
    trainingLRPlot.set_xlabel('Number of epoches')
    trainingLRPlot.set_ylabel('Loss')
    img.savefig('./trained_models/learning_curve.jpg')
    plt.show()
    return  bestModel

def validate(model,val_loader:DataLoader,config:Config):
    '''
    return validation loss
    '''
    model.eval()
    with torch.no_grad():
        running_loss=0.0
        for i, data in enumerate(val_loader, 0):
            image = data['image'].to(config.device)
            temp_location = data['label']['location'].to(config.device)
            temp_xvector = data['label']['xvector'].to(config.device)
            temp_yvector = data['label']['yvector'].to(config.device)
            temp_center = data['label']['center'].to(config.device)
            label = {'location': temp_location, 'xvector': temp_xvector, 'yvector': temp_yvector,'center':temp_center}
            outputs=model(image)
            loss=model.cal_loss(outputs,label)
            running_loss+=loss.item()*outputs[-1]['location'].size(0)
    return running_loss/len(val_loader.dataset)

def main():
    config=init()
    config.set_seed()
    startTIME = time.asctime(time.localtime(time.time()))
    print('Start time is ' + startTIME)

    model=PolyNet(config)
    if config.resume:
        print('model resumed')
        model.load_state_dict(torch.load('./trained_models/trainedModel.pth'))
    model=model.to(config.device)
    if config.summary:
        summary(model,input_size=(3,config.image,config.image))
        # print number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('num of total parameters', total_params)
        print('num of trainable parameters', train_params)
        print(f'training for {config.epoches} epoches')

    # delete previous model
    if os.path.isfile('./trained_models/trainedModel.pth'):
        os.remove('./trained_models/trainedModel.pth')

    # initialize training dataset
    traindata=ReadDataset(config.path+'train',GaussianRadius=config.tolerance,randomflip=True)
    traindata,valdata=torch.utils.data.random_split(traindata,[math.ceil(len(traindata)*0.9),math.floor(len(traindata)*0.1)])
    print(f'number of training samples: {len(traindata)}, number of validation samples: {len(valdata)}')
    print('-----------------------')
    train_loader = DataLoader(dataset=traindata, batch_size=config.batchsize, shuffle=True, num_workers=1)
    val_loader=DataLoader(dataset=valdata,batch_size=config.batchsize,shuffle=True,num_workers=1)

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # training
    bestmodel=train(model, train_loader,val_loader, config, scheduler, optimizer)
    torch.save(bestmodel, './trained_models/trainedModel.pth')

if __name__ == '__main__':
    main()