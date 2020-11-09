import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from task.visualize import *
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from backup.test_model import *
from task.DataReader import ReadDataset
from models.loss import *
import os

def train(model,data_loader: DataLoader,config:Config, scheduler, optimizer):
    lossRecord = []
    # best_loss=
    # start_time=time.time()
    for epoch in range(config.epoches):
        running_loss = 0.0
        model.train()
        toggle=True
        for i, data in enumerate(data_loader, 0):
            # fetch data

            temp= data # label = {'location':location, 'xvector':xvector, 'yvector':yvector}
            image=temp['image']
            label=temp['label']


            assert len(list(image.shape)) == 4 #dimension check
            # Set gradient to zeros
            optimizer.zero_grad()
            # Training phase

            outputs = model(image) # {"location":fBranch1,"xvector":fBranch2,"yvector":fBranch3}

            if toggle==True:
                toggle=False

                plt.imshow(image[0].permute(1,2,0).detach())
                plt.title('input' + str(i))
                plt.colorbar()
                plt.show()

                plt.imshow(label['xvector'][0][0].detach())
                plt.title('xvector gt'+str(i))
                plt.colorbar()
                plt.show()

                plt.imshow(outputs['xvector'][0][0].detach())
                plt.title('xvector output'+str(i))
                plt.colorbar()
                plt.show()


            #
            # input()




            loss = model.cal_loss(outputs,label)

            # weightedLoss=weightedBCE(positiveWeight=100)
            # loss=weightedLoss(outputs['xvector'],label['xvector'],average=True)#+weightedLoss(outputs['xvector'],label['xvector'],average=True)+weightedLoss(outputs['yvector'],label['yvector'],average=True)

            # criterion = nn.BCELoss()  # standard BCEloss
            # loss = criterion(outputs['location'], label['location'])

            loss.backward()
            optimizer.step()
            # training cruve
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                lossRecord.append(running_loss / 5)
                running_loss = 0.0
                print('current loss:', lossRecord[-1])
        print(f'epoch {epoch} finished')
        print('***********************')
        scheduler.step()
    print('Training Finished')
    xAxis = list(range(len(lossRecord)))
    img , trainingLRPlot = plt.subplots()
    trainingLRPlot.set_title('training loss')
    temp=trainingLRPlot.plot(xAxis, lossRecord)
    img.savefig('./trained_models/learning_curve.jpg')
    plt.show()
    return lossRecord


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--epoch', type=int, default=3,help='number of epoches to train the model')
    parser.add_argument('-p', '--path', type=str, default='/./Data/',
                        help='path of data')
    args = parser.parse_args()
    return args

def init()->Config:
    '''
    reload configurations or read from args
    :return:
    '''
    config=Config()
    args=parse_command_line()
    config.batchsize=args['batch']
    config.epoches=args["epoch"]
    return config

def main():
    random.seed(666)
    np.random.seed(666)
    torch.manual_seed(666)

    testnet=TestCornerNet2()

    #summary(testnet,input_size=(3,128,128))
    config=Config()
    config.path='./Data/'
    config.epoches=50
    config.batchsize=16
    optimizer = optim.Adam(testnet.parameters(),lr=5e-4,weight_decay=0.0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    traindata=ReadDataset(config.path,GaussianRadius=4,train=True)
    retrain=True
    train_loader = DataLoader(dataset=traindata, batch_size=config.batchsize, shuffle=True, num_workers=0)
    if not retrain:
        if os.path.isfile('./trained_models/trainedModel.pth'):
            testnet.load_state_dict(torch.load('./trained_models/trainedModel.pth'))
        else:
            raise FileNotFoundError
    else:
        if os.path.isfile('./trained_models/trainedModel.pth'):
            # testnet.load_state_dict(torch.load('./trained_models/trainedModel.pth'))
            os.remove('./trained_models/trainedModel.pth')
        if os.path.isfile('./trained_models/learning_curve.jpg'):
            os.remove('./trained_models/learning_curve.jpg')
        lossRecord = train(testnet,train_loader, config, scheduler, optimizer)
        torch.save(testnet.state_dict(), './trained_models/trainedModel.pth')
    testdata = ReadDataset(config.path,GaussianRadius=4,train=False)


    test_loader = DataLoader(dataset=testdata, batch_size=1, shuffle=True, num_workers=0)
    dataiter = iter(test_loader)
    for count in range(5):
        for i in range(random.randint(0,int((len(test_loader.dataset) - 1)/3))):
            dataiter.next()
        labels = dataiter.next()  # a randomly selected sample
        image=labels["original"]
        imagetensor=labels['image']
        label=labels["label"]
        testresult=testnet(imagetensor)

        # plt.imshow(image.squeeze())
        # plt.show()
        # plt.imshow(label['location'].detach().squeeze())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(label['xvector'].detach().squeeze())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(label['yvector'].detach().squeeze())
        # plt.colorbar()
        # plt.show()
        # img = visualize(image, label, NMS=False)
        # plt.imshow(img.squeeze())
        # plt.show()

        # plt.imshow(testresult['location'].detach().squeeze(),cmap='jet')
        # plt.colorbar()
        # plt.show()




        img = visualize(image, label, NMS=False)
        plt.imshow(img.squeeze())
        plt.show()
        img=visualize(image,testresult,NMS=True)
        plt.imshow(img.squeeze())
        plt.show()
        # testresult['location'] = label['location']
        # img=visualize(image,testresult,NMS=False)
        # plt.imshow(img.squeeze())
        # if os.path.isfile('./visualization_result/'+str(count)+'.jpg'):
        #     os.remove('./visualization_result/'+str(count)+'.jpg')
        # plt.imsave('./visualization_result/'+str(count)+'.jpg',img.squeeze())
        # plt.show()



if __name__ == '__main__':
    main()