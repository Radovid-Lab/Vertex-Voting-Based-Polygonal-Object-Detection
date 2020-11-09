import shutil

from torch.utils.data import DataLoader
from models.network import *
from conf.config import Config
from task.DataReader import ReadDataset,read_images
from train import init
from skimage import io
import os
from task.visualize import visualize,colorbar,showGroupingResult,showGTExample
import matplotlib.pyplot as plt
from task.grouping import grouping,center_grouping
from task.mAP.evaluate import calmAP
from task.mAP.evalCOCO import evalCOCO

def test(model,test_loader:DataLoader,config:Config, datapath:str, debug=False):
    # delete previous result
    dst = './task/mAP/input/'
    cleanfiles(dst)
    if os.path.exists('./visualization_result/'):
        shutil.rmtree('./visualization_result/')
        os.mkdir('./visualization_result/')

    print(f'top {config.EXTRACTIONNUM} corner predictions are extracted in each image')
    print(f'top {config.EXTRACTIONNUM_CENTER} center predictions are extracted in each image')

    paireddict,_=read_images(datapath)# load paired file directionary {filename:[image path,annotation path]}
    for i, data in enumerate(test_loader, 0):
        data['filename']=data['filename'][0]
        original_image = torch.from_numpy(io.imread(paireddict[data['filename']][0]))
        original_image = original_image
        imagetensor=data["image"].to(config.device)
        temp_location = data['label']['location']
        temp_xvector = data['label']['xvector']
        temp_yvector = data['label']['yvector']
        temp_center=data['label']['center']
        label = {'location': temp_location, 'xvector': temp_xvector, 'yvector': temp_yvector, 'center':temp_center}


        fig, axs = plt.subplots(nrows=2, ncols=4, sharex=False)
        plt.tight_layout(pad=1, w_pad=4, h_pad=1.0)

        img1=visualize(original_image,label,config,NMS=False)
        handle=axs[0][0].imshow(label['location'].squeeze())
        axs[0][0].set_title('gt location')
        fig.colorbar(handle, cax=colorbar(axs[0][0]))
        handle = axs[0][1].imshow(label['xvector'].squeeze())
        axs[0][1].set_title('gt x offset')
        fig.colorbar(handle, cax=colorbar(axs[0][1]))
        handle = axs[0][2].imshow(label['center'].squeeze())
        axs[0][2].set_title('center map')
        fig.colorbar(handle, cax=colorbar(axs[0][2]))
        axs[0][3].imshow(img1.squeeze())
        axs[0][3].set_title('Input image')

        model.to(config.device)
        test_result = model(imagetensor)[-1]
        for output in test_result:
            test_result[output]=test_result[output].cpu()
        img2 = visualize(original_image, test_result,config, NMS=True)
        handle = axs[1][0].imshow(test_result['location'].detach().squeeze())
        axs[1][0].set_title('predicted location')
        fig.colorbar(handle, cax=colorbar(axs[1][0]))
        handle = axs[1][1].imshow(test_result['xvector'].detach().squeeze())
        axs[1][1].set_title('predicted x offset')
        fig.colorbar(handle, cax=colorbar(axs[1][1]))
        handle = axs[1][2].imshow(test_result['center'].detach().squeeze())
        axs[1][2].set_title('center map')
        fig.colorbar(handle, cax=colorbar(axs[1][2]))
        axs[1][3].imshow(img2)
        axs[1][3].set_title('predicted')

        # plt.imshow(temp_center.squeeze())
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(test_result['center'].detach().squeeze())
        # plt.colorbar()
        # plt.show()

        if debug:
            plt.show()
        fig.savefig('./visualization_result/'+data['filename']+'.jpg',dpi=300)
        fig.clear()
        plt.close(fig)


        print(data['filename'],':')
        img=showGTExample(paireddict[data['filename']][0],paireddict[data['filename']][1])
        plt.imshow(img)
        plt.savefig('./visualization_result/'+data['filename']+'_gt.jpg',dpi=300)
        grouped_corners=center_grouping(test_result,config)
        grouping_image=showGroupingResult(original_image, grouped_corners)
        plt.imshow(grouping_image)
        plt.savefig('./visualization_result/'+data['filename']+'_gp.jpg')
        plt.clf()
        plt.close()

        if debug:
            plt.show()

        writeResult(dst,grouped_corners,data['filename'],datapath)
    return




######## TBD in future###############
######## TBD in future###############
######## TBD in future###############
def cleanfiles(path):
    dst=os.path.join(path,'detection-results')
    if os.path.exists(dst):
        shutil.rmtree(dst)
        os.mkdir(dst)
    dst = os.path.join(path, 'ground-truth')
    if os.path.exists(dst):
        shutil.rmtree(dst)
        os.mkdir(dst)

def writeResult(dst,grouped_corners,filename,datapath):
    with open(os.path.join(os.path.join(dst, 'detection-results'), filename + '.txt'), 'w') as f:
        for i in grouped_corners.keys():
            l = []
            for j in grouped_corners[i]:
                l.append(j)
            l = str(l)
            f.write(f'polygon|1.0|{l}\n')

    original_corners = []
    with open(os.path.join(datapath,filename+'.txt')) as f:
        tmp = f.readline()
        while tmp != "":
            original_corners.append([[i[1],i[0]] for i in eval(tmp)])
            tmp = f.readline()

    with open(os.path.join(os.path.join(dst, 'ground-truth'), filename + '.txt'), 'w') as f:
        for i in original_corners:
            l = []
            for j in i:
                l.append(j)
            l = str(l)
            f.write(f'polygon|{l}\n')
######## TBD in future###############
######## TBD in future###############
######## TBD in future###############

def main():
    # initialize configuration parameters
    config=init()
    config.set_seed()

    # load trained model
    model= PolyNet(config)
    if os.path.isfile('./trained_models/trainedModel.pth'):

        torch.load('./trained_models/trainedModel.pth', map_location=lambda storage, loc: storage)

        model.load_state_dict(torch.load('./trained_models/trainedModel.pth', map_location=torch.device('cpu')))
    else:
        raise FileNotFoundError

    # initialize testing dataset

    datapath=os.path.join(config.path,'test')
    testdata = ReadDataset(datapath,randomflip=False)
    test_loader = DataLoader(dataset=testdata, batch_size=1, shuffle=True, num_workers=1)
    # testing
    test(model, test_loader, config, datapath)

    print('calculating AP...')
    evalCOCO()# COCO measure for AP evaluation, averaged on 10 iou threshold
    # calmAP()# traditional AP evaluation, AP is computed at a single IoU of .50
    print('test finished')

if __name__ == '__main__':
    main()
