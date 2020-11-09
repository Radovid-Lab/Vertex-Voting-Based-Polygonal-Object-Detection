import cv2
import argparse
import os
import sys


def resize(imageH,imageW,path):
    '''
    resize data in path to imageH*imageW
    :param imageH: new height of image
    :param imageW: new width of image
    :param path: path of data set
    '''
    print(f'new image size, height(vertical):{imageH}, width(horizontal):{imageW}')
    paireddict,filelist=read_images(path)
    for i in filelist:
        print(f'processing image {i}')
        img=cv2.imread(paireddict[i][0])
        ori_height,ori_width,_=img.shape
        img = cv2.resize(img, (imageW,imageH), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(paireddict[i][0],img)
        newtxt=[]
        with open(paireddict[i][1]) as f:
            tmp=f.readline()
            while tmp!="":
                tmp=eval(tmp)
                for j in range(len(tmp)):
                    tmp[j][0]=round(tmp[j][0]/ori_height*imageH)
                    if tmp[j][0]==imageH:
                        tmp[j][0]-=1
                    if tmp[j][0] < 0:
                        tmp[j][0]=0
                    tmp[j][1]=round(tmp[j][1]/ori_width*imageW)
                    if tmp[j][1]==imageW:
                        tmp[j][1]-=1
                    if tmp[j][1]<0:
                        tmp[j][1]=0
                newtxt.append(str(tmp)+'\n')
                tmp=f.readline()
        with open(paireddict[i][1],'w') as f:
            f.writelines(newtxt)
    print('resizing finished')
    return

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--path',type=str,help='path of data',metavar='str')
    parser.add_argument('--imageH',type=int,default=256,help='height of resized image',metavar='int')
    parser.add_argument('--imageW', type=int, default=256, help='width of resized image', metavar='int')
    args = parser.parse_args()
    resize(args.imageH,args.imageW,args.path)

if __name__ == '__main__':
    currentpath = os.getcwd()
    currentpath = '/'.join(currentpath.split('/')[:-1])
    sys.path.append(currentpath)
    from task.DataReader import read_images
    main()
