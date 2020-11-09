import argparse
import os
from shutil import move
import cv2
import math
'''
Please make sure the data is arranged in following format:
|-- data/
|   |-- test_images/ (has all images for prediction)
|   |-- train/
|   |   |-- images (has all the images for training)
|   |   |__ annotation (in text form)
|   |-- val/
|   |   |-- images (has all the images for training)
|   |   |__ annotation (in text form)
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-p","--path",type=str,default="./Data",help="path of raw data",metavar="str")
    parser.add_argument("--train",type=str,default="train",help="path of training set",metavar="str")
    parser.add_argument("-v","--val",type=str,default="val",help="path of validation set",metavar="str")
    parser.add_argument("--test",type=str,default="test_images",help="path of test set",metavar="str")
    args=parser.parse_args()
    TRAIN_SET=os.path.join(args.path,args.train)
    VAL_SET=os.path.join(args.path,args.val)
    TEST_SET=os.path.join(args.path,args.test)
    assert os.path.isdir(TRAIN_SET) and os.path.isdir(VAL_SET) and os.path.isdir(TEST_SET),'wrong file structure.\nPlease make sure the data is arranged in following format:\n|-- data/\n|   |-- test_images/ (has all images for prediction)\n|   |-- train/\n|   |   |-- images (has all the images for training)\n|   |   |__ annotation (in text form)\n|   |-- val/\n|   |   |-- images (has all the images for training)\n|   |   |__ annotation (in text form)\n'
    os.makedirs(os.path.join(args.path,"images"), exist_ok=True)
    PATH_IMAGE=os.path.join(args.path,"images")
    for cwd in [TRAIN_SET,VAL_SET]:
        print('-----------------------')
        print(f'in direcroty {cwd}')
        imgtype = None
        if cwd==TRAIN_SET:
            jsonname="train.json"
            img_prefix='train-'
        else:
            jsonname="valid.json"
            img_prefix='val-'
        for i in os.listdir(cwd):
            if len(i.split("."))!=2:
                continue
            if i.split(".")[1] in ['jpg','png']:
                imgtype=i.split(".")[1]
                print(f"image type is {imgtype}")
                break    
        print(f'number of image is {math.floor(len(os.listdir(cwd))/2)}')
        write_to_json=[]
        for i in os.listdir(cwd):
            if len(i.split("."))!=2 or i.split(".")[1]!="txt":
                continue

            filename=img_prefix+i
            os.rename(os.path.join(cwd,i),os.path.join(cwd,filename))
            os.rename(os.path.join(cwd,i.split(".")[0]+"."+imgtype),os.path.join(cwd,filename.split('.')[0]+"."+imgtype))
            filename=filename.split(".")[0]
            height,width,_=cv2.imread(os.path.join(cwd,filename+"."+imgtype)).shape
            move(os.path.join(cwd,filename+"."+imgtype),PATH_IMAGE)
            tmp={}
            tmp["filename"]=filename+"."+imgtype
            edges=[]
            with open(os.path.join(cwd,filename+'.txt')) as source:
                for line in source.readlines():
                    points=list(map(lambda t:[float(t[1]),float(t[0])],eval(line)))
                    for point in range(len(points)):
                        edges.append(points[point]+points[(point+1)%len(points)])
                tmp["lines"]=edges
                tmp["height"]=height
                tmp["width"]=width
                write_to_json.append(tmp)
            os.remove(os.path.join(cwd, filename+'.txt'))

        with open(os.path.join(args.path,jsonname),"w") as des:
            des.write(str(write_to_json).replace('\'','\"')) 
        print('-----------------------')

    with open(os.path.join(args.path,"test.txt"),"a") as test:
        img_prefix='test-'
        for i in os.listdir(TEST_SET):
            filename=img_prefix+i
            os.rename(os.path.join(TEST_SET,i),os.path.join(TEST_SET,filename))
            filename=filename.split(".")[0]
            move(os.path.join(TEST_SET,filename+"."+imgtype),PATH_IMAGE)
            test.write(filename+"."+imgtype+"\n")
    os.rmdir(TEST_SET)
    os.rmdir(TRAIN_SET)
    os.rmdir(VAL_SET)
    print('finished')


if __name__=="__main__":
    main()