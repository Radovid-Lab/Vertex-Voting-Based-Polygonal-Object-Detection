import argparse
import os
import math
import shutil
import random


def split_image(path,ratio,dst_eva,copy=True):
    '''
    spliting data into training set and testing set according to ratio
    Also, images in test set will be moved to eval directory
    :param path: path of data
    :param ratio: splitting ratio
    :param copy: indicate whether copy or move
    :return:
    '''
    # clean previous files in dst directory:
    if path[-1] != '/':
        path += '/'
    dst_train=path + 'train'
    dst_test=path + 'test'
    for i in [dst_eva,dst_test,dst_train]:
        if os.path.exists(i):
            shutil.rmtree(i)
            os.mkdir(i)

    if not os.path.exists(dst_train):
        os.makedirs(dst_train)
    if not os.path.exists(dst_test):
        os.makedirs(dst_test)
    tmp = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random.shuffle(tmp)
    totalnum = len(tmp)
    num_training= math.ceil(totalnum * ratio)
    i=0
    # move files to training directory
    while i < num_training:
        temp_file_fullname=tmp[0]
        temp_file_name=temp_file_fullname.split('.')[0]
        source_list=[temp_file_fullname]
        for j in tmp[1:]:
            if temp_file_name==j.split('.')[0]:
                source_list.append(j)
        for j in source_list:
            tmp.pop(tmp.index(j))
        if copy:
            for j in source_list:
                shutil.copy(path+j,dst_train)
        else:
            for j in source_list:
                shutil.move(path+j,dst_train)
        assert len(source_list)<=2
        i+=len(source_list)
    totaltrain=i
    totaltest=len(tmp)
    # move files to testing directory
    for i in tmp:
        if copy:
            shutil.copy(path+i,dst_test)
        else:
            shutil.move(path+i,dst_test)
        if 'png' in i:
            shutil.copy(path + i, dst_eva)
    print(f'{totalnum} files are distributed into {totaltrain} files for training and {totaltest} for testing')
    return

def main():
    parser = argparse.ArgumentParser()
    currentpath=os.getcwd()
    currentpath=os.path.dirname(currentpath)
    parser.add_argument('-r', '--ratio', type=float, default=0.8, help='ratio of training data',metavar='float')
    parser.add_argument('-p', '--path', type=str, default=os.path.join(currentpath,'Data'), help='path of data',metavar='str')
    parser.add_argument('-e','--eval',type=str,default=os.path.join(currentpath,'task/mAP/input/images-optional'))# copy images to mAP evaluation folder for future use
    args = parser.parse_args()
    split_image(args.path, args.ratio, args.eval)
    print(f'split files in {args.path} according to ratio {args.ratio}')
    return

if __name__ == '__main__':
    main()