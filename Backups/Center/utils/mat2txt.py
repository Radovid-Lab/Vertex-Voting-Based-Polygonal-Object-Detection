from scipy.io import loadmat
import os

def mat2txt(path:str):
    filelist=[ f for f in os.listdir(path) if '.mat' in f]
    for i in filelist:
        filename=i.split('.')[0]
        mat=loadmat(os.path.join(path,i))['gt']
        with open(os.path.join(path,filename+'.txt'),'w') as newfile:
            for object in mat:
                for points in object:
                    points=[ [int(round(point[1])),int(round(point[0]))] for point in points]
                    newfile.write(str(points)+'\n')
    return