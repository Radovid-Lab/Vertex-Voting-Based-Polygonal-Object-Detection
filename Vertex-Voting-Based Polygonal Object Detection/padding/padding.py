import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil

def pad_texture(texture_paths,directory,outputdir):
    def recal_mask(points):
        dim0=[i[0] for i in points]
        dim1=[i[1] for i in points]
        center=[int(np.sum(dim0)/len(dim0)),int(np.sum(dim1)/len(dim1))]
        result=[]
        for i in points:
            vec=center-i
            newi=1/3*vec+i
            result.append([int(newi[0]),int(newi[1])])
        return result
    
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    textures=[]
    for i in texture_paths:
        textures.append(cv2.imread(i))
    file_list=glob.glob(os.path.join(directory,'*.png'))
    file_list.sort()
    for i in file_list:
        num=np.random.choice(len(textures),1)[0]
        print(f'processing {i}')
        texture=textures[num]
        img=cv2.imread(i)
        shape=img.shape
        texture=cv2.resize(texture,(shape[0],shape[1]))
        mask=np.zeros_like(img)
        txt_name=i.replace('.png','.txt')
        npz_name=i.replace('.png','_label.npz')
        with open(txt_name) as f:
            for line in f.readlines():
                points=np.array([[p[1],p[0]] for p in eval(line.split('|')[1])])
                points=np.array(recal_mask(points))
                cv2.fillPoly(mask,[points],(255,255,255))
        mask_inv=cv2.bitwise_not(mask)
        padding= cv2.bitwise_and(texture,texture,mask = mask[:,:,0])
        img = cv2.bitwise_and(img,img,mask = mask_inv[:,:,0])
        img=cv2.add(img,padding)
#         res = img[:,:,::-1]
        img_name=i.split('/')[-1]
#         txt_name=txt_name.split('/')[-1]
#         npz_name=npz_name.split('/')[-1]
        cv2.imwrite(os.path.join(outputdir,img_name),img)
        shutil.copy(txt_name,outputdir)
        shutil.copy(npz_name,outputdir)
    return  

        
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--path",type=str,help="path of images to be padded",metavar="str")
    parser.add_argument("--out",type=str,help="path of outputdir",metavar="str")
    texture_list=['./lava.png','./cloud.jpg','./brick.jpeg']
    
    args=parser.parse_args()
    pad_texture(texture_list,args.path,args.out)
    
    return 
if __name__=='__main__':
    main()
