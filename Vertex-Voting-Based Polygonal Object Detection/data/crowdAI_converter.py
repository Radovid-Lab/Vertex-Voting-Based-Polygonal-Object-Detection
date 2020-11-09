from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import argparse
import shutil
import math

'''
Please make sure the data is arranged in following format:
|-- data/
|   |-- test_images/ (has all images for prediction)
|   |-- train/
|   |   |-- images (has all the images for training)
|   |   |__ annotation.json : Annotation of the data in MS COCO format
|   |   |__ annotation-small.json : Smaller version of the previous dataset
|   |-- val/
|   |   |-- images (has all the images for training)
|   |   |__ annotation.json : Annotation of the data in MS COCO format
|   |   |__ annotation-small.json : Smaller version of the previous dataset
'''


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a','--animation',action='store_true',help='show animation or not, will slow down program')
	parser.add_argument('-p','--path',type=str,default='./',help='path of CrowdAI data')
	args = parser.parse_args()
	data_directory = args.path
	annotation_file_template = "{}/{}/annotation{}.json"

	TRAIN_IMAGES_DIRECTORY = data_directory+"/train/images"
	TRAIN_ANNOTATIONS_SMALL_PATH = data_directory+"/train/annotation-small.json"
	TRAIN_ANNOTATIONS_PATH = data_directory+"/train/annotation.json"

	VAL_IMAGES_DIRECTORY = data_directory+"/val/images"
	VAL_ANNOTATIONS_SMALL_PATH = data_directory+"/val/annotation-small.json"
	VAL_ANNOTATIONS_PATH = data_directory+"/val/annotation.json"

	if not os.path.exists(TRAIN_IMAGES_DIRECTORY) or not os.path.exists(TRAIN_ANNOTATIONS_PATH) or not os.path.exists(VAL_IMAGES_DIRECTORY) or not os.path.exists(VAL_ANNOTATIONS_PATH):
		print('cannot find directory!')
		return 
	
	for imgpath,annopath in [[TRAIN_IMAGES_DIRECTORY,TRAIN_ANNOTATIONS_PATH],[VAL_IMAGES_DIRECTORY,VAL_ANNOTATIONS_PATH]]:
		print('---------------------------------------')
		print(f'now processing {imgpath}, {annopath}')
		# load annotation into memory
		coco = COCO(annopath)

		# This generates a list of all `image_ids` available in the dataset
		image_ids = coco.getImgIds(catIds=coco.getCatIds())
		
		count=0

		for i in image_ids:
			if count%1000==0:
				print(f'processing {count/len(image_ids)}% images...')
			count+=1
			img = coco.loadImgs(i)[0]
			imgname=img["file_name"]
			if len(imgname.split('.'))==2:
				imgname=imgname.split('.')[0]
			annotation_ids = coco.getAnnIds(imgIds=img['id'])
			annotations = coco.loadAnns(annotation_ids)
			shutil.move(os.path.join(imgpath, img["file_name"]),os.path.dirname(imgpath))
			for j in annotations:
				segmentation=j["segmentation"][0]
				assert len(segmentation)%2==0,'segmentation error'
				segmentation=[[math.floor(segmentation[i]),math.floor(segmentation[i+1])] for i in range(0,len(segmentation)-1,2)]
				with open(os.path.join(os.path.dirname(imgpath),imgname+'.txt'),'a') as f:
					f.write(str(segmentation)+'\n')

			if args.animation:
				plt.imshow(io.imread(os.path.join(imgpath, img["file_name"])));plt.axis('off')
				coco.showAnns(annotations)
				plt.show()
				plt.pause(0.01)
				plt.close()

	os.rmdir(TRAIN_IMAGES_DIRECTORY)
	os.remove(TRAIN_ANNOTATIONS_PATH)
	os.remove(TRAIN_ANNOTATIONS_SMALL_PATH)
	os.rmdir(VAL_IMAGES_DIRECTORY)
	os.remove(VAL_ANNOTATIONS_PATH)
	os.remove(VAL_ANNOTATIONS_SMALL_PATH)

	return


if __name__ == '__main__':
	main()