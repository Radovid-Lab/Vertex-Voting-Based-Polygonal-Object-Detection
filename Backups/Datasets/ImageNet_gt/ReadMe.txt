This dataset includes images from 10 different object categories from ImageNet. They are sign, screen, remote control, cleaver, computer mouse, ipod, wine bottle, mug, beer bottle, and lampshade. We selected around 100 images for each category, and randomly split them into equal-sized training and testing sets.

We do not have coypright to publish the original images, we only provide the ground truth object outlines for corresponding images. To use this dataset, you have to download the input images from ImageNet website(http://www.image-net.org/).  The WordnetIDs for corresponding folders in ImageNet are:
sign:		n06793231 n06794110
screen:		n03085602
remote control:	n04074963
cleaver:	n03041632
computer mouse	n03793489
ipod:		n03584254
whine bottle:	n04591713
mug:		n03797390
beer bottle:	n02823428 
lampshade:	n04380533
 
To download them, please 
1)Open the website using corresponding WordnetID: 
http://www.image-net.org/synset?wnid=XXXX(WordnetID)
e.g. 
beer bottle:	http://www.image-net.org/synset?wnid=n02823428
lampshade:	http://www.image-net.org/synset?wnid=n04380533
sign:		http://www.image-net.org/synset?wnid=n06793231
		http://www.image-net.org/synset?wnid=n06794110
2)Log in. If you don’t have an ImageNet account, please register and obtain the download permission.
3)Click the “Downloads” tab, and then click the download button just below “Download images in the synset”, you will get the tar file for all the images of this category.

The ground truth is saved in Matlab mat format using the same name with corresponding image, in which there a cell (gt), recording the polygonal outlines (n*2 matrix) for all the ground truth objects in the corresponding image.

gtannotate.m is a Matlab function to annotate polygon.

*If you use our ground truth outlines, please cite both ImageNet and the reference below.*

Reference:
X. Sun, C. M. Christoudias and P. Fua. Free-Shape Polygonal Object Localization. European Conference on Computer Vision (ECCV), Zurich, Switzerland, 2014.