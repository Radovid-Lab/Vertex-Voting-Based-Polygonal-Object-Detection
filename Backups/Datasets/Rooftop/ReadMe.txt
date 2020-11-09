This dataset consists of 65 aerial images of rural scenes containing several building rooftops many of which exhibit a fairly complex polygonal geometry. Each image is of size 1000 * 750 pixels.

The ground truth is saved in Matlab mat format using the same name with corresponding image, in which there a cell (gt), recording the polygonal outlines (n*2 matrix) for all the ground truth objects in the corresponding image.

gtannotate.m is a Matlab function to annotate polygon.

If you use our ground truth outlines, please cite the reference below.

Reference:
X. Sun, C. M. Christoudias and P. Fua. Free-Shape Polygonal Object Localization. European Conference on Computer Vision (ECCV), Zurich, Switzerland, 2014.