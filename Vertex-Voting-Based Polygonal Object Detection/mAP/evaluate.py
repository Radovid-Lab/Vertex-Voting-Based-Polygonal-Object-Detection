import glob
import json
import os
import shutil
import sys
import argparse

import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import polygonize, unary_union

# COCO measures for evaluation
# http://cocodataset.org/#detection-eval
# AP is averaged over multiple Intersection over Union (IoU) values. Specifically we use 10 IoU thresholds of .50:.05:.95.
# This is a break from tradition, where AP is computed at a single IoU of .50 (which corresponds to our metric APIoU=.50).
# Averaging over IoUs rewards detectors with better localization.

def checkPolygon(box_1,box_2):
    '''
    check if the two point lists can form polygon
    '''
    if len(box_1) < 3 or len(box_2) < 3: # cannot form polygon if vertices less than 3
        return False
    if len(set([i[0] for i in box_1]))==1 or len(set([i[1] for i in box_1]))==1: # if all vertices are aligned
        return False
    if len(set([i[0] for i in box_2])) == 1 or len(set([i[1] for i in box_2]))==1: # if all vertices are aligned
        return False

    return True


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def evalCOCO():
    # hyperparameters
    IOU_THRESHOLDS= np.linspace(0.5, 0.95, 10)

    def calculate_iou(predicted, gt):
        if not checkPolygon(predicted,gt):
            assert 1==0,'gt or predicted cannot form polygon'
            return 0
        gt_x = [i[0] for i in gt]
        gt_y = [i[1] for i in gt]
        predicted=Polygon(predicted)
        gt=Polygon(gt)
        assert predicted.is_valid,'prediction is not valid'+str(predicted)
        if gt.is_valid:
            iou=predicted.intersection(gt).area/predicted.union(gt).area
        else:
            # original data
            ls = LineString(np.c_[gt_x, gt_y])
            # closed, non-simple
            lr = LineString(ls.coords[:] + ls.coords[0:1])
            assert lr.is_simple is False,'lr is simple'
            mls = unary_union(lr)
            #mls.geom_type  # MultiLineString'
            gt = MultiPolygon(list(polygonize(mls)))
            intersection=0
            union=predicted
            for gt_polygons in gt:
                intersection+=predicted.intersection(gt_polygons).area
                union=union.union(gt_polygons)
            iou= intersection/union.area
        return iou

    '''
        0,0 ------> x (width)
         |
         |  (Left,Top)
         |      *_________
         |      |         |
                |         |
         y      |_________|
      (height)            *
                    (Right,Bottom)
    '''

    # make sure that the cwd() is the location of the python script (so that every path makes sense)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # directory that stores the gt files
    GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
    # directory that stores the prediction results
    DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')


    """
     throw error and exit
    """
    def error(msg):
        print(msg)
        sys.exit(0)

    """
     check if the number is a float between 0.0 and 1.0
    """
    def is_float_between_0_and_1(value):
        try:
            val = float(value)
            if val > 0.0 and val < 1.0:
                return True
            else:
                return False
        except ValueError:
            return False

    """
     Convert the lines of a file to a list
    """
    def file_lines_to_list(path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content

    """
     Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        shutil.rmtree(TEMP_FILES_PATH)
    os.makedirs(TEMP_FILES_PATH)

    output_files_path = "output"
    if os.path.exists(output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)

    os.makedirs(output_files_path)


    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, pointlist, _difficult = line.split('|')
                        pointlist=eval(pointlist)
                        is_difficult = True
                else:
                        class_name, pointlist = line.split('|')
                        pointlist=eval(pointlist)
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <point list: [[],[]...]> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)

            bbox = pointlist
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "pointlist":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "pointlist":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)


        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)


    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            #print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, pointlist = line.split('|')
                    pointlist = eval(pointlist)
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    #print("match")
                    bbox = pointlist
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "pointlist":bbox})
                    #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    os.mkdir(output_files_path + "/classes/")
    sum_for_map=0
    sum_for_mar = 0
    for class_index, class_name in enumerate(gt_classes):
        with open(output_files_path + "/classes/" + f"{class_name}.txt", 'w') as output_file:
            output_file.write(f"# IOU and average precision for class: {class_name} \n")
            """
             Load detection-results of that class
            """
            sum_ap = 0
            sum_ar = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            """
             Assign detection-results to ground-truth objects
            """
            # 10 IOU thresholds
            iou_thresholds = IOU_THRESHOLDS
            precision=[] #avg precision
            recall=[]
            gt_list=glob.glob(f"{TEMP_FILES_PATH}/*_ground_truth.json")
            total_gt=0
            for f in gt_list:
                ground_truth_data = json.load(open(f))
                total_gt+=len(ground_truth_data) # calculate total gt for classname

            for MINOVERLAP in iou_thresholds:
                min_overlap = MINOVERLAP
                nd = len(dr_data)
                count_true_positives = 0 # count number of tp under current iou
                tp = [0]*nd
                fp = [0]*nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]
                    # assign detection-results to ground truth object if any
                    # open ground-truth with that file_id
                    gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bb = detection["pointlist"]
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = obj["pointlist"]

                            ov=calculate_iou(bb,bbgt)
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                    if ovmax >= min_overlap:
                        if "difficult" not in gt_match:
                                if not bool(gt_match["used"]):
                                    # true positive
                                    tp[idx]=1
                                    count_true_positives += 1
                                    gt_match["used"] = True
                                else:
                                    # false positive (multiple detection)
                                    fp[idx] = 1

                    else:
                        # false positive
                        fp[idx] = 1

                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                ap, mrec, mprec = voc_ap(rec[:], prec[:])
                # p=tp/(fp+tp+1e-6)
                sum_ap+=ap
                precision.append(ap)
                output_file.write(f'IOU threshold: {np.round(min_overlap,3)}, Average Precision: {np.round(ap,4)} \n')
                # recall
                r=count_true_positives/(total_gt+1e-6)
                sum_ar+=r
                recall.append(r)
                output_file.write(f'IOU threshold: {np.round(min_overlap,3)}, Recall: {np.round(r,4)} \n')

            sum_for_map+=sum_ap
            sum_for_mar+=sum_ar
            # save precison/iou curve
            plt.plot(iou_thresholds,precision,'-o')
            plt.fill_between(iou_thresholds, 0, precision, alpha=0.2, edgecolor='r')
            plt.xlabel('iou threshold')
            plt.ylabel('average precision')
            plt.title(f'AP: {round(sum_ap/len(iou_thresholds),4)} class: {class_name}')
            fig = plt.gcf()
            fig.canvas.set_window_title('AP ' + class_name)
            axes = plt.gca()  # gca - get current axes
            axes.set_ylim([0.0, 1.05])
            axes.set_xlim([0.5, 1.0])
            fig.savefig(output_files_path + "/classes/" + class_name + "ap.png")
            plt.cla()  # clear axes for next plot

            # save recall/iou curve
            plt.plot(iou_thresholds, recall, '-o')
            plt.fill_between(iou_thresholds, 0, recall, alpha=0.2, edgecolor='r')
            plt.xlabel('iou threshold')
            plt.ylabel('recall')
            plt.title(f'AR: {round(sum_ar / len(iou_thresholds), 4)} class: {class_name}')
            fig = plt.gcf()
            fig.canvas.set_window_title('AR ' + class_name)
            axes = plt.gca()  # gca - get current axes
            axes.set_ylim([0.0, 1.05])
            axes.set_xlim([0.5, 1.0])
            fig.savefig(output_files_path + "/classes/" + class_name + "ar.png")
            plt.cla()  # clear axes for next plot
            print(f'AP: {round(sum_ap/len(iou_thresholds),4)} class: {class_name}')
            print(f'AR: {round(sum_ar/len(iou_thresholds), 4)} class: {class_name}')
    print(f'mAP: {round(sum_for_map/(len(gt_classes)*len(iou_thresholds)),4)}')
    print(f'mAR: {round(sum_for_mar/(len(gt_classes)*len(iou_thresholds)), 4)}')
