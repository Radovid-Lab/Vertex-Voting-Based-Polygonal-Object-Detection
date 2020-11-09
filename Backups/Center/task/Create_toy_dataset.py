import numpy as np
import os
import cv2
import sys
import argparse
import random
from typing import List
import math
from utils.misc_math import subtract, length, angle_radian


def getVertex(center:List[int], pictureSize: List[int], angleList: List[int],lengthLimitRatio: List[int] = [1 / 5, 1 / 1.5]):
    '''

    :param center:initial point of affine lines
    :param pictureSize:picture size
    :param angleList:vertexs will be generated on affine lines which are created based angleList
    :param lengthLimitRatio:distance between center and vertex should between lengthLimitRatio*affine line length
    :return:vertex of a polygon
    '''

    rand = random.random() * (lengthLimitRatio[1] - lengthLimitRatio[0]) + lengthLimitRatio[0]
    vertexlist = []
    for i in angleList:
        if i == 90:
            vertexlist.append([center[0], rand * (pictureSize[1] - center[1]) + center[1]])
            continue
        if i == 270:
            vertexlist.append([center[0], center[1] - rand * (center[1])])
            continue
        k = math.tan(i / 360 * math.pi * 2)
        if 90 > i >= 0:  # right and upper
            assert k >= 0
            if k * (pictureSize[0] - center[0]) > (pictureSize[1] - center[1]):  # on upperEdge
                randy = (pictureSize[1] - center[1]) * rand
                randx = randy / k
            else:
                randx = (pictureSize[0] - center[0]) * rand
                randy = randx * k
            assert randx >= 0 and randy >= 0
        elif 180 >= i > 90:
            assert k < 0
            if k * (0 - center[0]) > (pictureSize[1] - center[1]):  # on upperEdge
                randy = (pictureSize[1] - center[1]) * rand
                randx = randy / k
            else:
                randx = (0 - center[0]) * rand
                randy = randx * k
                assert randx < 0 and randy > 0
        elif 270 > i > 180:
            assert k > 0
            if k * (0 - center[0]) < (-center[1]):
                randy = (-center[1]) * rand
                randx = randy / k
            else:
                randx = -center[0] * rand
                randy = randx * k
            assert randy < 0 and randx < 0
        elif 360 >= i > 270:  # (270-360]
            assert k <= 0
            if k * (pictureSize[0] - center[0]) < -center[1]:
                randy = -center[1] * rand
                randx = randy / k
            else:
                randx = (pictureSize[0] - center[0]) * rand
                randy = randx * k
            assert randy <= 0 and randx >= 0
        vertexlist.append([center[0] + randx, center[1] + randy])
    vertexlist = [[math.floor(i[0]), math.floor(i[1])] for i in vertexlist]

    return vertexlist


def Perimeter(vectorList: List[int]):
    '''
    :param vectorList: vectors of a polygon
    :return:perimeter of the polygon, will be used to constrain the polygon's size
    '''
    assert len(vectorList) >= 3
    perimeter = 0
    for i in range(len(vectorList) - 1):
        temp = subtract(vectorList[i], vectorList[i + 1])
        perimeter += length(temp)
    temp = subtract(vectorList[-1], vectorList[0])
    perimeter += length(temp)
    return perimeter


def vertexOfOnePolygon(pictureSize: List[int], varianceOfEdge: List[int] = [3, 5]):
    '''
    Generate a polygon
    :param pictureSize:picture size
    :param varianceOfEdge: number of edges will be randomly selected from this
    :return: vertex list of a polygon
    '''
    while True:
        centerPoint = [pictureSize[0] / 2, pictureSize[1] / 2]
        numberOfEdges = random.randint(varianceOfEdge[0], varianceOfEdge[1])
        angleList = [random.randint(0, 360)]
        while len(angleList) != numberOfEdges:
            angle = random.randint(0, 360)
            toggle = False
            for i in angleList:
                if abs(i - angle) < 30:
                    toggle = True
            if toggle:
                continue
            angleList.append(angle)
        angleList.sort()
        vertexlist = getVertex(centerPoint, pictureSize, angleList)
        angleConstraint=True # true if all angles of a polygon are between 30-150 degree
        for i in range(len(vertexlist)):
            v1=subtract(vertexlist[(i-1)%len(vertexlist)],vertexlist[i])
            v2=subtract(vertexlist[(i+1)%len(vertexlist)],vertexlist[i])
            if length(v1)==0 or length(v2)==0:
                angleConstraint=False
            else:
                if 160>angle_radian(v1,v2)/math.pi*180>20:
                    pass
                else:
                    angleConstraint=False
        if Perimeter(vertexlist) > max(pictureSize) and angleConstraint:
            break
    return vertexlist


def writePolygon(filename,polygonList):
    with open(filename, 'w') as the_file:
        for k in polygonList:
            the_file.write(str(k) + '\n')

def createData(imageSize, destination, count , fill:bool=True):
    '''
    Create Dataset
    :param imageSize: picture size
    :param destination: destination to store files
    :param count: number of pictures generated
    :return: None
    '''
    splitter1 = random.uniform(-1,1)/5 + 0.5
    splitter1 = int(splitter1 * imageSize[0])
    splitter2 = random.uniform(-1,1)/5 + 0.5
    splitter2 = int(splitter2 * imageSize[1])
    splitter3 = random.uniform(-1,1)/5 + 0.5
    splitter3 = int(splitter3 * imageSize[0])
    result = []

    numOfPolygons = random.randint(0, 3)
    dropList = []
    for i in range(numOfPolygons):
        j = random.randint(0, 3)
        while j in dropList:
            j = random.randint(0, 3)
        dropList.append(j)

    if 0 not in dropList:
        polygons = vertexOfOnePolygon([splitter1, splitter2])
        result.append(polygons)

    if 1 not in dropList:
        polygons =vertexOfOnePolygon([imageSize[0] - splitter1, splitter2])
        for i in range(len(polygons)):
            polygons[i][0] = polygons[i][0] + splitter1
        result.append(polygons)

    if 2 not in dropList:
        polygons = vertexOfOnePolygon([splitter3, imageSize[1] - splitter2])
        for i in range(len(polygons)):
            polygons[i][1] = polygons[i][1] + splitter2
        result.append(polygons)

    if 3 not in dropList:
        polygons = vertexOfOnePolygon([imageSize[0] - splitter3, imageSize[1] - splitter2])
        for i in range(len(polygons)):
            polygons[i][0] = polygons[i][0] + splitter3
            polygons[i][1] = polygons[i][1] + splitter2
        result.append(polygons)

    filename = os.path.join(destination, 'img' + str(count) + '.txt')
    writePolygon(filename, result)

    img = np.zeros((imageSize[0], imageSize[1], 3), np.uint8)
    for j in result:
        for i in range(len(j)):
            temp = j[i][0]
            j[i][0] = j[i][1]
            j[i][1] = temp
        # img = cv2.polylines(img, [(np.array(j, np.int32)).reshape((-1, 1, 2))], True, color, thickness=1)
        if fill:
            thickness=-1
            while True:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                if sum(color) > 150:
                    break
        else:
            thickness=1
            color=[255,255,255]
        img = cv2.drawContours(img, [(np.array(j, np.int32)).reshape((-1, 1, 2))], -1, color, thickness)

    filename = os.path.join(destination, 'img' + str(count) + '.png')
    cv2.imwrite(filename, img)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--number',help='the number of data pieces you want to generate',type=int,default=1000)
    parser.add_argument("-f","--fill", help="indicate to fill the polygons or not",action="store_true")
    parser.add_argument('-s','--size',help="indicate the size of images, eg.[256,256]",type=list,nargs='+',default='[256,256]')
    parser.add_argument('-d', '--destination', help="destination to store the data", type=str,default='./Data')
    args=parser.parse_args()
    pictureSize=''.join(args.size)
    pictureSize=pictureSize.strip('[]')
    pictureSize=pictureSize.split(',')
    pictureSize=[int(i) for i in pictureSize]
    SizeOfData = args.number
    destination = args.destination

    print(type(SizeOfData))
    print(f'{SizeOfData} images has been generated')
    print(f'Picture size is {pictureSize[0]} by {pictureSize[0]}')

    if os.path.exists(destination):
        pass
    else:
        os.mkdir(destination)
    print('pictures are stored in folder %s' % destination)

    for i in range(SizeOfData):
        createData(pictureSize, destination, i,args.fill)

if __name__ == "__main__":
    main(sys.argv)