import math

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def scalarproduct(a,v):
    return [a*i for i in v]

def length(v):
    return math.sqrt(dotproduct(v, v))

def subtract(v1,v2):
    return [v1[0]-v2[0],v1[1]-v2[1]]

def add_vec(v1,v2):
    return [v1[0]+v2[0],v1[1]+v2[1]]

def angle_radian(v1,v2):
    eps=1/float('inf')
    return math.acos(dotproduct(v1,v2)/(length(v1)*length(v2)+eps))

import numpy as np

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))