import numpy as np
from config.config import *


def run_length_encode(component):
    if component.sum() == 0:
        return '-1'
    component = np.hstack([np.array([0]), component.T.flatten(), np.array([0])])
    start  = np.where(component[1: ] > component[:-1])[0]
    end    = np.where(component[:-1] > component[1: ])[0]
    length = end-start

    rle = []
    for i in range(len(length)):
        if i==0:
            rle.extend([start[0],length[0]])
        else:
            rle.extend([start[i]-end[i-1],length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height,width), np.float32)
    if rle == '-1':
        return component
    component = component.reshape(-1)
    rle  = np.array([int(s) for s in rle.split(' ')])
    rle  = rle.reshape(-1, 2)

    start = 0
    for index,length in rle:
        start = start+index
        end   = start+length
        component[start : end] = fill_value
        start = end

    component = component.reshape(width, height).T
    return component
