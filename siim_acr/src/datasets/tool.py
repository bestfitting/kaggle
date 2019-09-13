import numpy as np
import torch

## transform (input is numpy array, read in by cv2)
def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image-mean)/std
    if len(image.shape) == 3:
        image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)

    return tensor

def label_to_tensor(label):
    label_ret = label.astype(np.float32)
    label_ret = (label_ret > 0.1).astype(np.float32)
    tensor = torch.from_numpy(label_ret).type(torch.FloatTensor)
    return tensor
