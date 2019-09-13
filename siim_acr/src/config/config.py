import os
ope = os.path.exists
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hostname = socket.gethostname()
print('run on %s' % hostname)

RESULT_DIR = "/data4/data/siim_open/result"
DATA_DIR = "/data5/data/siim_open"
PRETRAINED_DIR = "/data5/data/pretrained"
PI  = np.pi
INF = np.inf
EPS = 1e-12
ID = 'ImageId'
TARGET = 'EncodedPixels'
IMG_SIZE = 1024
CROP_ID = 'CropImageId'
MASK_AREA = 'MaskArea'
DATASET = 'dataset'

PNEUMOTHORAX = 'Pneumothorax'
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', PNEUMOTHORAX
]
