import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

from config.config import *
from utils.common_util import *

if __name__ == '__main__':
    split_df = pd.read_csv(opj(DATA_DIR, 'split', 'chexpert_188521.csv'))
    bbox_df = pd.read_csv(opj(DATA_DIR, 'external/CheXpert-v1.0/images', 'boxes_cropv3.csv.gz'))
    bbox_df = bbox_df[[ID, CROP_ID]]

    crop_split_df = pd.merge(bbox_df, split_df, on=ID, how='left')
    crop_split_df.to_csv(opj(DATA_DIR, 'cropv3_split', 'chexpert_188521.csv'),
                         index=False, encoding='utf-8')
