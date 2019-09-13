import sys
sys.path.insert(0, '..')
import pandas as pd
from tqdm import tqdm
from utils.common_util import *
from utils.mask_functions import *

def get_mask(df, id):
    mask_pixels = df.loc[df[ID] == id][TARGET].values
    mask = np.zeros((1024, 1024))
    for i, mask_pixel in enumerate(mask_pixels):
        if mask_pixel != '-1':
            mask = mask + run_length_decode(mask_pixel, 1024, 1024).T
    if mask.max() != 0:
        if mask.max() != 255:
            print('%s    %s' % (id, mask.max()))
        mask = mask.clip(0, 255)
        mask = mask.astype(np.uint8)
        return mask
    else:
        return None

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))
    data_type = 'train'

    meta_dir = opj(DATA_DIR, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    df = pd.read_csv(opj(DATA_DIR, 'raw/train-rle.csv'), index_col=0).reset_index()
    rles = []
    for id in tqdm(df[ID].unique()):
        sum_mask = get_mask(df, id)
        if sum_mask is not None:
            rel = run_length_encode(sum_mask)
        else:
            rel = '-1'
        rles.append([id, rel])
    df = pd.DataFrame(rles, columns=[ID, TARGET])
    df.to_csv(opj(meta_dir, '%s-rle.csv' % data_type), index=False)
