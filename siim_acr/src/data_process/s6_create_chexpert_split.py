import sys
sys.path.insert(0, '..')
import pandas as pd

from config.config import *
from utils.common_util import *

def load_meta():
    chexpert_meta_df = pd.concat((
        pd.read_csv(opj(DATA_DIR, 'external', 'CheXpert-v1.0', 'train.csv')),
        pd.read_csv(opj(DATA_DIR, 'external', 'CheXpert-v1.0', 'valid.csv')),
    ))
    chexpert_meta_df[TARGET] = ' '
    chexpert_meta_df[ID] = chexpert_meta_df['Path'].apply(lambda x: '_'.join(x[:x.rfind('.jpg')].split('/')[2:]))
    chexpert_meta_df = chexpert_meta_df[chexpert_meta_df['Frontal/Lateral'] == 'Frontal']
    chexpert_meta_df = chexpert_meta_df[chexpert_meta_df[PNEUMOTHORAX] != -1]

    print(chexpert_meta_df.shape)
    print(chexpert_meta_df[PNEUMOTHORAX].unique())
    print()

    return chexpert_meta_df

def create_split_file(meta_df, name="train", num=None):
    split_dir = opj(DATA_DIR, 'split')
    os.makedirs(split_dir, exist_ok=True)

    if num is None:
        split_df = meta_df
    elif name == "valid":
        split_df = meta_df.iloc[-num:].copy()
    else:
        split_df = meta_df.iloc[:num]

    num = len(split_df)
    print("create split file: %s_%d" % (name, num))
    fname = opj(split_dir, "%s_%d.csv" % (name, num))
    split_df.to_csv(fname, index=False)

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))

    chexpert_meta_df = load_meta()
    create_split_file(chexpert_meta_df, name="chexpert", num=None)
