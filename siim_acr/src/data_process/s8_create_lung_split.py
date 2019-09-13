import sys
sys.path.insert(0, '..')
import pandas as pd
from sklearn.model_selection import KFold

from config.config import *
from utils.common_util import *

def create_split_file(meta_df, data_set="lung", name="train", num=None):
    split_dir = opj(DATA_DIR, '%s_split' % data_set)
    os.makedirs(split_dir, exist_ok=True)

    ds = meta_df.copy()

    if num is None:
        split_df = ds
    elif name == "valid":
        split_df = ds.iloc[-num:].copy()
    else:
        split_df = ds.iloc[:num]

    num = len(split_df)
    print("create split file: %s_%d" % (name, num))
    fname = opj(split_dir, "%s_%d.csv" % (name, num))
    split_df.to_csv(fname, index=False)

def create_random_split(train_meta, n_splits=4):
    train_meta = train_meta.copy()
    split_dir = opj(DATA_DIR, '%s_split' % dataset, 'random_folds%d' % n_splits)
    os.makedirs(split_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=100)
    for idx, (train_indices, valid_indices) in enumerate(kf.split(train_meta)):
        train_split_df = train_meta.loc[train_indices]
        valid_split_df = train_meta.loc[valid_indices]

        fname = opj(split_dir, 'random_train_cv%d.csv' % idx)
        print("train: create split file: %s; samples: %s"
              % (fname, train_split_df.shape[0]))
        train_split_df.to_csv(fname, index=False)

        fname = opj(split_dir, 'random_valid_cv%d.csv' % idx)
        print("valid: create split file: %s; samples: %s"
              % (fname, valid_split_df.shape[0]))
        valid_split_df.to_csv(fname, index=False)

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))

    dataset = 'lung'
    metas = ['train_montgomery_meta.csv']

    train_meta = []
    for meta in metas:
        train_meta_temp = pd.read_csv(opj(DATA_DIR, 'meta', meta))
        train_meta_temp = train_meta_temp.drop_duplicates(ID).reset_index(drop=True)
        train_meta_temp = train_meta_temp[[ID]]
        train_meta.append(train_meta_temp)
    train_meta = pd.concat(train_meta)
    train_meta = train_meta.reset_index(drop=True)
    train_meta[TARGET] = ' '
    train_meta['CropImageId'] = train_meta[ID]

    create_split_file(train_meta, data_set=dataset, name="train", num=None)

    create_random_split(train_meta, n_splits=4)
