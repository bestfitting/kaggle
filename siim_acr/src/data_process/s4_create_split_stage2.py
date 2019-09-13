import sys
sys.path.insert(0, '..')
import pandas as pd
from config.config import *
opj = os.path.join
ope = os.path.exists

def create_split_file(meta_df, data_set="train", name="train", num=None):
    split_dir = opj(DATA_DIR, 'split')
    os.makedirs(split_dir, exist_ok=True)

    if data_set == 'train':
        ds = meta_df.copy()
    else:
        ds = pd.read_csv(opj(DATA_DIR, 'raw', 'sample_submission.csv'))
        ds = ds.drop_duplicates(ID).reset_index(drop=True)

    if num is None:
        split_df = ds
    elif name == "valid":
        split_df = ds.iloc[-num:].copy()
    else:
        split_df = ds.iloc[:num]

    if num is None:
        print("create split file: %s" % (name))
        fname = opj(split_dir, "%s.csv" % (name))
    else:
        num = len(split_df)
        print("create split file: %s_%d" % (name, num))
        fname = opj(split_dir, "%s_%d.csv" % (name, num))
    split_df.to_csv(fname, index=False)

def create_random_split(train_meta, n_splits=4):
    train_meta = train_meta.copy()
    split_dir = opj(DATA_DIR, 'split', 'random_folds%d' % n_splits)
    os.makedirs(split_dir, exist_ok=True)

    split_dir_old = opj(DATA_DIR, 'split_stage1', 'random_folds%d' % n_splits)

    train_meta_new = train_meta[~train_meta[ID].isin(train_meta_old[ID].values)].reset_index(drop=True)
    for idx in range(n_splits):
        train_split_df_old = pd.read_csv(opj(split_dir_old, 'random_train_cv%d.csv' % idx))
        valid_split_df = pd.read_csv(opj(split_dir_old, 'random_valid_cv%d.csv' % idx))

        train_split_df = pd.concat([train_split_df_old, train_meta_new])

        fname = opj(split_dir, 'random_train_cv%d.csv' % idx)
        print("train: create split file: %s; samples: %s" % (fname, train_split_df.shape[0]))
        train_split_df.to_csv(fname, index=False)

        fname = opj(split_dir, 'random_valid_cv%d.csv' % idx)
        print("valid: create split file: %s; samples: %s" % (fname, valid_split_df.shape[0]))
        valid_split_df.to_csv(fname, index=False)


if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))

    train_meta = pd.read_csv(opj(DATA_DIR, 'raw', 'train-rle.csv'))
    train_meta = train_meta.drop_duplicates(ID).reset_index(drop=True)

    train_meta_old = pd.read_csv(opj(DATA_DIR, 'split_stage1', 'train.csv'))
    train_meta_old = train_meta_old.drop_duplicates(ID).reset_index(drop=True)

    create_split_file(train_meta, data_set="train", name="train", num=160)
    create_split_file(train_meta, data_set="train", name="valid", num=160)
    create_split_file(train_meta, data_set="test", name="test", num=160)

    create_split_file(train_meta, data_set="train", name="train", num=None)
    create_split_file(train_meta, data_set="test", name="test", num=None)

    create_random_split(train_meta, n_splits=4)
    create_random_split(train_meta, n_splits=10)
