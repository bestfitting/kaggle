import sys
sys.path.insert(0, '..')
import pandas as pd

from config.config import *
from utils.common_util import *

def get_meta(update=False):
    meta_dir = opj(DATA_DIR, 'meta')
    os.makedirs(meta_dir, exist_ok = True)

    train_meta_fname = opj(meta_dir, 'train_%s_meta.csv' % crop_version)
    test_meta_fname = opj(meta_dir, 'test_%s_meta.csv' % crop_version)

    if ope(train_meta_fname) and ope(test_meta_fname) and not update:
        train_meta_df = pd.read_csv(train_meta_fname)
        test_meta_df = pd.read_csv(test_meta_fname)
    else:
        train_meta_df = pd.read_csv(opj(DATA_DIR, 'train', 'images', 'boxes_%s.csv.gz' % crop_version))
        test_meta_df = pd.read_csv(opj(DATA_DIR, 'test', 'images', 'boxes_%s.csv.gz' % crop_version))

        train_meta_df.to_csv(train_meta_fname, index=False, encoding='utf-8')
        test_meta_df.to_csv(test_meta_fname, index=False, encoding='utf-8')

    return train_meta_df, test_meta_df

def create_split_file(meta_df, name='train', num=None):
    split_df = meta_df
    if num is not None:
        if name == 'valid':
            split_df = split_df.iloc[-num:]
        else:
            split_df = split_df.iloc[:num]

    split_dir = opj(DATA_DIR, '%s_split' % crop_version)
    os.makedirs(split_dir, exist_ok=True)

    if num is None:
        print("create split file: %s" % (name))
        fname = opj(split_dir, "%s.csv" % (name))
    else:
        num = len(split_df)
        print("create split file: %s_%d" % (name, num))
        fname = opj(split_dir, "%s_%d.csv" % (name, num))
    split_df.to_csv(fname, index=False)

def create_random_split(meta_df, n_splits=4):
    split_dir = opj(DATA_DIR, '%s_split/random_folds%d' % (crop_version, n_splits))
    os.makedirs(split_dir, exist_ok=True)

    for idx in range(n_splits):
        # train
        original_train_split_df = pd.read_csv(opj(DATA_DIR, 'split', 'random_folds%d' % n_splits,
                                                  'random_train_cv%d.csv' % idx))
        train_split_df = pd.merge(original_train_split_df[[ID]], meta_df, on=ID, how='inner')
        fname = opj(split_dir, 'random_train_cv%d.csv' % idx)
        print("train: create split file: %s; samples: %s" % (fname, train_split_df.shape[0]))
        train_split_df.to_csv(fname, index=False, encoding='utf-8')

        # valid
        original_valid_split_df = pd.read_csv(opj(DATA_DIR, 'split', 'random_folds%d' % n_splits,
                                                  'random_valid_cv%d.csv' % idx))
        valid_split_df = pd.merge(original_valid_split_df[[ID]], meta_df, on=ID, how='inner')
        fname = opj(split_dir, 'random_valid_cv%d.csv' % idx)
        print("valid: create split file: %s; samples: %s" % (fname, valid_split_df.shape[0]))
        valid_split_df.to_csv(fname, index=False, encoding='utf-8')

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    crop_version = 'cropv3'

    train_meta_df, test_meta_df = get_meta(update=True)

    create_split_file(train_meta_df, name='train', num=160)
    create_split_file(train_meta_df, name='valid', num=160)
    create_split_file(test_meta_df, name='test', num=160)

    create_split_file(train_meta_df, name="train", num=None)
    create_split_file(test_meta_df, name="test", num=None)

    create_random_split(train_meta_df, n_splits=4)
    create_random_split(train_meta_df, n_splits=10)
