import sys
sys.path.insert(0, '..')
import pandas as pd
from config.config import *
from sklearn.model_selection import KFold
opj = os.path.join
ope = os.path.exists

def create_random_split(train_meta, n_splits=4):
    train_meta = train_meta.copy()
    split_dir = opj(DATA_DIR, 'split_stage1', 'random_folds%d' % n_splits)
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

    train_meta = pd.read_csv(opj(DATA_DIR, 'raw', 'train-rle_stage1.csv'))
    train_meta = train_meta.drop_duplicates(ID).reset_index(drop=True)

    create_random_split(train_meta, n_splits=4)
    create_random_split(train_meta, n_splits=10)
