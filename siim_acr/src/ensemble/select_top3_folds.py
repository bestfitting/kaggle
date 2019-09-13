import sys
sys.path.insert(0, '..')
import pandas as pd
import argparse

from config.config import *
from utils.common_util import *
from layers.loss_funcs.kaggle_metric import get_kaggle_score

def generate_score(base_alias, base_model, model_name, num_folds=10):
    base_dir = opj(RESULT_DIR, 'submissions', base_model, 'fold_en')
    valid_result_old_df = pd.read_csv(opj(base_dir, 'results_val.csv.gz'))
    df_truth = pd.read_csv(opj(DATA_DIR, 'meta', 'train-rle.csv'))
    df_truth = df_truth[df_truth[ID].isin(valid_result_old_df[ID].values)]

    score_list = []
    for fold in range(num_folds):
        result_dir = opj(RESULT_DIR, 'submissions', model_name, 'fold%d' % fold, 'epoch_final', 'average')
        try:
            valid_result_df = pd.read_csv(opj(result_dir, 'results_val.csv.gz'))
        except:
            continue
        valid_result_df = valid_result_df[valid_result_df[ID].isin(df_truth[ID].values)]
        new_score = get_kaggle_score(valid_result_df, df_truth=df_truth)

        sub_valid_result_old_df = valid_result_old_df[valid_result_old_df[ID].isin(valid_result_df[ID].values)]
        old_score = get_kaggle_score(sub_valid_result_old_df, df_truth=df_truth)

        print('fold%d, %s score: %.4f, new score: %.4f, diff score: %.4f' % (fold, base_alias, old_score, new_score, new_score - old_score))
        score_list.append((fold, old_score, new_score, new_score - old_score))
    score_df = pd.DataFrame(data=score_list, columns=['fold', '%s_score' % base_alias, 'new_score', 'diff_score'])
    return score_df

parser = argparse.ArgumentParser(description='PyTorch Siim segmentation')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--out_name', type=str, default=None)
parser.add_argument('--topn', type=int, default=3)
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model_name
    out_name = args.out_name
    topn = args.topn

    base_alias = '8740'
    base_model = 'siim_cropv3_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds'

    score_df = generate_score(base_alias, base_model, model_name)
    score_df = score_df.sort_values(by='diff_score', ascending=False)
    score_df = score_df.reset_index(drop=True)
    select_folds = score_df['fold'].values[:topn]
    print(select_folds)

    out_dir = opj(RESULT_DIR, 'select_folds')
    os.makedirs(out_dir, exist_ok=True)
    np.save(opj(out_dir, '%s_top%d_folds.npy' % (out_name, topn)), select_folds)

