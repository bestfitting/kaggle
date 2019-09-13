import pandas as pd

from config.config import *
from utils.common_util import *
from utils.mask_functions import *

def do_kaggle_metric(predict, truth, threshold=0.5):
    num = len(predict)

    prob = predict > threshold
    truth = truth > 0.5

    prob = prob.reshape(num, -1)
    truth = truth.reshape(num, -1)
    intersection = (prob * truth)

    score = 2. * (intersection.sum(1) + EPS) / (prob.sum(1) + truth.sum(1) + EPS)
    score[score >= 1] = 1
    return score


def get_batch_kaggle_score(param):
    df_submit, ids, start, batch_size = param
    end = np.minimum(start + batch_size, len(ids))
    N = end - start
    predict = np.zeros((N,IMG_SIZE,IMG_SIZE),np.bool)
    truth   = np.zeros((N,IMG_SIZE,IMG_SIZE),np.bool)
    for i,n in enumerate(range(start, end)):
        id = ids[n]
        p = df_submit.loc[id][TARGET+'_pred']
        t = df_submit.loc[id][TARGET]
        p = run_length_decode(p,fill_value=1)
        t = run_length_decode(t,fill_value=1)
        predict[i] = p
        truth[i] = t
    score = do_kaggle_metric(predict, truth, threshold=0.5)
    return score


def get_kaggle_score(df_submit, df_truth=None, result_type='val', pool=None, return_mean=True):
    if df_truth is None:
        if result_type == 'val':
            df_truth = pd.read_csv(opj(DATA_DIR, 'meta', 'train-rle.csv'))
        else:
            return 0.

    df_submit = df_submit.merge(df_truth, how='left', on=ID, suffixes=('_pred', ''))
    ids = df_submit[ID].values
    df_submit = df_submit.set_index(ID).fillna('-1')

    batch_size = 100000
    N = len(ids)
    iter_count = int(np.ceil(N / batch_size))
    params = []
    for it in range(iter_count):
        start = it * batch_size
        params.append((df_submit, ids, start, batch_size))

    score = get_batch_kaggle_score(params[0])
    if return_mean:
        score_mean = np.mean(score)
        return score_mean
    else:
        return score, ids

def get_kaggle_score_prob(prob_submit, image_ids):

    df_truth = pd.read_csv(opj(DATA_DIR, 'meta', 'train-rle.csv'))
    ids_submit = pd.DataFrame(image_ids, columns=[ID])
    ids_submit = ids_submit.merge(df_truth, how='left', on=ID)
    ids_submit = ids_submit.set_index(ID).fillna('-1')
    prob_truth = np.zeros((len(ids_submit), IMG_SIZE, IMG_SIZE), np.bool)
    for i, t in enumerate(ids_submit[TARGET].values):
        t = run_length_decode(t, fill_value=1)
        prob_truth[i] = t
    score = np.mean(do_kaggle_metric(prob_submit, prob_truth))
    return score
