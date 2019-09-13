import sys
sys.path.insert(0, '..')
from tqdm import tqdm
import gc
import pandas as pd

from config.config import *
from utils.common_util import *
from utils.augment_util import *
from layers.loss_funcs.kaggle_metric import *

def unaugment_probs(probs, augment_name=None):
    size = probs.shape[0]
    unaug_probs = []
    for prob in tqdm(probs, total=size, desc=augment_name):
        unaug_probs.append(prob)
    probs = np.array(unaug_probs, dtype='uint8')
    return probs

def load_info(probs_dir, dataset):
    prob_fname = opj(probs_dir, 'prob_%s.npz' % dataset)
    probs = np.load(prob_fname)['data']

    img_ids_fname = opj(probs_dir, 'img_ids_%s.npy' % dataset)
    img_ids = np.load(img_ids_fname, allow_pickle=True)

    assert len(img_ids) == probs.shape[0], 'num img_ids: %d, num probs: %d' % (len(img_ids), probs.shape[0])

    result_fname = opj(probs_dir, 'results_%s.csv.gz' % dataset)
    results = pd.read_csv(result_fname)

    return results, probs, img_ids

def save_result(out_dir, probs, img_ids, rles, dataset, kaggle_score=0.):
    result_csv_file = opj(out_dir, 'results_%s.csv.gz' % (dataset))
    result_prob_fname = opj(out_dir, "prob_%s.npz" % (dataset))
    result_img_ids_fname = opj(out_dir, "img_ids_%s.npy" % (dataset))

    print('save csv file...')
    result_df = pd.DataFrame({ID: img_ids, TARGET: rles})
    result_df.to_csv(result_csv_file, index=False, compression='gzip')

    print('save probs file...')
    np.savez_compressed(result_prob_fname, data=probs)

    np.save(result_img_ids_fname, img_ids)

    score = get_kaggle_score(result_df, result_type=dataset)
    print('kaggle score:%.5f' % score)

    if dataset == 'val':
        kaggle_score = score

    result_csv_file = opj(out_dir, 'results_%s_%.4f.csv.gz' % (dataset, kaggle_score))
    result_df.to_csv(result_csv_file, index=False, compression='gzip')

    return kaggle_score

import argparse
parser = argparse.ArgumentParser(description='PyTorch Siim segmentation')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--epoch_name', default='final', type=str, help='epoch name (default: final)')
parser.add_argument('--model_name', default=None, type=str, help='complete model name')
parser.add_argument('--augment_names', default=None, type=str, help='')
parser.add_argument('--do_valid', default=1, type=int)
parser.add_argument('--do_test', default=1, type=int)
parser.add_argument('--crop_version', default='cropv1', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    fold = args.fold
    threshold = 0.5

    epoch_name = args.epoch_name
    model_name = args.model_name
    epoch_name = 'epoch_%s' % epoch_name
    augment_names = args.augment_names
    do_valid = args.do_valid == 1
    do_test = args.do_test == 1
    crop_version = args.crop_version

    if augment_names is None:
        augment_names = [
            'default',
        ]
    else:
        augment_names = augment_names.split(',')

    dataset_list = []
    if do_valid:
        dataset_list.append('val')
    if do_test:
        dataset_list.append('test')

    kaggle_score = 0.
    for dataset in dataset_list:
        crop_img_ids = None
        crop_probs = None

        base_dir = opj(RESULT_DIR, 'submissions', model_name, 'fold%s' % fold, epoch_name)
        print(base_dir)
        for augment_name in augment_names:
            probs_dir = opj(base_dir, augment_name)
            results, probs, img_ids = load_info(probs_dir, dataset)

            probs = probs / 255.
            print('%20s probs mean: %.5f' % (augment_name, probs.mean()))

            if dataset == 'val':
                df_truth = pd.read_csv(opj(DATA_DIR, 'meta', 'train_%s_meta.csv' % crop_version))
                df_truth = df_truth[df_truth[CROP_ID].isin(img_ids)]
                score = get_kaggle_score(results, df_truth=df_truth, result_type=dataset)
            else:
                score = 0.
            print('%s kaggle score:%.5f' % (augment_name, score))

            if crop_probs is None:
                crop_img_ids = img_ids
                crop_probs = probs
            else:
                assert np.all(crop_img_ids == img_ids)
                crop_probs = crop_probs + probs
        crop_probs = crop_probs / len(augment_names)
        print('avg probs mean: %.5f' % crop_probs.mean())

        print('generate original probabilities')
        box_fname = opj(DATA_DIR, 'train' if dataset == 'val' else dataset,'images', 'boxes_%s_512.csv.gz' % crop_version)
        if not ope(box_fname):
            box_fname = opj(DATA_DIR, 'train' if dataset == 'val' else dataset, 'images', 'boxes_%s.csv.gz' % crop_version)
        box_df = pd.read_csv(box_fname)
        box_df = pd.merge(pd.DataFrame({CROP_ID: crop_img_ids}), box_df, on=CROP_ID, how='left')
        assert np.all(box_df[ID].notnull())

        base_img_ids = np.sort(box_df[ID].unique())
        base_probs = np.zeros((len(base_img_ids), 1024, 1024), dtype='float32')
        base_nums = np.zeros((len(base_img_ids), 1024, 1024), dtype='int32')
        for img_idx, img_id in tqdm(enumerate(base_img_ids), total=len(base_img_ids)):
            sub_crop_img_ids = box_df[box_df[ID] == img_id][CROP_ID].values
            for sub_crop_img_id in sub_crop_img_ids:
                crop_idx = np.where(crop_img_ids == sub_crop_img_id)[0][0]
                x, y, w, h = box_df.iloc[crop_idx][['x', 'y', 'w', 'h']].values
                _crop_probs = crop_probs[crop_idx]
                _crop_probs = cv2.resize(_crop_probs, (w, h), interpolation=cv2.INTER_LINEAR)
                base_probs[img_idx, y:y + h, x:x + w] += _crop_probs
                base_nums[img_idx, y:y + h, x:x + w] += 1

        del crop_probs
        gc.collect()

        base_nums = base_nums.clip(min=1)
        base_probs = base_probs / base_nums

        del base_nums
        gc.collect()

        base_probs = (base_probs.clip(0, 1) * 255).astype('uint8')

        rles = []
        for img_id, prob in tqdm(zip(base_img_ids, base_probs), total=len(base_img_ids), desc='rle encode'):
            prob = cv2.resize(prob, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            mask = prob > (threshold * 255)
            rle = run_length_encode(mask)
            rles.append(rle)

        sub_dir = opj(base_dir, 'average')
        os.makedirs(sub_dir, exist_ok=True)
        kaggle_score = save_result(sub_dir, base_probs, base_img_ids, rles, dataset, kaggle_score=kaggle_score)

        del base_probs
        gc.collect()

    print('\nsuccess!')
