import sys
sys.path.insert(0, '..')
import cv2
import gc
from tqdm import tqdm
from timeit import default_timer as timer
from config.config import *
from utils.common_util import *
from layers.loss_funcs.kaggle_metric import *
from config.en_config import *

def load_info(probs_dir, dataset):
    prob_fname = opj(probs_dir, 'prob_%s.npz' % dataset)
    probs = np.load(prob_fname)['data']

    img_ids_fname = opj(probs_dir, 'img_ids_%s.npy' % dataset)
    img_ids = np.load(img_ids_fname, allow_pickle=True)

    assert len(img_ids) == probs.shape[0], 'num img_ids: %d, num probs: %d' % (len(img_ids), probs.shape[0])

    result_fname = opj(probs_dir, 'results_%s.csv.gz' % dataset)
    results = pd.read_csv(result_fname)

    if probs.shape[-1] != IMG_SIZE:
        new_probs = []
        for prob in tqdm(probs, total=len(probs)):
            prob = cv2.resize(prob, (IMG_SIZE, IMG_SIZE))
            new_probs.append(prob.tolist())
        del probs;gc.collect()
        probs = np.array(new_probs)
        del new_probs;gc.collect()

    print(probs.shape)

    return results, probs, img_ids

def save_result(out_dir, probs, img_ids, rles, dataset, kaggle_score=0):
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
    print('kaggle score:%.6f' % score)
    if dataset == 'val':
        kaggle_score = score

    result_score_csv_file = opj(out_dir, 'results_%s_%.4f.csv.gz' % (dataset, kaggle_score))
    result_df.to_csv(result_score_csv_file, index=False, compression='gzip')
    return kaggle_score

def main():
    threshold = 0.5
    en_cfg_name = args.en_cfgs
    en_cfgs = eval(en_cfg_name)
    do_valid = args.do_valid == 1
    do_test = args.do_test == 1

    print(en_cfgs)

    dataset_list = []
    if do_valid:
        dataset_list.append('val')
    if do_test:
        dataset_list.append('test')
    kaggle_score = 0.
    for dataset in dataset_list:
        base_img_ids = None
        base_probs = None
        weight_sum = 0
        weights = []
        for i,en_cfg in enumerate(en_cfgs):
            is_en = en_cfg['is_en']
            model_name = en_cfg['model_name']
            weight = en_cfg['weight']
            weights.append(weight)
            weight_sum += weight
            if is_en:
                fold = en_cfg.get('fold', 0)
                if fold == 'en':
                    result_dir = opj(RESULT_DIR, 'submissions', model_name, 'fold_en')
                else:
                    result_dir = opj(RESULT_DIR, 'submissions', 'ensemble', model_name)
            else:
                fold = en_cfg['fold']
                epoch_name = en_cfg['epoch_name']
                augment = en_cfg['augment']
                result_dir = opj(RESULT_DIR, 'submissions', model_name,
                                 'fold%d' % fold, 'epoch_%s' % epoch_name, augment)

            print('load result directory: %s' % result_dir)
            results, probs, img_ids = load_info(result_dir, dataset)

            score = get_kaggle_score(results, result_type=dataset)
            print('%s kaggle score:%.5f' % (dataset, score))

            probs = probs / 255.
            print('probs mean: %.5f' % probs.mean())
            probs = probs * weight

            if base_probs is None:
                base_img_ids = img_ids
                base_probs = probs
            else:
                assert np.all(np.sort(base_img_ids) == np.sort(img_ids))
                if not np.all(base_img_ids == img_ids):
                    df = pd.merge(pd.DataFrame({ID: base_img_ids}),
                                  pd.DataFrame({ID: img_ids, '_idx_': np.arange(len(img_ids))}), on=ID, how='left')
                    indices = df['_idx_'].values
                    img_ids = img_ids[indices]
                    new_probs = []
                    for idx in tqdm(indices, total=len(indices)):
                        new_probs.append(probs[idx].tolist())
                    del probs;gc.collect()
                    probs = np.array(new_probs)
                    del new_probs;gc.collect()
                assert np.all(base_img_ids == img_ids)

                base_probs = base_probs + probs
        base_probs = base_probs / weight_sum
        print('avg probs mean: %.5f' % (base_probs.mean()))
        base_probs = (base_probs * 255).astype('uint8')

        rles = []
        for img_id, prob in tqdm(zip(base_img_ids, base_probs), total=len(base_img_ids), desc='rle encode'):
            prob = cv2.resize(prob, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            assert prob.shape == (IMG_SIZE, IMG_SIZE)
            mask = prob > (threshold * 255)
            rle = run_length_encode(mask)
            rles.append(rle)

        sub_dir = opj(RESULT_DIR, 'submissions', 'ensemble', en_cfg_name)
        print('save result: %s' % sub_dir)
        os.makedirs(sub_dir, exist_ok=True)
        kaggle_score = save_result(sub_dir, base_probs, base_img_ids, rles, dataset, kaggle_score=kaggle_score)

import argparse
parser = argparse.ArgumentParser(description='PyTorch Siim segmentation')
parser.add_argument('--en_cfgs', default=None, type=str, help='en configs name')
parser.add_argument('--do_valid', default=1, type=int)
parser.add_argument('--do_test', default=1, type=int)
parser.add_argument('--use_max', default=0, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    start_time = timer()
    main()
    end_time = timer()
    print('\nSpent time: %3.1f min' % ((end_time - start_time) / 60))
    print('\nsuccess!')
