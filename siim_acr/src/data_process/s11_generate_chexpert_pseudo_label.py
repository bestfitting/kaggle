import sys
sys.path.insert(0, '..')
import pandas as pd
from tqdm import tqdm
import cv2

from config.config import *
from utils.common_util import *
from utils.mask_functions import run_length_encode

if __name__ == '__main__':
    model_name = 'siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds'
    result_dir = opj(RESULT_DIR, 'submissions', model_name, 'fold0/epoch_final/default')

    img_dir = opj(DATA_DIR, 'external/CheXpert-v1.0/images', 'images_1024')
    mask_dir = opj(DATA_DIR, 'external/CheXpert-v1.0/images', 'masks_1024')
    os.makedirs(mask_dir, exist_ok=True)

    img_ids = np.load(opj(result_dir, 'img_ids_chexpert.npy'), allow_pickle=True)
    probs = np.load(opj(result_dir, 'prob_chexpert.npz'))['data']

    meta_df = pd.read_csv(opj(DATA_DIR, 'split', 'chexpert_188521.csv'))
    meta_df[DATASET] = 'external/CheXpert-v1.0'
    sub_meta_df = meta_df[meta_df[PNEUMOTHORAX] == 1]
    assert np.array_equal(img_ids, sub_meta_df[ID].values)

    indices = np.where(sub_meta_df[PNEUMOTHORAX] == 1)[0]
    print(len(indices))

    threshold = 0.5
    for idx in tqdm(indices, total=len(indices)):
        prob = probs[idx]
        img_id = img_ids[idx]
        mask = prob > (threshold * 255)
        if mask.sum() > 100:
            img = cv2.imread(opj(img_dir, '%s.jpg' % img_id))
            mask = mask.astype('uint8')

            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask * 255).astype('uint8')
            cv2.imwrite(opj(mask_dir, '%s.png' % img_id), mask)

    img_ids = os.listdir(mask_dir)

    neg_pseudo_df = meta_df[meta_df[PNEUMOTHORAX] == 0]
    neg_pseudo_df[TARGET] = '-1'
    pos_pseudo_df = meta_df[meta_df[ID].isin([img_id[:img_id.rfind('.png')] for img_id in img_ids])]
    rle_list = []
    for img_id in tqdm(pos_pseudo_df[ID].values, total=len(pos_pseudo_df)):
        mask = cv2.imread(opj(DATA_DIR, 'external', 'CheXpert-v1.0', 'images', 'masks_1024', '%s.png' % img_id),
                          flags=cv2.IMREAD_GRAYSCALE)
        if mask.shape != (IMG_SIZE, IMG_SIZE):
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        rle = run_length_encode((mask > (255 * 0.5)).astype('uint8'))
        rle_list.append(rle)
    pos_pseudo_df[TARGET] = rle_list

    pseudo_df = pd.concat([neg_pseudo_df, pos_pseudo_df])
    print(len(pseudo_df), len(neg_pseudo_df), len(pos_pseudo_df))

    pseudo_df = pseudo_df[[ID, TARGET, DATASET]]
    print(pseudo_df.shape)
    print(pseudo_df.head())

    pseudo_dir = opj(DATA_DIR, 'pseudo')
    os.makedirs(pseudo_dir, exist_ok=True)
    pseudo_df.to_csv(opj(pseudo_dir, 'chexpert_pseudo.csv'), index=False, encoding='utf-8')
