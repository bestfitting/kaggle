import sys
sys.path.insert(0, '..')
from tqdm import tqdm
import cv2
from skimage import measure

from config.config import *
from utils.common_util import *
from layers.loss_funcs.kaggle_metric import *
from utils.mask_functions import run_length_encode

def get_rect(mask, paddings):
    top, bottom, left, right = paddings
    label_image = measure.label(mask)
    max_area = 0
    min_rows, min_cols, max_rows, max_cols = [], [], [], []
    for region in measure.regionprops(label_image):
        if region.area < 2000:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        min_rows.append(min_row)
        min_cols.append(min_col)
        max_rows.append(max_row)
        max_cols.append(max_col)
        if region.area > max_area:
            max_area = region.area

    min_row, min_col, max_row, max_col = min(min_rows), min(min_cols), max(max_rows), max(max_cols),
    min_row, min_col=max(min_row - top, 0), max(min_col - left, 0)
    max_row, max_col=min(max_row + bottom, mask.shape[0]), min(max_col + right, mask.shape[1])
    return min_row, min_col, max_row, max_col, max_area


def generate_crop_images(params):
    idx, crop_size, fname, result_probs, threshold, image_dir, mask_dir, out_mask_dir, out_image_dir, dataset = params
    paddings = (40, 40, 40, 40)

    assert dataset in ['train', 'test'], dataset

    image = cv2.imread(opj(image_dir, '%s.png' % fname))
    image_size = image.shape[0]

    base_mask = cv2.imread(opj(mask_dir, '%s.png' % fname), flags=cv2.IMREAD_GRAYSCALE)
    if base_mask is None:
        base_mask = np.zeros((image_size, image_size), dtype=image.dtype)

    pred_mask = ((result_probs[idx] > (threshold * 255)) * 255).astype('uint8')
    if pred_mask.shape[0] != image_size or pred_mask.shape[1] != image_size:
        pred_mask = cv2.resize(pred_mask, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    min_row, min_col, max_row, max_col, max_area = get_rect(pred_mask, paddings)
    if ((max_col - min_col) * (max_row - min_row)) < 300 * 300:
        print(fname)
        min_row, min_col, max_row, max_col = 0, 0, image.shape[0], image.shape[0]

    crop_fname = '%s' % (fname)

    crop_image = image[min_row:max_row, min_col:max_col]
    cv2.imwrite(opj(out_image_dir, '%s.png' % crop_fname), crop_image)

    crop_mask = base_mask[min_row:max_row, min_col:max_col]
    mask_area = (crop_mask > 0).sum()
    crop_rle = run_length_encode(crop_mask)
    if dataset == 'train':
        cv2.imwrite(opj(out_mask_dir, '%s.png' % crop_fname), crop_mask)

    box_info_list = []
    w = max_col - min_col
    h = max_row - min_row
    box_info_list.append((fname, crop_fname, min_col, min_row, w, h, mask_area, crop_rle))
    box_df = pd.DataFrame(data=box_info_list, columns=[ID, CROP_ID, 'x', 'y', 'w', 'h', MASK_AREA, TARGET])
    return box_df

def main():
    threshold = 0.5
    crop_version = 'cropv3'
    crop_size = 512
    np.random.seed(100)

    for dataset in ['train', 'test']:
        print(dataset)
        mask_dir = opj(DATA_DIR, dataset, 'images/masks_1024')
        image_dir = opj(DATA_DIR, dataset, 'images/images_1024')

        out_mask_dir = opj(DATA_DIR, dataset, 'images/masks_%s' % (crop_version))
        out_image_dir = opj(DATA_DIR, dataset, 'images/images_%s' % (crop_version))
        os.makedirs(out_mask_dir, exist_ok=True)
        os.makedirs(out_image_dir, exist_ok=True)

        result_dir = opj(RESULT_DIR, 'submissions',
                         'siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds',
                         'fold0/epoch_final/default')

        result_df = pd.read_csv(opj(result_dir, 'results_%s.csv.gz' % dataset))
        result_probs = np.load(opj(result_dir, 'prob_%s.npz' % dataset))['data']

        print(result_df.shape, result_probs.shape)

        result_list = []
        for idx, fname in tqdm(enumerate(result_df[ID].values), total=len(result_df)):
            params = (idx, crop_size, fname, result_probs, threshold, image_dir, mask_dir, out_mask_dir, out_image_dir, dataset)
            result_list.append(generate_crop_images(params))

        box_df = pd.concat(result_list, ignore_index=True)
        box_file = opj(DATA_DIR, dataset, 'images/boxes_%s.csv.gz' % (crop_version))
        box_df.to_csv(box_file, index=False, compression='gzip')

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))
    main()
