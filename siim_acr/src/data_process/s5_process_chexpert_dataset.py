import sys
sys.path.insert(0, '..')
from tqdm import tqdm
import cv2
import mlcrate as mlc

from config.config import *
from utils.common_util import *

def process_image(params):
    src, dst = params
    if ope(dst):
        return
    img = cv2.imread(src)
    if img is None:
        return
    scale = float(IMG_SIZE) / min(img.shape[0], img.shape[1])
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    cv2.imwrite(dst, img)

if __name__ == '__main__':
    base_dir = opj(DATA_DIR, 'external/CheXpert-v1.0/')
    out_dir = opj(DATA_DIR, 'external/CheXpert-v1.0/images/images_1024')
    os.makedirs(out_dir, exist_ok=True)

    fpath_list = []
    for dataset in ['train', 'valid']:
        dataset_dir = opj(base_dir, dataset)
        patient_list = os.listdir(dataset_dir)
        for patient in tqdm(patient_list, total=len(patient_list), desc='patient'):
            patient_dir = opj(dataset_dir, patient)
            study_list = os.listdir(patient_dir)
            for study in study_list:
                study_dir = opj(patient_dir, study)
                fname_list = os.listdir(study_dir)
                for fname in fname_list:
                    dst_fname = '%s_%s_%s' % (patient, study, fname)
                    fpath_list.append((opj(study_dir, fname), opj(out_dir, dst_fname)))

    pool = mlc.SuperPool(8)
    pool.map(process_image, fpath_list, description='resize')
