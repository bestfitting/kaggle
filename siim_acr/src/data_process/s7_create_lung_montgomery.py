import sys
sys.path.insert(0, '..')
import cv2
import pandas as pd
from tqdm import tqdm
from config.config import *
from utils.common_util import *

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))
    info_keys = ['SOPClassUID', 'PatientName',
                'PatientID', 'PatientAge', 'PatientSex', 'Modality', 'BodyPartExamined', 'ViewPosition',

                'SpecificCharacterSet', 'SOPInstanceUID', 'StudyDate', 'StudyTime', 'AccessionNumber',
                'ConversionType', "ReferringPhysicianName", 'SeriesDescription', "PatientBirthDate",
                'StudyInstanceUID', 'SeriesInstanceUID', 'StudyID', 'SeriesNumber  ', 'InstanceNumber',
                'PatientOrientation', 'SamplesPerPixel', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelSpacing',
                'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'LossyImageCompression', 'LossyImageCompressionMethod']

    data_type = 'train'
    dataset = 'montgomery'
    src_images_dir = opj(DATA_DIR, 'external', 'NLM-MontgomeryCXRSet/MontgomerySet/CXR_png')
    src_masks_dir = opj(DATA_DIR, 'external', 'NLM-MontgomeryCXRSet/MontgomerySet/ManualMask')

    image_dir = opj(DATA_DIR, data_type, 'images/images_%s_%s' % ('lung', IMG_SIZE))
    mask_dir = opj(DATA_DIR, data_type, 'images/masks_%s_%s' % ('lung', IMG_SIZE))
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    info_df = pd.DataFrame(columns=[ID, 'mask_num', 'mask_pixel'])
    for file_name in tqdm(os.listdir(src_images_dir)):
        img = cv2.imread(opj(src_images_dir, file_name))
        leftmask = cv2.imread(opj(src_masks_dir, 'leftMask',file_name), cv2.IMREAD_GRAYSCALE)
        rightmask = cv2.imread(opj(src_masks_dir, 'rightMask',file_name), cv2.IMREAD_GRAYSCALE)

        mask = leftmask + rightmask
        mask = mask.clip(0, 255)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(opj(image_dir, '%s_%s' % (dataset, file_name)), img)
        cv2.imwrite(opj(mask_dir, '%s_%s' % (dataset, file_name)), mask)

        _id = '%s_%s' % (dataset, file_name.split('/')[-1][:-4])

        info_df.loc[len(info_df)] = [_id, 1, (mask > 125).sum()]
    for key in info_keys:
        info_df[key] = 0
    info_df['PatientSex'] = 0
    info_df['ViewPosition'] = 0
    info_df.to_csv(opj(DATA_DIR, 'meta/%s_%s_meta.csv' % (data_type, dataset)), index=False)
