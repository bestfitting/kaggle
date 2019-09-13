import sys
sys.path.insert(0, '..')
import pandas as pd
from tqdm import tqdm
import cv2
import glob
import pydicom

from config.config import *
from utils.common_util import *

def get_dcm_info(dataset):
    value_list = []
    for info_key in info_keys:
        value_list.append(dataset.get(info_key))
    return value_list

def get_mask(df, file):
    mask_pixels = df.loc[df[ID] == file][TARGET].values
    mask = np.zeros((IMG_SIZE, IMG_SIZE))
    for i, mask_pixel in enumerate(mask_pixels):
        if mask_pixel != '-1':
            mask = mask + run_length_decode(mask_pixel, IMG_SIZE, IMG_SIZE).T
    if mask.max() != 0:
        if mask.max() != 255:
            print('%s    %s' % (file, mask.max()))
        mask = mask.clip(0, 255)
        return mask
    else:
        return None

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))
    info_keys = ['SOPClassUID', 'PatientName',
                'PatientID', 'PatientAge', 'PatientSex', 'Modality', 'BodyPartExamined', 'ViewPosition',

                'SpecificCharacterSet', 'SOPInstanceUID', 'StudyDate', 'StudyTime', 'AccessionNumber',
                'ConversionType', "ReferringPhysicianName", 'SeriesDescription', "PatientBirthDate",
                'StudyInstanceUID', 'SeriesInstanceUID', 'StudyID', 'SeriesNumber  ', 'InstanceNumber',
                'PatientOrientation', 'SamplesPerPixel', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelSpacing',
                'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'LossyImageCompression', 'LossyImageCompressionMethod']

    os.makedirs(opj(DATA_DIR, 'meta'), exist_ok=True)
    for data_type in ['train', 'test']:
        dicom_dir = opj(DATA_DIR, data_type, 'dicom_files/*.dcm')
        image_dir = opj(DATA_DIR, data_type, 'images/images_%s' % IMG_SIZE)
        mask_dir = opj(DATA_DIR, data_type, 'images/masks_%s' % IMG_SIZE)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        df = pd.read_csv(opj(DATA_DIR, 'raw/train-rle.csv'), index_col=0).reset_index()

        info_df = pd.DataFrame(columns=info_keys+[ID, 'mask_num', 'mask_pixel'])
        for file_path in tqdm(glob.glob(dicom_dir)):
            dataset = pydicom.dcmread(file_path)
            img = dataset.pixel_array
            _id = file_path.split('/')[-1][:-4]
            cv2.imwrite(opj(image_dir, file_path.split('/')[-1].replace('dcm', 'png')), img)

            mask_pixel = 0
            mask_num = 0
            if data_type == 'train':
                mask = get_mask(df, _id)
                if mask is not None:
                    cv2.imwrite(opj(mask_dir, file_path.split('/')[-1].replace('dcm', 'png')), mask)
                    mask_pixel = (mask > 125).sum()
                mask_num = (df[ID] == _id).sum()
            value_list = get_dcm_info(dataset)

            info_df.loc[len(info_df)] = value_list + [_id, mask_num, mask_pixel]
        info_df.loc[info_df['PatientSex'] == 'F', 'PatientSex'] = 0
        info_df.loc[info_df['PatientSex'] == 'M', 'PatientSex'] = 1
        info_df.loc[info_df['ViewPosition'] == 'AP', 'ViewPosition'] = 0
        info_df.loc[info_df['ViewPosition'] == 'PA', 'ViewPosition'] = 1
        info_df.to_csv(opj(DATA_DIR, 'meta/%s_meta.csv' % data_type), index=False)
