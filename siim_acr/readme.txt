0. Environment and file descriptions
0.0 The basic Runtime Environment is python3.6, pytorch1.1.0, you can refer to requriements.txt to set up your environment.
0.1 siim_open.7z: my code
0.2 For all the 7z files, you can extract them with: 7z x siim_open.7z

1. Prepare data
1.1 Please unpack siim_open.7z into a directory, I refer it as: CODE_DIR
1.2 Please config your local directory in CODE_DIR/src/config/config.py
1.3 Please move dicom-images-train and dicom-images-test-stage_1 to DATA_DIR/train/dicom_files (provided by Kaggle forum, is not included in my solution file)
1.4 Please move dicom-images-test-stage_2 to DATA_DIR/test/dicom_files (provided by Kaggle forum, is not included in my solution file)
1.5 Please move train-rle.csv to DATA_DIR/raw and rename train-rle_stage1.csv (provided by Kaggle, is not included in my solution file)
1.6 Please move stage_2_train-rle.csv to DATA_DIR/raw and rename train-rle.csv (provided by Kaggle, is not included in my solution file)
1.7 Please move stage_2_sample_submission.csv to DATA_DIR/raw and rename sample_submission.csv (provided by Kaggle, is not included in my solution file)
1.8 Please download Montgomery County X-ray Set to DATA_DIR/external/NLM-MontgomeryCXRSet
https://lhncbc.nlm.nih.gov/publication/pub9931
1.9 Please download CheXpert to DATA_DIR/external/CheXpert-v1.0
https://stanfordmlgroup.github.io/competitions/chexpert/
1.10 Please download NIH Chest X-ray Dataset to DATA_DIR/external/NIH_chest_x-rays
https://www.kaggle.com/nih-chest-xrays/data
1.11 Please download NIH_Data_Entry_2017.csv to DATA_DIR/raw
https://www.kaggle.com/nih-chest-xrays/data
1.12 Please download kaggle_to_nih_id.csv to DATA_DIR/raw
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/100941

2. Preprocess data
2.1 DICOM to PNG, and extracts information of the images
process :
    cd CODE_DIR/src/data_process/
    python s1_dicom2image.py
input :
    DATA_DIR/raw/train-rle.csv
    DATA_DIR/train/dicom_files/
    DATA_DIR/test/dicom_files/
output :
    DATA_DIR/train/images/images_1024/
    DATA_DIR/train/images/masks_1024/
    DATA_DIR/meta/train_meta.csv
    DATA_DIR/test/images/images_1024/
    DATA_DIR/meta/test_meta.csv

2.2 Merge mult-masks
process :
    cd CODE_DIR/src/data_process/
    python s2_merge_multmask_rle.py
input :
    DATA_DIR/raw/train-rle.csv
output :
    DATA_DIR/meta/train-rle.csv

2.3 Split training set and validation set
process :
    cd CODE_DIR/src/data_process/
    python s3_create_split_stage1.py
    python s4_create_split_stage2.py
input :
    DATA_DIR/raw/train-rle.csv
    DATA_DIR/raw/sample_submission.csv
output :
    DATA_DIR/split/

2.4 Process CheXpert image
process :
    cd CODE_DIR/src/data_process/
    python s5_process_chexpert_dataset.py
input :
    DATA_DIR/external/CheXpert-v1.0/train
    DATA_DIR/external/CheXpert-v1.0/valid
output :
    DATA_DIR/external/CheXpert-v1.0/images/images_1024

2.5 Split CheXpert
process :
    cd CODE_DIR/src/data_process/
    python s6_create_chexpert_split.py
input :
    DATA_DIR/external/CheXpert-v1.0/train.csv
    DATA_DIR/external/CheXpert-v1.0/valid.csv
output :
    DATA_DIR/split/chexpert_188521.csv


3.Crop image, only keep the lung area
3.1 Process lung data
process :
    cd CODE_DIR/src/data_process/
    python s7_create_lung_montgomery.py
    python s8_create_lung_split.py
input :
    DATA_DIR/external/NLM-MontgomeryCXRSet
output :
    DATA_DIR/meta/train_montgomery_meta.csv
    DATA_DIR/train/images/images_lung_1024/
    DATA_DIR/train/images/masks_lung_1024/
    DATA_DIR/lung_split/

3.2 Train/predict Lung-Segmentation model, resnet34, image size:256x256, augment:aug9
process :
    cd CODE_DIR/src/
    python train.py --arch unet_resnet34_cbam_v0a --batch_size 16 --loss SymmetricLovaszLoss --scheduler Adam3 --img_size 256 --out_dir siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds --epochs 30 --split_type lung_split --split_name random_folds4 --gpu_id 0 --crop_version lung --is_balance 0 --fold 0
    python test.py --arch unet_resnet34_cbam_v0a --batch_size 8 --img_size 256 --split_type split --dataset train --out_dir siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds --gpu_id 0 --augment default --predict_epoch final
    python test.py --arch unet_resnet34_cbam_v0a --batch_size 8 --img_size 256 --split_type split --dataset test --out_dir siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds --gpu_id 0 --augment default --predict_epoch final
input :
    DATA_DIR/lung_split/
    DATA_DIR/train/images_lung_1024/
    DATA_DIR/train/masks_lung_1024/
    DATA_DIR/split/train.csv
    DATA_DIR/split/test.csv
output :
    RESULT_DIR/models/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/
    RESULT_DIR/logs/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/
    RESULT_DIR/submissions/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/

3.3 Generate cropped images
process :
    cd CODE_DIR/src/data_process/
    python s9_create_cropv3.py
    python s10_create_cropv3_split.py
input :
    RESULT_DIR/submissions/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/
    DATA_DIR/train/images/images_1024
    DATA_DIR/train/images/masks_1024
    DATA_DIR/test/images/images_1024
output :
    DATA_DIR/cropv3_split/
    DATA_DIR/train/images/images_cropv3/
    DATA_DIR/train/images/masks_cropv3/
    DATA_DIR/train/images/boxes_cropv3.csv.gz
    DATA_DIR/meta/train_cropv3_meta.csv
    DATA_DIR/test/images/images_cropv3/
    DATA_DIR/test/images/boxes_cropv3.csv.gz
    DATA_DIR/meta/test_cropv3_meta.csv

4.Generate pseudo labels
4.1 Train resnet34 as base model, image size:768x768, augment:aug9
process :
    cd CODE_DIR/src/
    python train.py --arch unet_resnet34_cbam_v0a --batch_size 9 --loss SymmetricLovaszLoss --scheduler Adam3 --img_size 768 --out_dir siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds --epochs 40 --split_type split --split_name random_folds4 --gpu_id 1,2,3 --is_balance 1 --sample_times 3 --fold 0
input :
    DATA_DIR/split/random_folds4/
    DATA_DIR/train/images/images_1024/
    DATA_DIR/train/images/masks_1024/
output :
    RESULT_DIR/logs/siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/
    RESULT_DIR/models/siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/

4.2 Predict CheXpert by resnet34 base model
process :
    cd CODE_DIR/src/
    python test.py --arch unet_resnet34_cbam_v0a --batch_size 2 --img_size 768 --split_type split --dataset chexpert --out_dir siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds --gpu_id 0 --augment default --predict_pos 1 --predict_epoch final
input :
    DATA_DIR/split/chexpert_188521.csv
    DATA_DIR/external/CheXpert-v1.0/images/images_1024/
    RESULT_DIR/models/siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/fold0/final.pth
output :
    RESULT_DIR/submissions/siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/fold0/epoch_final/default/

4.3 Generate pseudo label of CheXpert by resnet34 base model
process :
    cd CODE_DIR/src/data_process/
    python s11_generate_chexpert_pseudo_label.py
input :
    DATA_DIR/split/chexpert_188521.csv
    DATA_DIR/external/CheXpert-v1.0/images/images_1024/
    RESULT_DIR/submissions/siim_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/fold0/epoch_final/default/
output :
    DATA_DIR/pseudo/chexpert_pseudo.csv
    DATA_DIR/external/CheXpert-v1.0/images/masks_1024/

4.4 Predict CheXpert by Lung-Segmentation model
process :
    cd CODE_DIR/src/
    python test.py --arch unet_resnet34_cbam_v0a --batch_size 24 --img_size 256 --split_type split --dataset chexpert --out_dir siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds --gpu_id 1,2,3 --augment default --predict_epoch final
input :
    DATA_DIR/split/chexpert_188521.csv
    DATA_DIR/external/CheXpert-v1.0/images/images_1024/
    RESULT_DIR/models/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/fold0/final.pth
output :
    RESULT_DIR/submissions/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/fold0/epoch_final/default/

4.5 Crop CheXpert image
process :
    cd CODE_DIR/src/data_process/
    python s12_create_chexpert_cropv3.py
input :
    DATA_DIR/external/CheXpert-v1.0/images/images_1024/
    DATA_DIR/external/CheXpert-v1.0/images/masks_1024/
    RESULT_DIR/submissions/siim_lung_nobalance_unet_resnet34_cbam_v0a_i256_aug9_symmetriclovaszloss_4folds/fold0/epoch_final/default/
output :
    DATA_DIR/external/CheXpert-v1.0/images/images_cropv3/
    DATA_DIR/external/CheXpert-v1.0/images/masks_cropv3/
    DATA_DIR/external/CheXpert-v1.0/images/boxes_cropv3.csv.gz

4.6 Split crop CheXpert
process :
    cd CODE_DIR/src/data_process/
    python s13_create_chexpert_cropv3_split.py
input :
    DATA_DIR/split/chexpert_188521.csv
    DATA_DIR/external/CheXpert-v1.0/images/boxes_cropv3.csv.gz
output :
    DATA_DIR/cropv3_split/chexpert_188521.csv

5. Train/predict
5.1 Train cropv3_resnet34_768x768 4folds model(Public LB: 0.8740)
process :
    cd CODE_DIR/src/
    python train.py --arch unet_resnet34_cbam_v0a --batch_size 9 --loss SymmetricLovaszLoss --scheduler Adam3 --img_size 768 --out_dir siim_cropv3_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds --epochs 40 --split_type cropv3_split --split_name random_folds4 --gpu_id 1,2,3 --crop_version cropv3 --is_balance 1 --sample_times 3 --fold {0-3}
    python test.py --arch unet_resnet34_cbam_v0a --batch_size 2 --img_size 768 --split_type cropv3_split --split_name random_folds4 --dataset val --out_dir siim_cropv3_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds --gpu_id 1,2,3 --augment default --crop_version cropv3 --fold {0-3} --predict_epoch final
input :
    DATA_DIR/cropv3_split/random_folds4/
    DATA_DIR/train/images/images_cropv3/
    DATA_DIR/train/images/masks_cropv3/
output :
    RESULT_DIR/logs/siim_cropv3_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/
    RESULT_DIR/models/siim_cropv3_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/
    RESULT_DIR/submissions/siim_cropv3_unet_resnet34_cbam_v0a_i768_aug9_symmetriclovaszloss_4folds/

5.2 Train/predict x50_c1_576x576 10folds model
process :
    cd CODE_DIR/src/
    python train.py --arch unet_se_resnext50_cbam_v0a --batch_size 9 --loss SymmetricLovaszLoss --scheduler Adam3 --img_size 576 --out_dir siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds --epochs 20 --split_type cropv3_split --split_name random_folds10 --gpu_id 1,2,3 --crop_version cropv3 --is_balance 1 --sample_times 1 --ema --ema_start 7 --ema_decay 0.99 --pseudo chexpert_pseudo --pseudo_ratio 0.5 --fold {0-9}
    python test.py --arch unet_se_resnext50_cbam_v0a --batch_size 3 --img_size 576 --split_type cropv3_split --split_name random_folds10 --dataset test --out_dir siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds --gpu_id 1,2,3 --augment default --crop_version cropv3 --ema --augment default,fliplr --fold {0-9} --predict_epoch {top1-top3}
    python test.py --arch unet_se_resnext50_cbam_v0a --batch_size 3 --img_size 576 --split_type cropv3_split --split_name random_folds10 --dataset val --out_dir siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds --gpu_id 1,2,3 --augment default --crop_version cropv3 --ema --augment default,fliplr --fold {0-9} --predict_epoch {top1-top3}
input :
    DATA_DIR/cropv3_split/
    DATA_DIR/external/CheXpert-v1.0/images/images_cropv3/
    DATA_DIR/external/CheXpert-v1.0/images/masks_cropv3/
    DATA_DIR/train/images/images_cropv3/
    DATA_DIR/train/images/masks_cropv3/
output :
    RESULT_DIR/models/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/
    RESULT_DIR/logs/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/
    RESULT_DIR/submissions/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/

6. Ensemble
6.1 Augment ensemble: x50_c1_576x576 10folds model
process :
    cd CODE_DIR/src/ensemble
    python ensemble_crop.py --model_name siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds --fold {0-9} --epoch_name {top1-top3} --do_valid 1 --do_test 1 --crop_version cropv3 --augment_names default,fliplr
input :
    DATA_DIR/meta/train_cropv3_meta.csv
    DATA_DIR/train/images/boxes_cropv3.csv.gz
    DATA_DIR/test/images/boxes_cropv3.csv.gz
    RESULT_DIR/submissions/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/
output :
    RESULT_DIR/submissions/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/

6.2 Select top folds: x50_c1_576x576 10folds model
process :
    cd CODE_DIR/src/ensemble
    python select_topn_folds.py --model_name siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds --out_name i576_c1 --topn 3
input :
    RESULT_DIR/submissions/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/
output :
    RESULT_DIR/submissions/ensemble/select_folds/i576_c1_top3_folds.npy

6.3 Ensemble top3 folds: x50_c1_576x576 10folds model
process :
    cd CODE_DIR/src/ensemble
    python ensemble_models.py --en_cfgs siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds_select3_3epochs_tta2 --do_valid 0
input :
    RESULT_DIR/submissions/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds/
output :
    RESULT_DIR/submissions/ensemble/siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds_select3_3epochs_tta2/
