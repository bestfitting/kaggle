from config.config import *

i576_c1_top3_folds = np.load(f'{RESULT_DIR}/select_folds/i576_c1_top3_folds.npy')
print('i576_c1_top3_folds', i576_c1_top3_folds)

# i576 c1 select 3 folds
siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds_select3_3epochs_tta2 = [
    *[{
        'model_name': 'siim_ema_c1_cropv3_unet_se_resnext50_cbam_v0a_i576_aug9_symmetriclovaszloss_10folds',
        'is_en': False, 'fold': fold, 'epoch_name': 'top%d'%top, 'augment': 'average', 'weight':  weight,
    } for fold in i576_c1_top3_folds for top,weight in zip([1, 2, 3], [0.5, 0.3, 0.2])],
]
