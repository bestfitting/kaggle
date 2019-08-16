**The solution to kaggle competition Generative Dog Images**
the detail description is here:
https://www.kaggle.com/c/generative-dog-images/discussion/104281#latest-600250.

The code is base on the two projects:
https://github.com/ajbrock/BigGAN-PyTorch
https://github.com/rosinality/style-based-gan-pytorch

**How to run:**
1.confige data dir according to you local setting
1.1 dog_preprocess.py  line 14 and line 15: 
1.2 utils.py line 542
<br>
2. Run:
<pre>python dog_preprocess.py to save dogs with same type to dir accordingly.</pre>
<br>
3. run:
<pre>python calculate_inception_moments.py  --base_root ../output --dataset DogOrigin96</pre>
<br>
4.FID 12.4<br>
<pre>
export CUDA_VISIBLE_DEVICES=0;python train.py --shuffle --batch_size 32 --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 200 --num_D_steps 1 --G_lr 1e-4 --D_lr 6e-4 --dataset DogOrigin96 --bottom_width 6 --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init ortho --D_init ortho --ema --use_ema --ema_start 2000 --test_every 25 --save_every 10 --num_best_copies 5 --num_save_copies 2 --G_ch 24 --D_ch 24 --seed 0 --augment 1 --add_blur --add_style --on_kaggle --base_root ../output --crop_mode 8 --experiment_name i96_ch24_hinge_ema_dstep1_bs32_noatt_glr0001_glr0006_aug_init_ortho_blur_style_origin_crop_mode8
</pre>
5.FID 13.4<br>
<pre>
export CUDA_VISIBLE_DEVICES=0;python train.py --shuffle --batch_size 32 --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 200 --num_D_steps 1 --G_lr 1e-4 --D_lr 6e-4 --dataset DogOrigin96 --bottom_width 6 --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init ortho --D_init ortho --ema --use_ema --ema_start 2000 --test_every 25 --save_every 10 --num_best_copies 5 --num_save_copies 2 --G_ch 24 --D_ch 24 --seed 0 --augment 1 --add_blur --add_style --on_kaggle --base_root ../output --crop_mode 3 --experiment_name i96_ch24_hinge_ema_dstep1_bs32_noatt_glr0001_glr0006_aug_init_ortho_blur_style_origin_crop_mode3
</pre>