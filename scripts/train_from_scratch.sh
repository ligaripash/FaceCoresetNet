
DATA_ROOT='<YOUR_CHOICE>'
PRECOMPUTE_TRAIN_REC='FaceCoresetNet.ckpt'
BACKBONE_MODEL='../pretrained_models/AdaFaceWebFace4M.ckpt'


python main_template.py \
          --data_root datasets \
          --train_data_path ${DATA_ROOT}\webface4m_subset_images \
          --val_data_path faces_emore \
          --prefix h_0.666_no_noise_fix_aug_0.5_K=3_max_t_30_gpu_1_b_20_sa \
          --devices 1 \
          --accelerator gpu \
          --strategy auto \
          --batch_size 20 \
          --epochs 150 \
          --lr_milestones 60 \
          --head adaface \
          --low_res_augmentation_prob 0.5 \
          --crop_augmentation_prob 0.5 \
          --photometric_augmentation_prob 0.5 \
          --num_workers 6 \
          --arch ir_101 \
          --limit_train_batches 0.1 \
          --lr 0.0001 \
          --start_from_model_statedict ../pretrained_models/AdaFaceWebFace4M.ckpt \
          --m 0.85 \
          --weight_decay 0.0001 \
          --gamma_lr 0.0001 \
          --coreset_size 3 \
          --gradient_clip_val 1 \
          --precision 32 \
          --max_template_size 30 \
          --ijb_root ${DATA_ROOT}/ijb \
          --h 0.666
