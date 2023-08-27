
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
DATA_PATH=D:/temp/data/

python ./main_template.py --data_root datasets --train_data_path ${DATA_PATH}/webface4m_subset_images --val_data_path faces_emore --prefix s=48_clip_max_3_h_0.333_aug_0.5_K=3_max_t_20_2gpu_m_0.8_wd_0.001 --devices 2 --accelerator gpu --strategy auto --batch_size 20 --epochs 150 --lr_milestones 20 --head adaface --low_res_augmentation_prob 0.5 --crop_augmentation_prob 0.5 --photometric_augmentation_prob 0.5 --num_workers 6 --arch ir_101 --limit_train_batches 0.1 --lr 0.0001 --start_from_model_statedict ./pretrained_models/AdaFaceWebFace4M.ckpt --m 0.8 --weight_decay 0.001 --gamma_lr 0.0001 --coreset_size 3 --gradient_clip_val 1 --precision 32 --max_template_size 20 --ijb_root ${DATA_PATH}/ijb --h 0.333 --s 48 --save_all_models
