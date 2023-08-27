
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
DATA_PATH=D:/temp/data/

python ./main_template.py --train_data_path ${DATA_PATH}/webface4m_subset_images --data_root datasets --prefix K=3_gamma=learned_new_q_s=48_clip_max_3_h_0.333 --devices 1 --accelerator gpu --strategy auto --batch_size 20 --epochs 150 --lr_milestones 13 --head adaface --low_res_augmentation_prob 0.5 --crop_augmentation_prob 0.5 --photometric_augmentation_prob 0.5 --num_workers 6 --arch ir_101 --limit_train_batches 0.1 --lr 0.0001 --start_from_model_statedict D:/temp/gil/face_set_recognition/AdaFace/pretrained/AdaFaceWebFace4M.ckpt --m 0.8 --weight_decay 0.001 --gamma_lr 0.0001 --coreset_size 3 --gradient_clip_val 1 --precision 32 --max_template_size 20 --ijb_root ${DATA_PATH}/ijb --h 0.333 --s 48 --lr_gamma 0.2 --resume_from_checkpoint ./pretrained_models/FaceCoresetNet.ckpt --evaluate --coreset_size 3 --wandb_disable