
DATA_ROOT="/media/keller/data1/"

# IJBB
#python validate_IJB_BC.py \
#      --data_root ${DATA_ROOT} \
#      --ijb_meta_path IJB/insightface_helper/ijb \
#      --dataset_name IJBB \
#      --pretrained_model_path ../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt \
#      --center_path ../pretrained_models/center_WebFace4MAdaFace_webface4m_subset.pth

# IJBC

python validate_IJB_BC_compute_templates.py \
       --calc_avg_per_media False \
       --experiment ir101_webface4m_average \
       --dataset_name IJBC \
       --model_name ir101_webface4m \
       --data_root ../datasets/ijb \
       --batch_size 1 \
       --use_flip_test False

#python validate_IJB_BC.py \
#      --data_root ${DATA_ROOT} \
#      --ijb_meta_path IJB/insightface_helper/ijb \
#      --dataset_name IJBC \
#      --pretrained_model_path ../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt

