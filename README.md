# FaceCoresetNet: Differentiable Coresets for Face Set Recognition

Official github repository for FaceCoresetNet: Differentiable Coresets for Face Set Recognition


> Abstract: In set-based face recognition, we aim to compute the most discriminative descriptor from an unbounded set of images and videos showing a single person. A discriminative descriptor balances two policies when aggregating information from a given set. The first is a quality-based policy: emphasizing high-quality and down-weighting low-quality images. The second is a diversity-based policy: emphasizing unique images in the set and down-weighting multiple occurrences of similar images as found in video clips which can overwhelm the set representation.
This work frames face-set representation as a differentiable coreset selection problem. Our model learns how to select a small coreset of the input set that balances quality and diversity policies using a learned metric parameterized by the face quality, optimized end-to-end. The selection process is a differentiable farthest-point-sampling (FPS) realized by approximating the non-differentiable Argmax operation with differentiable sampling from the Gumbel-Softmax distribution of distances. The small coreset is later used as queries in a self and cross-attention architecture to enrich the descriptor with information from the whole set. Our model is order-invariant and linear in the input set size.
We set a new SOTA to set face verification on the IJB-B and IJB-C datasets. Our code is publicly available \footnote{\url{https://github.com/ligaripash/face_set_adaface/tree/fps_followed_by_pool-IJBB}}.


<img src="assets/main_figure.png"  />


# Installation

```
conda create --name adaface pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
conda activate adaface
conda install scikit-image matplotlib pandas scikit-learn 
pip install -r requirements.txt
```

# Dataset (MS1MV2)
1. Download MS1M-ArcFace (85K ids/5.8M images) from [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) and unzip at DATASET_ROOT
2. Unpack mxrecord files to imgs with the following code.
```
python convert.py --rec_path <DATASET_ROOT>/faces_emore
```

# Train
```
# training small model (resnet18) on a subset of MS1MV2 dataset
python main.py \
    --data_root <DATASET_ROOT> \
    --train_data_path faces_emore/imgs \
    --val_data_path faces_emore \
    --train_data_subset \
    --prefix run_ir18_ms1mv2_subset \
    --gpus 2 \
    --use_16bit \
    --batch_size 512 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2
```

# Pretrained Models


| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R18  | CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing) |
| R18  | VGGFace2     | [gdrive](https://drive.google.com/file/d/1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv/view?usp=sharing) |
| R18  | WebFace4M     | [gdrive](https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing) |
| R50  | CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing) |
| R50  | WebFace4M     | [gdrive](https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing) |
| R50  | MS1MV2     | [gdrive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) |
| R100 | MS1MV2     | [gdrive](https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view?usp=sharing) |
| R100 | MS1MV3     | [gdrive](https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing) |
| R100 | WebFace4M  | [gdrive](https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing) |
| R100 | WebFace12M | [gdrive](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing) |


# Inferece
Download the pretrained adaface model and place it in `pretrained/`

For inference, refer to 
```
python inference.py
```
We provide example images for inference. 

|                              img1                              |                              img2                              |                                                           img3 |
|:--------------------------------------------------------------:|:--------------------------------------------------------------:|---------------------------------------------------------------:|
| <img src="face_alignment/test_images/img1.jpeg" width="200" /> | <img src="face_alignment/test_images/img2.jpeg" width="200" /> | <img src="face_alignment/test_images/img3.jpeg" width="200" /> |

The similarity score result should be 
```
tensor([[ 1.0000,  0.7334, -0.0655],
        [ 0.7334,  1.0000, -0.0277],
        [-0.0655, -0.0277,  1.0000]], grad_fn=<MmBackward0>)
```
# Validation

## High Quality Image Validation Sets (LFW, CFPFP, CPLFW, CALFW, AGEDB)
For evaluation on 5 HQ image validation sets with pretrained models,
refer to 
```
bash validation_hq/eval_5valsets.sh
```

| Arch | Dataset        | Method   | LFW    | CFPFP  | CPLFW   | CALFW   | AGEDB  | AVG       |
|------|----------------|----------|--------|--------|---------|---------|--------|-----------|
| R18  | CASIA-WebFace	 | AdaFace  | 0.9913 | 0.9259 | 0.8700  | 0.9265  | 0.9272 | 0.9282    |
| R18  | VGGFace2       | AdaFace  | 0.9947 | 0.9713 | 0.9172  | 0.9390  | 0.9407 | 0.9526    |
| R18  | WebFace4M      | AdaFace  | 0.9953 | 0.9726 | 0.9228  | 0.9552  | 0.9647 | 0.9621    |
| R50  | CASIA-WebFace	 | AdaFace  | 0.9942 | 0.9641 | 0.8997  | 0.9323  | 0.9438 | 0.9468    |
| R50  | MS1MV2         | AdaFace  | 0.9982 | 0.9786 | 0.9283  | 0.9607  | 0.9785 | 0.9688    |
| R50  | WebFace4M      | AdaFace  | 0.9978 | 0.9897 | 0.9417  | 0.9598  | 0.9778 | 0.9734    |
| R100 | MS1MV2         | AdaFace  | 0.9982 | 0.9849 | 0.9353  | 0.9608  | 0.9805 | 0.9719    |
| R100 | MS1MV3         | AdaFace  | 0.9978 | 0.9891 | 0.9393  | 0.9602  | 0.9817 | 0.9736    |
| R100 | WebFace4M      | AdaFace  | 0.9980 | 0.9917 | 0.9463  | 0.9605  | 0.9790 | 0.9751    |
| R100 | WebFace12M     | AdaFace  | 0.9982 | 0.9926 | 0.9457  | 0.9612  | 0.9800 | 0.9755    |


#### Comparison with Other Methods

| Arch | Dataset       | Method    | LFW    | CFPFP  | CPLFW  | CALFW  | AGEDB  | AVG    |
|------|---------------|-----------|--------|--------|--------|--------|--------|--------|
| R50  | CASIA-WebFace	 | AdaFace  | 0.9942 | 0.9641 | 0.8997  | 0.9323  | 0.9438 | 0.9468    |
| R50  | CASIA-WebFace | (ArcFace) | 0.9945 | 0.9521 | NA      | NA     | 0.9490 | NA        |
| R100 | MS1MV2         | AdaFace  | 0.9982 | 0.9849 | 0.9353  | 0.9608  | 0.9805 | 0.9719    |
| R100 | MS1MV2        | (ArcFace) | 0.9982 | NA     | 0.9208 | 0.9545 | NA     | NA     |


## Mixed Quality Scenario (IJBB, IJBC Dataset)

For IJBB, IJBC validation, refer to 
```
cd validation_mixed
bash eval_ijb.sh
```

| Arch | Dataset    | Method      | IJBB TAR@FAR=0.01% | IJBC TAR@FAR=0.01% |
|------|------------|-------------|--------------------|--------------------|
| R18  | VGG2       | AdaFace | 90.67              | 92.95              |
| R18  | WebFace4M  | AdaFace | 93.03              | 94.99              |
| R50  | WebFace4M  | AdaFace | 95.44              | 96.98              |
| R50  | MS1MV2     | AdaFace | 94.82              | 96.27              |
| R100 | MS1MV2     | AdaFace | 95.67              | 96.89              |
| R100 | MS1MV3     | AdaFace | 95.84              | 97.09              |
| R100 | WebFace4M  | AdaFace      | 96.03              | 97.39              |
| R100 | WebFace12M | AdaFace      | 96.41              | 97.66              |

#### Comparison with Other Methods
- Numbers for other methods come from their respective papers.

| Arch | Dataset       | Method           | Venue  | IJBB TAR@FAR=0.01% | IJBC TAR@FAR=0.01% |
|------|---------------|------------------|--------|--------------------|--------------------|
| R100 | MS1MV2        | **AdaFace**          | CVPR22 | **95.67**          | **96.89**          |
| R100 | MS1MV2        | (MagFace)        | CVPR21 | 94.51              | 95.97              |
| R100 | MS1MV2        | (SCF-ArcFace)    | CVPR21 | 94.74              | 96.09              |
| R100 | MS1MV2        | (BroadFace)      | ECCV20 | 94.97              | 96.38              |
| R100 | MS1MV2        | (CurricularFace) | CVPR20 | 94.80              | 96.10              |
| R100 | MS1MV2        | (MV-Softmax)     | AAAI20 | 93.60              | 95.20              |
| R100 | MS1MV2        | (AFRN)           | ICCV19 | 88.50              | 93.00              |
| R100 |        MS1MV2 | (ArcFace)        | CVPR19 | 94.25              | 96.03              |
| R100 |        MS1MV2 |        (CosFace) | CVPR18 | 94.80              | 96.37              |

| Arch | Dataset    | Method           |  IJBC TAR@FAR=0.01% |
|------|------------|------------------|---------------------|
| R100 | WebFace4M  | **AdaFace**      |  **97.39**              |
| R100 | WebFace4M  | (CosFace)        |  96.86              |
| R100 | WebFace4M  | (ArcFace)        |  96.77              |
| R100 | WebFace4M  | (CurricularFace) |  97.02              |

| Arch | Dataset    | Method           |  IJBC TAR@FAR=0.01% |
|------|------------|------------------|---------------------|
| R100 | WebFace12M | **AdaFace**      |  **97.66**              |
| R100 | WebFace12M | (CosFace)        |  97.41              |
| R100 | WebFace12M | (ArcFace)        |  97.47              |
| R100 | WebFace12M | (CurricularFace) |  97.51              |

# Low Quality Scenario (IJBS)

For IJBB, IJBC validation, refer to
```
cd validation_lq
python validate_IJB_S.py
```

#### Comparison with Other Methods


|      |                  |  |  Sur-to-Single            |        |       | Sur-to-Book |        |       | Sur-to-Sur |           |          | TinyFace  |           |
|------|------------------|----------|--------------|--------|-------|-------------|--------|-------|------------|-----------|----------|-----------|-----------|
| Arch | Method           | Dataset  | Rank1       | Rank5 | 1%    | Rank1       | Rank5 | 1%    | Rank1      | Rank5     | 1%       | rank1     | rank5     |
| R100 | **AdaFace**      | WebFace4M |        **70.42** | **75.29**  | **58.27** | **70.93**   | **76.11**  | **58.02** | **35.05**  | **48.22** | **4.96** | **72.02** | **74.52** |
| R100 | **AdaFace**      | MS1MV2   | **65.26**        | **70.53**  | **51.66** | **66.27**   | **71.61**  | **50.87** | **23.74**  | **37.47** | 2.50     | **68.21** | **71.54** |
| R100 | (CurricularFace) | MS1MV2   | 62.43        | 68.68  | 47.68 | 63.81       | 69.74  | 47.57 | 19.54      | 32.80     | 2.53     | 63.68     | 67.65     |
| R100 | (URL)            | MS1MV2   | 58.94        | 65.48  | 37.57 | 61.98       | 67.12  | 42.73 | NA         | NA        | NA       | 63.89     | 68.67     |
| R100 | (ArcFace)        | MS1MV2   | 57.35        | 64.42  | 41.85 | 57.36       | 64.95  | 41.23 | NA         | NA        | NA       | NA        | NA        |
| R100 | (PFE)            | MS1MV2   | 50.16        | 58.33  | 31.88 | 53.60       | 61.75  | 35.99 | 9.20       | 20.82     | 0.84     | NA        | NA        |

- Sur-to-Single: Protocol comparing surveillance video (probe) to single enrollment image (gallery)
- Sur-to-Book: Protocol comparing surveillance video (probe) to all enrollment images (gallery)
- Sur-to-Sur: Protocol comparing surveillance video (probe) to surveillance video (gallery)
