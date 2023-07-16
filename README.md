# FaceCoresetNet: Differentiable Coresets for Face Set Recognition

Official github repository for FaceCoresetNet: Differentiable Coresets for Face Set Recognition


> Abstract: In set-based face recognition, we aim to compute the most discriminative descriptor from an unbounded set of images and videos showing a single person. A discriminative descriptor balances two policies when aggregating information from a given set. The first is a quality-based policy: emphasizing high-quality and down-weighting low-quality images. The second is a diversity-based policy: emphasizing unique images in the set and down-weighting multiple occurrences of similar images as found in video clips which can overwhelm the set representation.
This work frames face-set representation as a differentiable coreset selection problem. Our model learns how to select a small coreset of the input set that balances quality and diversity policies using a learned metric parameterized by the face quality, optimized end-to-end. The selection process is a differentiable farthest-point-sampling (FPS) realized by approximating the non-differentiable Argmax operation with differentiable sampling from the Gumbel-Softmax distribution of distances. The small coreset is later used as queries in a self and cross-attention architecture to enrich the descriptor with information from the whole set. Our model is order-invariant and linear in the input set size.
We set a new SOTA to set face verification on the IJB-B and IJB-C datasets. Our code is publicly available \footnote{\url{https://github.com/ligaripash/face_set_adaface/tree/fps_followed_by_pool-IJBB}}.


<img src="assets/arch.png"  />


# Installation and Preparation

## 1. Environment
We use pytorch (1.10.0) in our experiments.
```
pip install -r requirements.txt
```

## 2. Pretrained Models
We release the FaceCoresetNet model pretrained on AdaFace backbone. 
The backbone is trained on WebFace4M dataset. 
And FaceCoresetNet is trained on a subset of WebFace4M dataset. 

- Pretrained Model (You need both):
  - Precomputed Class Center for WebFace4M subset: [center_WebFace4MAdaFace_webface4m_subset.pth](https://drive.google.com/file/d/1WmiWjLSsfQU2PTwQAvnrep9u6Jfvd3tR/view?usp=share_link)
  - Pretrained FaceCoresetNet model:  put link here

Place these two files under `pretrained_models/`
```
pretrained_models/
├── FaceCoresetNet_AdaFaceWebFace4M.ckpt                         
└── center_WebFace4MAdaFace_webface4m_subset.pth         
```

# Testing on Arbitrary Videos (Demo)


# Evaluation

### IJBB and IJBC

For evaluation with IJBB/IJBC you may download the related files from. 
- [InsightFace IJB Dataset](https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb) or
- [Download](https://forms.gle/7zURRo2tca96ZKyf6) for convenience, here is an additional link we provide.

Place the downloaded files in `<DATA_ROOT>`, i.e
```
<DATA_ROOT>
└── IJB
    ├── aligned (only needed during training)                                                                                                                      │➜  ffhq mv FFHQ_png_512.zip /hddata/data/ffhq/
    └── insightface_helper
        ├── ijb                                                                                                                             │➜  ffhq mv FFHQ_png_512.zip /hddata/data/ffhq/
        └── meta        
```

Refer to the below code for evaluation.
```bash
cd validation_IJBB_IJBC
bash scripts/run.sh  # DATA_ROOT and IJB_META_PATH has to be specified.
```


# Training from scratch

## WebFace4M Subset (as in paper)

- **Dataset Preparation**: For training FaceCoresetNet we use pre-computed feature vectors.
  Using a face recognition model trained on WebFace4M, we extract 512 dim feature vectors on a subset of WebFace4M.
  - Precomputed training data features (adaface_webface4m_subset_ir101_style35_augmenterv3_fp16): [precomputed_features](https://drive.google.com/file/d/1U615roLaCGYAmcWRVOJWO1jk6e8Oo3sA/view?usp=share_link)
  - Validation Set (IJBB): [Download](https://forms.gle/7zURRo2tca96ZKyf6)

  - Place the files under `<DATA_ROOT>`
```
<DATA_ROOT>
├── adaface_webface4m_subset_ir101_style35_augmenterv3_fp16/
└── IJB
    ├── aligned (only needed during training)                                                                                                                      │➜  ffhq mv FFHQ_png_512.zip /hddata/data/ffhq/
    └── insightface_helper
        ├── ijb                                                                                                                             │➜  ffhq mv FFHQ_png_512.zip /hddata/data/ffhq/
        └── meta        
```

- Get pretrained face recognition model backbone
  - AdaFace trained on WebFace4M [AdaFaceWebFace4M.ckpt](https://drive.google.com/file/d/19AfGaGZjDqwPQR00kck0GBknePmQOFnU/view?usp=share_link)
  - Place the files under `pretrained_models/`


For training script, refer to
```bash
cd FaceCoresetNet
bash scripts/run.sh  # DATA_ROOT has to be specified. 
```

## Extract features using different model
The raw WebFace4M subset dataset used in the paper can be downloaded here [AdaFace4M_subset](https://drive.google.com/file/d/1LuhyxoTdMoVTsrlmZ5_F26Oia3bXsIpu/view?usp=share_link).

We also provide the preprocessing code for creating
1. precomputed feature blob 
   1. First you should extract individual images from the above mxdataset, using [preprocess/extract_images.py](./preprocess/extract_images.py)
   2. Make sure that images are saved correctly (check the color channel)
   3. Then use [preprocess/precompute_features.py](./preprocess/precompute_features.py) to save the features.
2. class center 
   1. [preprocess/make_center.py](./preprocess/make_center.py) creates a `pth` file with class center for the dataset. This will be used in loss calculation.
