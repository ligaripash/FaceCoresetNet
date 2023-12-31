import numpy as np
import torchvision
from sklearn.manifold import TSNE
import argparse
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
#import net

from validation_IJBB_IJBC.insightface_ijb_helper.dataloader import prepare_dataloader
from validation_IJBB_IJBC.insightface_ijb_helper import eval_helper_identification
from validation_IJBB_IJBC.insightface_ijb_helper import eval_helper as eval_helper_verification


import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
import pandas as pd

import pickle

#write a function to sort an array of strings

def sort_array(array):
    array = np.array(array)
    return array[np.argsort(array)]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm

def infer_images(model, img_root, landmark_list_path, batch_size, use_flip_test, fusion_method, gpu_id):
    img_list = open(landmark_list_path)
    # img_aligner = ImageAligner(image_size=(112, 112))

    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_paths = []
    landmarks = []
    for img_index, each_line in enumerate(files):
        name_lmk_score = each_line.strip().split(' ')
        img_path = os.path.join(img_root, name_lmk_score[0])
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_paths.append(img_path)
        landmarks.append(lmk)
        faceness_scores.append(name_lmk_score[-1])

    print('total images : {}'.format(len(img_paths)))
    dataloader = prepare_dataloader(img_paths, landmarks, batch_size, num_workers=0, image_size=(112,112))

    model.eval()
    model.compute_feature()
    features = []
    norms = []
    DEBUG = False
    if DEBUG:
        i = 0
    blur_function = torchvision.transforms.GaussianBlur(21, sigma=(30, 30))

    TSNE_DIAG = False

    if TSNE_DIAG:
        with torch.no_grad():
            for images, idx in tqdm(dataloader):
                # images = images.to("cuda:{}".format(gpu_id))
                if i == 20:
                    break
                if i < 5:
                    images = blur_function(images)

                images = images.to("cuda:{}".format(gpu_id))
                # feature = model(images.to("cuda:{}".format(gpu_id)))
                feature = model(images)
                if isinstance(feature, tuple):
                    feature, norm = feature
                else:
                    norm = None

                if use_flip_test:
                    fliped_images = torch.flip(images, dims=[3])
                    flipped_feature = model(fliped_images.to("cuda:{}".format(gpu_id)))
                    if isinstance(flipped_feature, tuple):
                        flipped_feature, flipped_norm = flipped_feature
                    else:
                        flipped_norm = None

                    stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                    if norm is not None:
                        stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                    else:
                        stacked_norms = None

                    fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms,
                                                                        fusion_method=fusion_method)
                    features.append(fused_feature.cpu().numpy())
                    norms.append(fused_norm.cpu().numpy())
                else:
                    features.append(feature.cpu().numpy())
                    norms.append(norm.cpu().numpy())
                if DEBUG:
                    i += 1

        features = np.concatenate(features, axis=0)
        img_feats = np.array(features).astype(np.float32)
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        norms = np.concatenate(norms, axis=0)

        X_embedded = TSNE(n_components=2, learning_rate='auto',init = 'random', metric='cosine').fit_transform(img_feats)
        pass

    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            #images = images.to("cuda:{}".format(gpu_id))
            if DEBUG:
                if i == 20:
                    break
#            images = blur_function(images)
            images = images.to("cuda:{}".format(gpu_id))
            images = images.unsqueeze(0)
            #feature = model(images.to("cuda:{}".format(gpu_id)))
            feature = model(images)
            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                norm = None

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda:{}".format(gpu_id)))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    flipped_norm = None

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())
            if DEBUG:
                i+=1
    features = np.concatenate(features, axis=0)
    img_feats = np.array(features).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    norms = np.concatenate(norms, axis=0)

    #gil
    #assert len(features) == len(img_paths)

    return img_feats, faceness_scores, norms

def identification(data_root, dataset_name, img_input_feats, save_path):

    # Step1: Load Meta Data
    meta_dir = os.path.join(data_root, dataset_name, 'meta')
    if dataset_name == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (dataset_name.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (dataset_name.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (dataset_name.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (dataset_name.lower())
    gallery_s1_templates, gallery_s1_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    gallery_templates = np.concatenate(
        [gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate(
        [gallery_s1_subject_ids, gallery_s2_subject_ids])
    print(gallery_templates.shape, gallery_subject_ids.shape)

    media_record = "%s_face_tid_mid.txt" % dataset_name.lower()
    total_templates, total_medias = eval_helper_identification.read_template_media_list(
        os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)

    # # Step2: Get gallery Features
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = eval_helper_identification.image2template_feature(
        img_input_feats, total_templates, total_medias, gallery_templates, gallery_subject_ids, calc_avg_per_media=True)
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)

    # # step 4 get probe features
    probe_mixed_record = "%s_1N_probe_mixed.csv" % dataset_name.lower()
    probe_mixed_templates, probe_mixed_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = eval_helper_identification.image2template_feature(
        img_input_feats, total_templates, total_medias, probe_mixed_templates,
        probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids",
          probe_mixed_unique_subject_ids.shape)

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = eval_helper_identification.gen_mask(probe_ids, gallery_ids)
    identification_result = eval_helper_identification.evaluation(probe_feats, gallery_feats, mask)
    pd.DataFrame(identification_result, index=['identification']).to_csv(os.path.join(save_path, "identification_result.csv"))
    
def verification(data_root, dataset_name, img_input_feats, save_path, calc_avg_per_media = True, model=None, norms=None):
    templates, medias = eval_helper_verification.read_template_media_list(
        os.path.join(data_root, '%s/meta' % dataset_name, '%s_face_tid_mid.txt' % dataset_name.lower()))
    p1, p2, label = eval_helper_verification.read_template_pair_list(
        os.path.join(data_root, '%s/meta' % dataset_name,
                    '%s_template_pair_label.txt' % dataset_name.lower()))

    template_norm_feats, unique_templates = eval_helper_verification.image2template_feature(img_input_feats, templates, medias, calc_avg_per_media, model=model, norms=norms)
    #score = eval_helper_verification.verification(template_norm_feats, unique_templates, p1, p2)
    #score = eval_helper_verification.verification_quality_by_norm_metric(template_norm_feats, unique_templates, p1, p2)
    score = eval_helper_verification.verification2(template_norm_feats, unique_templates, p1, p2)
    # # Step 5: Get ROC Curves and TPR@FPR Table
    score_save_file = os.path.join(save_path, "verification_score.npy")
    np.save(score_save_file, score)
    result_files = [score_save_file]
    result = eval_helper_verification.write_result(result_files, save_path, dataset_name, label)
    os.remove(score_save_file)
    return result

def load_pretrained_model(model_name='ir50'):
    # load model and pretrained statedict

    if 'webface' in model_name:
        class_num = 205990
    else:
        class_num = 85742
    ckpt_path = adaface_models[model_name][0]
    arch = adaface_models[model_name][1]

    model = net.build_model(arch)
    import head
    statedict = torch.load(ckpt_path)['state_dict']
    head = head.build_head(head_type='adaface',
                                embedding_size=512,
                                class_num=class_num,
                                m=0.2,
                                h=0.333,
                                t_alpha=0.01,
                                s=64,
                                )

    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model_statedict = {key[5:]:val for key, val in statedict.items() if key.startswith('head.')}
    head.load_state_dict(model_statedict)
    model.eval()
    return model, head



def infer_images_or_get_from_pickle(dataset_name, model_name,
                                    model, img_root, landmark_list_path, args ):
    INFER_IMAGE = True
    precomputed_features_file = '_'.join((dataset_name, model_name + '.pickle'))
    precomputed_features_file = os.path.join(os.path.dirname(__file__), precomputed_features_file)
    if os.path.exists(precomputed_features_file):
        INFER_IMAGE = False
    if INFER_IMAGE:
        img_input_feats, faceness_scores, norms = infer_images(model=model,
                                                               img_root=img_root,
                                                               landmark_list_path=landmark_list_path,
                                                               batch_size=1,
                                                               use_flip_test=False,
                                                               fusion_method=None,
                                                               gpu_id=0)
        data = {}
        data['img_feats'] = img_input_feats
        data['norms'] = norms
        data['faceness_scores'] = faceness_scores
        with open(precomputed_features_file, 'wb') as f:
            pickle.dump(data, f)

    else:
        with open(precomputed_features_file, 'rb') as f:
            data = pickle.load(f)
            img_input_feats = data['img_feats']
            norms = data['norms']
            faceness_scores = data['faceness_scores']

    return img_input_feats, faceness_scores, norms

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do ijb test')
    # general
    parser.add_argument('--dataset_name', default='IJBC', type=str, help='dataset_name, set to IJBC or IJBB')
    parser.add_argument('--data_root', default='/data/data/faces/IJB/insightface_helper/ijb')
    parser.add_argument('--model_name', default='ir50')

    parser.add_argument('--experiment', default='no_media_averaging')
    parser.add_argument('--calc_avg_per_media', type=str2bool, default='True')

    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--fusion_method', type=str, default='pre_norm_vector_add', choices=('average',
                                           'norm_weighted_avg', 'pre_norm_vector_add', 'concat'))
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    assert dataset_name in ['IJBC', 'IJBB']
    print('dataset name: ', dataset_name)
    print('use_flip_test', args.use_flip_test)
    print('fusion_method', args.fusion_method)
    print('calc_avg_per_media', args.calc_avg_per_media)


    adaface_models = {
        'ir18_casia': ["../pretrained/adaface_ir18_casia.ckpt", 'ir_18'],
        'ir18_webface4m': ["../pretrained/adaface_ir18_webface4m.ckpt", 'ir_18'],
        #'ir18_vgg2': ["../pretrained/adaface_ir18_vgg2.ckpt", 'ir_18'],
        'ir18_vgg2': ["../experiments/run_ir18_ms1mv2_subset_04-22_5/last.ckpt", 'ir_18'],
        'ir50_casia': ["../pretrained/adaface_ir50_casia.ckpt", 'ir_50'],
        'ir50_webface4m': ["../pretrained/adaface_ir50_webface4m.ckpt", 'ir_50'],
        'ir50_ms1mv2': ["../pretrained/adaface_ir50_ms1mv2.ckpt", 'ir_50'],
        'ir101_ms1mv2': ["../pretrained/adaface_ir101_ms1mv2.ckpt", 'ir_101'],
        'ir101_ms1mv3': ["../pretrained/adaface_ir101_ms1mv3.ckpt", 'ir_101'],
        'ir101_webface4m': ["../pretrained/adaface_ir101_webface4m.ckpt", 'ir_101'],
        'ir101_webface12m': ["../pretrained/adaface_ir101_webface12m.ckpt", 'ir_101'],
    }
    assert args.model_name in adaface_models
    save_path = './result/{}/{}/{}'.format(args.dataset_name, args.model_name, args.experiment)
    print('result save_path', save_path)
    os.makedirs(save_path, exist_ok=True)

    # load model
    model, kernel = load_pretrained_model(args.model_name)
    model.to('cuda:{}'.format(args.gpu))

    # get features and fuse
    img_root = os.path.join(args.data_root, './%s/loose_crop' % dataset_name)
    landmark_list_path = os.path.join(args.data_root, './%s/meta/%s_name_5pts_score.txt' % (dataset_name, dataset_name.lower()))

    img_input_feats, faceness_scores, norms = infer_images_or_get_from_pickle(dataset_name, args.model_name,
                                    model, img_root, landmark_list_path, args )

    #k = kernel.kernel.detach().numpy()
    #img_input_feats = img_input_feats @ k

    print('Feature Shape: ({} , {}) .'.format(img_input_feats.shape[0], img_input_feats.shape[1]))

    # if args.fusion_method == 'pre_norm_vector_add':
    #     img_feats = img_feats * norms

    # run protocol
    #identification(args.data_root, dataset_name, img_input_feats, save_path)
    verification(args.data_root, dataset_name, img_input_feats, save_path, args.calc_avg_per_media, norms=norms)


########################################################################################################################


def validate_model_ijb(model, save_path, ijb_root, dataset_name='IJBB',model_name='ir101_webface4m',
                       args=None):
    #save_path = './result/{}/{}/{}'.format(args.dataset_name, args.model_name, args.experiment)
    #print('result save_path', save_path)
    os.makedirs(save_path, exist_ok=True)

    img_root = os.path.join(ijb_root, './%s/loose_crop' % dataset_name)
    landmark_list_path = os.path.join(ijb_root, './%s/meta/%s_name_5pts_score.txt' % (dataset_name, dataset_name.lower()))

    img_input_feats, faceness_scores, norms = infer_images_or_get_from_pickle(dataset_name, model_name,
                                    model, img_root, landmark_list_path, args )

    #img_feats = img_feats * norms
    # get features and fuse
    #img_feats = img_feats * norms
    # run protocol
    #identification(args.data_root, dataset_name, img_input_feats, save_path)
    scores = verification(ijb_root, dataset_name, img_input_feats, save_path, False, model, norms)
    return scores



