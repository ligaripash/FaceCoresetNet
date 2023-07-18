import numpy as np
import pickle
import pandas as pd
import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.metrics import roc_curve, auc
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import csv
from tqdm import tqdm
import torch

def read_template_media_list(path):
    #ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    #pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    #print(pairs.shape)
    #print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def angluar_dist_with_norm(norm, a, b):
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    dist = (1 - np.dot(norm_a, norm_b)) / 2
    gamma = 0.01
    dist = np.power(norm, gamma) * dist
    return dist


def angluar_dist(a, b):
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    dist = (1 - np.dot(norm_a, norm_b)) / 2
    return dist

def norm_dist(a, b):
    return np.abs(np.linalg.norm(a) - np.linalg.norm(b))



# def weighted_FPS(points, k, norms):
#     """
#     Algorithm 1: Semantics-guided Farthest Point Sampling Algorithm. N is the number of input points and M is the num-
#     ber of output points sampled by the algorithm.
#     Input: coordinates X = {x1, . . . , xN} ∈ RN×dim;
#     foreground scores P = {p1, . . . , pN} ∈ RN
#     Output: sampled key point set K = {k1, . . . , kM}
#     1: initialize an empty sampling point set K
#     2: initialize a distance array d of length N with all +∞
#     3: initialize a visit array v of length N with all zeros
#     4: for i = 1 to M do
#     5:
#     if i = 1 then
#     ki = arg max(P)
#     else
#     8: D = {pγk · dk|vk = 0}
#     9:
#     ki = arg max(D)
#     10: end if
#     11:
#     SA Layer
#     add ki to K, vki
#     = 1
#     12:
#     13:
#     for j = 1 to N do
#     dj = min(dj, xj − xki
#        14: end for
#     15: end for
#     16: return P
#     """
#     remaining_points = points[:].tolist()
#     norms = norms[:].tolist()
#     solution_set = []
#     max_index = np.argmax(norms)
#     solution_set.append(remaining_points.pop(max_index))
#     norms.pop(max_index)
#     for _ in range(k-1):
#         distances = np.Inf * len(remaining_points)
#         for i, p in enumerate(remaining_points):
#             norm = norms[i]
#             for j, s in enumerate(solution_set):
#                 distances[i] = min(distances[i], angluar_dist_with_norm(norm, p, s))
#         solution_set.append(remaining_points.pop(max_index))
#         norms.pop(max_index)
#     return np.array(solution_set)



def incremental_farthest_search_norms(points, k, norms):
    remaining_points = points[:].tolist()
    norms = norms[:].tolist()
    solution_set = []
    max_index = np.argmax(norms)
    solution_set.append(remaining_points.pop(max_index))
    norms.pop(max_index)
    for _ in range(k-1):
        #distances = [angluar_dist(p, solution_set[0]) for p in remaining_points]
        distances = [np.Inf] * len(remaining_points)
        for i, p in enumerate(remaining_points):
            norm = norms[i]
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], angluar_dist_with_norm(norm, p, s))
        max_index = distances.index(max(distances))
        solution_set.append(remaining_points.pop(max_index))
        norms.pop(max_index)
    return np.array(solution_set)



def incremental_farthest_search(points, k, norms):
    remaining_points = points[:].tolist()
    solution_set = []
    max_index = np.argmax(norms)
    solution_set.append(remaining_points.pop(max_index))
    for _ in range(k-1):
        distances = [angluar_dist(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], angluar_dist(p, s))
        solution_set.append(remaining_points.pop(distances.index(max(distances))))
    return np.array(solution_set)



def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def aggregate_dist_from_ud_center_priority(template_features, ud_center):

    ud_center_norm = ud_center / np.linalg.norm(ud_center)
    norms = np.linalg.norm(template_features, ord=2, axis=1, keepdims=False)
    norms = np.expand_dims(norms,axis=1)
    norm_features = template_features / norms

    e = (1-np.dot(norm_features, ud_center_norm))
    e_sum = e.sum()
    w = np.multiply(e, e)
    weighted_features = (w * norm_features.transpose(1,0)).transpose(1,0)
    weighted_features = weighted_features.sum(axis=0) / e_sum
    return weighted_features

def aggregate_fps_with_dist_from_ud_center_priority(template_features, ud_center):
    """
    Combine FPS with norm priority.
    Merge the norm size with farthest point in one metric.
    """

    K = 5

    number_features = template_features.shape[0]
    actual_K = min(K, number_features)

    # if number_features < actual_K:
    #     indices = rng.choice(template_features.shape[0], K_fps, replace=True)
    #     template_features = template_features[indices]

    ud_center_norm = ud_center / np.linalg.norm(ud_center)
    norms = np.linalg.norm(template_features, ord=2, axis=1, keepdims=False)
    norms = np.expand_dims(norms,axis=1)
    norm_features = template_features / norms
    semantic_importance = (1-np.dot(norm_features, ud_center_norm)) / 2
    semantic_importance = softmax(semantic_importance)

    FPS_sample = incremental_farthest_search_norms(template_features, actual_K, semantic_importance)
    template_features = FPS_sample

    agg_max = np.max(template_features, axis=0)
    agg_min = np.min(template_features, axis=0)
    agg_abs_max = np.abs(agg_max)
    agg_abs_min = np.abs(agg_min)
    agg = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)
    agg = np.expand_dims(agg, 0)
    return agg

def aggregate_with_abs_pool(template_features, norms, rng):
    #avg = np.average(template_features, axis=0)
    #std = np.std(template_features, axis=0)
    agg_max = np.max(template_features, axis=0)
    agg_min = np.min(template_features, axis=0)
    agg_abs_max = np.abs(agg_max)
    agg_abs_min = np.abs(agg_min)
    agg = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)


    return agg


def aggregate_fps_with_norm_priority(template_features, norms, rng):
    """
    Combine FPS with norm priority.
    Merge the norm size with farthest point in one metric.
    """
    K = 5

    number_features = template_features.shape[0]
    #actual_K = min(K, number_features)
    indexes = []
    t = []
    actual_K = K
    import random
    if number_features < K:
        for i in range(K):
            indexes.append(random.choice(range(number_features)))

        template_features = np.array([template_features[i] for i in indexes])
        norms = np.array([norms[i] for i in indexes])
        #number_to_append = K - number_features
        #features_to_append = template_features[0:number_to_append, :]
        #template_features = np.concatenate((template_features, features_to_append), axis=0)
        #norms_to_append = norms[0:number_to_append]
        #template_norms = np.concatenate((norms, norms_to_append), axis=0)
        #norms = template_norms

    # if number_features < actual_K:
    #     indices = rng.choice(template_features.shape[0], K_fps, replace=True)
    #     template_features = template_features[indices]


    #norms = np.linalg.norm(template_features, ord=2, axis=1, keepdims=False)
    sm_norms = softmax(norms)
    template_features = template_features * norms
    FPS_sample = incremental_farthest_search_norms(template_features, actual_K, sm_norms)
    template_features = FPS_sample

    agg_max = np.max(template_features, axis=0)
    agg_min = np.min(template_features, axis=0)
    agg_abs_max = np.abs(agg_max)
    agg_abs_min = np.abs(agg_min)
    agg = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)

    return FPS_sample




def aggregate(template_features, method='FPS'):
    """
    First do farthest point sampling (FPS) and sample fps_sample fraction of the template.
    Select the top norm_sample fraction out of the remaining template and sum.
    Args:
        template_features:

    Returns:

    """

    K_fps = 10
    K_largest_sample = 5

    number_features = template_features.shape[0]
    # if number_features < K_fps:
    #     indices = np.random.choice(template_features.shape[0], K_fps, replace=True)
    #     template_features = template_features[indices]

    if number_features < K_fps:
        agg_max = np.max(template_features, axis=0)
        agg_min = np.min(template_features, axis=0)
        agg_abs_max = np.abs(agg_max)
        agg_abs_min = np.abs(agg_min)
        agg = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)
        #agg = np.expand_dims(agg, 0)
        return agg



    norms = np.linalg.norm(template_features, ord=2, axis=1, keepdims=False)
    FPS_sample = incremental_farthest_search(template_features, K_fps, norms)
    norms2 = np.linalg.norm(FPS_sample, ord=2, axis=1, keepdims=False)
    indexes = np.argsort(norms2)
    last = FPS_sample[indexes[-K_largest_sample:]]
    template_features = last

    if method=='MAX_POOL':
        agg_max = np.max(template_features, axis=0)
        agg_min = np.min(template_features, axis=0)
        agg_abs_max = np.abs(agg_max)
        agg_abs_min = np.abs(agg_min)
        agg = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)
        #agg = np.expand_dims(agg, 0)
        return agg

    return last


def sigmoid(x):
      return 1 / (1 + np.exp(-x))

def image2template_feature(img_feats=None, templates=None, medias=None, calc_avg_per_media=False, aggregation=True, model=None, norms=None):
    # gil - calculating the average feature per media is cheating
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    # template_features_file = '/home/gilsh/template_features_new.pickle'
    # if os.path.isfile(template_features_file):
    #     with open(template_features_file, 'rb') as f:
    #         template_features = pickle.load(f)
    #     template_norms = (np.linalg.norm(template_features, axis=-1))
    #     avg_norm = np.average(template_norms)
    #     std_norm = np.std(template_norms)
    #     print('Average norm = {}, std = {}'.format(avg_norm, std_norm))
    #     #template_features = sklearn.preprocessing.normalize(template_features)
    #     return  template_features, unique_templates



    K = 5
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    #template_feats = np.zeros((len(unique_templates), K, img_feats.shape[1]))
    rng = np.random.default_rng(2021)

    # template_sizes = [np.where(templates == x)[0].size for x in unique_templates]
    # unique_template_size = np.unique(template_sizes)
    # for size in unique_template_size:
    #     template_indexes_for_size = np.where(template_sizes==size)[0]
    template_norms = []
    # with open('/home/gilsh/std_low_detectability_ir101_ms1mv2.pickle', 'rb') as f:
    #     ud_center = pickle.load(f)
    for count_template, uqt in enumerate(tqdm(unique_templates, position=0, leave=True)):
        (ind_t, ) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        if norms is not None:
            face_norms = norms[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []

        if calc_avg_per_media:
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m, ) = np.where(face_medias == u)
                # max_pooled_feats = aggregate(face_norm_feats[ind_m], method='MAX_POOL')
                # max_pooled_feats = np.squeeze(max_pooled_feats)
                #media_norm_feats += [aggregate_dist_from_ud_center_priority(face_norm_feats[ind_m], ud_center)]
                #media_norm_feats += [aggregate(face_norm_feats[ind_m], method='MAX_POOL')]
                media_norm_feats += [aggregate_fps_with_norm_priority(face_norm_feats[ind_m], rng)]

                if 0:
                    if ct == 1:
                        media_norm_feats += [face_norm_feats[ind_m]]
                        # media_norm_feats += [max_pooled_feats]
                    else:  # image features from the same video will be aggregated into one feature
                        media_norm_feats += [
                            np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                        ]
                    # media_norm_feats += [
                    #     np.mean(max_pooled_feats, axis=0, keepdims=True)
                    # ]

                    # media_norm_feats += [
                    #     np.sum(face_norm_feats[ind_m], axis=0, keepdims=True)
                    # ]
            media_norm_feats = np.array(media_norm_feats)
            #media_norm_feats = aggregate_dist_from_ud_center_priority(media_norm_feats, ud_center)
            #media_norm_feats = aggregate(media_norm_feats,method='MAX_POOL')

            if model==None:
                media_norm_feats = aggregate_fps_with_norm_priority(face_norm_feats, face_norms, rng)
            else:
                media_norm_feats = model(embeddings=face_norm_feats, norms=norms)
            #media_norm_feats = np.expand_dims(media_norm_feats, axis=0)
            # if len(media_norm_feats.shape) == 3:
            #     media_norm_feats.squeeze()

        elif aggregation:
            if model==None:
                #media_norm_feats = aggregate_fps_with_norm_priority(face_norm_feats, face_norms, rng)
                unnorm_feats = face_norms * face_norm_feats
                #media_norm_feats = aggregate_with_abs_pool(face_norm_feats, face_norms, rng)
                #media_norm_feats = np.sum(face_norm_feats,axis=0)
                #unnorm_feats = face_norms * face_norm_feats
                media_norm_feats = np.mean(unnorm_feats, axis=0)
                pass
            else:
                t_face_norm_feats = torch.Tensor(face_norm_feats).unsqueeze(0).to('cuda:0')
                face_norms = torch.Tensor(face_norms).unsqueeze(0).to('cuda:0')
                aggregate_embeddings, aggregate_norms, FPS_sample = model(embeddings=t_face_norm_feats, norms=face_norms)
                media_norm_feats = aggregate_embeddings * aggregate_norms
                #media_norm_feats = FPS_sample

            #media_norm_feats = aggregate(face_norm_feats, method='MAX_POOL')
            #media_norm_feats = aggregate_fps_with_norm_priority(face_norm_feats, rng)
            #media_norm_feats = aggregate_fps_with_dist_from_ud_center_priority(face_norm_feats, ud_center)

        else:
            media_norm_feats = face_norm_feats
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))

        if model==None:
            #template_feats[count_template] = np.sum(media_norm_feats, axis=0)
            template_feats[count_template] = media_norm_feats
        else:
            template_feats[count_template] = media_norm_feats.detach().to('cpu')
        #media_norm_feats is supposed to be (K, 512) dim.
        #template_feats[count_template] = media_norm_feats
        #gil - try to average instead of sum
        #template_feats[count_template] = np.average(media_norm_feats, axis=0)

#        if count_template % 2000 == 0:
#            print('Finish Calculating {} template features.'.format(
#                count_template))
    #template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    #gil - don't normalize - use the norm as as a quality descriptor
    #template_norm_feats = template_feats
    # with open(template_features_file, 'wb') as f:
    #     pickle.dump(template_feats, f)

    #template_norms = (np.linalg.norm(template_feats, axis=-1))
    #avg_norm = np.average(template_norms)
    #std_norm = np.std(template_norms)
    #print('Average norm = {}, std = {}'.format(avg_norm, std_norm))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # norms = np.linalg.norm(template_feats, axis=1)
    # sig_norms = norms / norms.max()
#    template_norm_feats = (template_norm_feats.transpose() * sig_norms).transpose()

    #avg_norm_feature = np.average(template_norm_feats, axis=0)
    #template_norm_feats = template_norm_feats - avg_norm_feature

    return template_norm_feats, unique_templates

def verification_quality_by_norm_metric(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    a=-3
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]


        #gil
        #similarity_score = np.matmul(feat1, feat2).max()
        n1 = np.expand_dims(np.linalg.norm(feat1, axis=-1), axis=-1)
        n2 = np.expand_dims(np.linalg.norm(feat2, axis=-1), axis=-1)

        # q1 = np.where(n1 < 40,0,1)
        # q2 = np.where(n2 < 40,0,1)

        #q1 = 1 - np.exp(a*n1)
        #q2 = 1 - np.exp(a*n2)
        q1 = np.where(n1 < 1, 0, 1)
        q2 = np.where(n2 < 1, 0, 1)
        #mm = np.minimum(q1, q2)

        #mm = np.squeeze(mm, axis=-1)
        q1 = np.squeeze(q1, axis=-1)
        q2 = np.squeeze(q2, axis=-1)

        feat1_norm = feat1 / n1
        feat2_norm = feat2 / n2

        similarity_score = np.sum(feat1_norm * feat2_norm, -1)
        similarity_score = (similarity_score+1) / 2 * q1 * q2
        #similarity_score = (similarity_score + 1) / 2
        #similarity_score = similarity_score * mm
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    a=-0.1
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]


        #gil
        #similarity_score = np.matmul(feat1, feat2).max()
        n1 = np.expand_dims(np.linalg.norm(feat1, axis=-1), axis=-1)
        n2 = np.expand_dims(np.linalg.norm(feat2, axis=-1), axis=-1)

        #q1 = 1 - np.exp(a*n1)
        #q2 = 1 - np.exp(a * n2)

        #q1 = np.squeeze(q1, axis=-1)
        #q2 = np.squeeze(q2, axis=-1)

        feat1_norm = feat1 / n1
        feat2_norm = feat2 / n2

        agg_max = np.max(feat1_norm, axis=2)
        agg_min = np.min(feat1_norm, axis=2)
        agg_abs_max = np.abs(agg_max)
        agg_abs_min = np.abs(agg_min)
        agg1 = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)

        agg_max = np.max(feat2_norm, axis=2)
        agg_min = np.min(feat2_norm, axis=2)
        agg_abs_max = np.abs(agg_max)
        agg_abs_min = np.abs(agg_min)
        agg2 = np.where(agg_abs_max > agg_abs_min, agg_max, agg_min)


        # b = feat2_norm.transpose(0, 1, 3, 2)
        # all_sim = feat1_norm @ b
        # sim_val1 = all_sim.max(axis=-1).min(axis=-1)
        # sim_val2 = all_sim.max(axis=-2).min(axis=-1)
        #sim_val = all_sim.max(axis=(2,3))
        #similarity_score = np.minimum(sim_val1, sim_val2)
        #similarity_score = np.sum(feat1_norm * feat2_norm, -1)
        #similarity_score = similarity_score * q1 * q
        agg1 = sklearn.preprocessing.normalize(agg1.squeeze())
        agg2 = sklearn.preprocessing.normalize(agg2.squeeze())
        similarity_score = np.sum(agg1 * agg2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1), ))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))

    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

def write_result(result_files, save_path, dataset_name, label):

    methods = []
    scores = []
    for file in result_files:
        methods.append(Path(file).parent.stem)
        scores.append(np.load(file))

    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    colours = dict(
        zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    #x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        plt.plot(fpr,
                tpr,
                color=colours[method],
                lw=1,
                label=('[%s (AUC = %0.4f %%)]' %
                        (method.split('-')[-1], roc_auc * 100)))
        tpr_fpr_row = []
        tpr_fpr_row.append("%s-%s" % (method, dataset_name))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
                list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            #tpr_fpr_row.append('%.4f' % tpr[min_index])
            tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10**-6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB')
    plt.legend(loc="lower right")
    #plt.show()
    fig.savefig(os.path.join(save_path, 'verification_auc.pdf'))
    print(tpr_fpr_table)

    # write to csv
    result = [tuple(filter(None, map(str.strip, splitline))) for line in str(tpr_fpr_table).splitlines()
                                                             for splitline in [line.split("|")] if len(splitline) > 1]
    with open(os.path.join(save_path, 'verification_result_no_media.csv'), 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)

    out_table = [float(x) for x in tpr_fpr_row[-6:]]
    return out_table