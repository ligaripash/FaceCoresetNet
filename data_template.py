import os
import random

import torchvision.datasets as datasets

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd


class TemplateBatch(object):
    def __init__(self, max_template_size, mode='train'):
        self.max_template_size = max_template_size
        self.mode = mode
        self.random_crop = transforms.RandomCrop(112,padding=0)
        self.gaussian_noise = transforms.GaussianBlur(kernel_size=(7,13), sigma=(7,10))
        self.aug_mix = transforms.RandAugment()
        #self.perspective_transform = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transform_val = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])

    def create_clip(self, anchor_image, clip_size, do_aug_mix=True):
        clip = []
        if self.mode == 'train':
            t = self.transform
        else:
            t = self.transform_val

        frame = anchor_image
        for i in range(clip_size):
            #gil
            if do_aug_mix:
                frame = self.aug_mix(anchor_image)
            #frame = anchor_image
            frame = t(frame)
            frame = torch.unsqueeze(frame, dim=0)
            clip.append(frame)

        clip = torch.cat(clip, dim=0)
        return clip


    def add_noise(self, sample):
        #gil - no noise
        if random.random() > 1.0:
            sample = self.gaussian_noise(sample)
        return sample


    # def create_fixed_size_template_val(self, image, output_template_size):
    #     """
    #     Create a fixed size template out of a variable size template
    #     Args:
    #         template_images: variable size template
    #
    #     Returns:
    #         fixed size template
    #
    #     Simulate an IJBB style template. The template should contain few 'video clips'. Each clip is created from a
    #     separate translated image.
    #     When creating a template image, noise should be added to the template to make it challenging for the model to
    #     recognize them.
    #
    #     The algorithm:
    #
    #     1. Assume N is the required template size
    #     2. Sample an image from the template, create a 'clip' out of the image. The size of the clip is kN k in [0,1]
    #     3. Sample (1-k)N images add noise to the image
    #     """
    #     fixed_size_template = []
    #     current_template_size = 0
    #     CREATE_CLIP = False
    #     while current_template_size < output_template_size:
    #         sample_image = image
    #         if CREATE_CLIP:
    #             clip_size = random.randint(1, output_template_size - current_template_size)
    #             current_template_size += clip_size
    #             clip = self.create_clip(sample_image, clip_size)
    #         else:
    #             clip = sample_image
    #         fixed_size_template.append(clip)
    #
    #     template_tensor = torch.cat(fixed_size_template, dim=0)
    #     return template_tensor



    def create_fixed_size_template(self, template_images, output_template_size):
        """
        Create a fixed size template out of a variable size template
        Args:
            template_images: variable size template

        Returns:
            fixed size template

        Simulate an IJBB style template. The template should contain few 'video clips'. Each clip is created from a
        separate translated image.
        When creating a template image, noise should be added to the template to make it challenging for the model to
        recognize them.

        The algorithm:

        1. Assume N is the required template size
        2. Sample an image from the template, create a 'clip' out of the image. The size of the clip is kN k in [0,1]
        3. Sample (1-k)N images add noise to the image
        """
        fixed_size_template = []
        current_template_size = 0
        max_clip_size = 3
        while current_template_size < output_template_size:
            sample_image = random.choice(template_images)
            #gil - try without noise
            sample_image = self.add_noise(sample_image)
            clip_size_rand = random.randint(1, output_template_size - current_template_size)
            clip_size = min(max_clip_size, clip_size_rand)
            current_template_size += clip_size
            clip = self.create_clip(sample_image, clip_size, do_aug_mix=True)

            fixed_size_template.append(clip)

        template_tensor = torch.cat(fixed_size_template, dim=0)

        return template_tensor



    def __call__(self, batch):

        template_size = random.randint(1, self.max_template_size)
        templates = []
        targets = []
        assert self.mode == 'train'
        if self.mode == 'train':
            for template_images, target in batch:
                fixed_size_template = self.create_fixed_size_template(template_images, template_size)
                #fixed_size_template should have the shape (template_size, 3, H, W)
                fixed_size_template = torch.unsqueeze(fixed_size_template, dim=0)
                templates.append(fixed_size_template)
                targets.append(target)
        # else:
        #     for template_images, target, dataname, index in batch:
        #         fixed_size_template = self.create_fixed_size_template_val(template_images, template_size)
        #         #fixed_size_template should have the shape (template_size, 3, H, W)
        #         fixed_size_template = torch.unsqueeze(fixed_size_template, dim=0)
        #         templates.append(fixed_size_template)
        #         targets.append(target)

        templates = torch.cat(templates, dim=0)
        targets = torch.tensor(targets)
        out_batch = (templates, targets)
        #list(zip(*out_batch))
        #out_batch should have the shape ( N, T, 3, H, W)
        # out_batch[0] = torch.stack(out_batch[0])
        # out_batch[0] = nested_tensor_from_tensor_list(out_batch[0])
        return out_batch



        out_batch = list(zip(*batch))
        out_batch[0] = torch.stack(out_batch[0])
        return tuple(out_batch)


class DataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.max_template_size=kwargs['max_template_size']
        self.data_root = kwargs['data_root']
        self.train_data_path = kwargs['train_data_path']
        self.val_data_path = kwargs['val_data_path']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.train_data_subset = kwargs['train_data_subset']

        self.low_res_augmentation_prob = kwargs['low_res_augmentation_prob']
        self.crop_augmentation_prob = kwargs['crop_augmentation_prob']
        self.photometric_augmentation_prob = kwargs['photometric_augmentation_prob']

        concat_mem_file_name = os.path.join(self.data_root, self.val_data_path, 'concat_validation_memfile')
        self.concat_mem_file_name = concat_mem_file_name


    # def prepare_data(self):
    #     # call this once to convert val_data to memfile for saving memory
    #     if not os.path.isdir(os.path.join(self.data_root, self.val_data_path, 'agedb_30', 'memfile')):
    #         print('making validation data memfile')
    #         evaluate_utils.get_val_data(os.path.join(self.data_root, self.val_data_path))
    #
    #     if not os.path.isfile(self.concat_mem_file_name):
    #         # create a concat memfile
    #         concat = []
    #         for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
    #             np_array, issame = evaluate_utils.get_val_pair(path=os.path.join(self.data_root, self.val_data_path),
    #                                                            name=key,
    #                                                            use_memfile=False)
    #             concat.append(np_array)
    #         concat = np.concatenate(concat)
    #         evaluate_utils.make_memmap(self.concat_mem_file_name, concat)


    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            print('creating train dataset')
            self.train_dataset = train_dataset(self.data_root,
                                               self.train_data_path,
                                               self.low_res_augmentation_prob,
                                               self.crop_augmentation_prob,
                                               self.photometric_augmentation_prob,
                                               )

            # checking same list for subseting
            if self.train_data_path == 'faces_emore/imgs' and self.train_data_subset:
                with open('ms1mv2_train_subset_index.txt', 'r') as f:
                    subset_index = [int(i) for i in f.read().split(',')]

                # remove too few example identites
                self.train_dataset.samples = [self.train_dataset.samples[idx] for idx in subset_index]
                self.train_dataset.targets = [self.train_dataset.targets[idx] for idx in subset_index]
                value_counts = pd.Series(self.train_dataset.targets).value_counts()
                to_erase_label = value_counts[value_counts<5].index
                e_idx = [i in to_erase_label for i in self.train_dataset.targets]
                self.train_dataset.samples = [i for i, erase in zip(self.train_dataset.samples, e_idx) if not erase]
                self.train_dataset.targets = [i for i, erase in zip(self.train_dataset.targets, e_idx) if not erase]

                # label adjust
                max_label = np.max(self.train_dataset.targets)
                adjuster = {}
                new = 0
                for orig in range(max_label+1):
                    if orig in to_erase_label:
                        continue
                    adjuster[orig] = new
                    new += 1

                # readjust class_to_idx
                self.train_dataset.targets = [adjuster[orig] for orig in self.train_dataset.targets]
                self.train_dataset.samples = [(sample[0], adjuster[sample[1]]) for sample in self.train_dataset.samples]
                new_class_to_idx = {}
                for label_str, label_int in self.train_dataset.class_to_idx.items():
                    if label_int in to_erase_label:
                        continue
                    else:
                        new_class_to_idx[label_str] = adjuster[label_int]
                self.train_dataset.class_to_idx = new_class_to_idx

            print('creating val dataset')

#            self.val_dataset = val_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

        # Assign Test split(s) for use in Dataloaders
        # if stage == 'test' or stage is None:
        #     self.test_dataset = test_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=TemplateBatch(self.max_template_size, 'train'))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=TemplateBatch(self.max_template_size, 'val'))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=TemplateBatch(self.max_template_size))


def train_dataset(data_root, train_data_path,
                  low_res_augmentation_prob,
                  crop_augmentation_prob,
                  photometric_augmentation_prob):

    train_dir = os.path.join(data_root, train_data_path)
    train_dataset = CustomImageFolderDataset(root=train_dir,
                                             transform=transforms.Compose([
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                             ]),
                                             low_res_augmentation_prob=low_res_augmentation_prob,
                                             crop_augmentation_prob=crop_augmentation_prob,
                                             photometric_augmentation_prob=photometric_augmentation_prob,
                                             )

    return train_dataset


# def val_dataset(data_root, val_data_path, concat_mem_file_name):
#     val_data = evaluate_utils.get_val_data(os.path.join(data_root, val_data_path))
#     # theses datasets are already normalized with mean 0.5, std 0.5
#     age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
#     val_data_dict = {
#         'agedb_30': (age_30, age_30_issame),
#         "cfp_fp": (cfp_fp, cfp_fp_issame),
#         "lfw": (lfw, lfw_issame),
#         "cplfw": (cplfw, cplfw_issame),
#         "calfw": (calfw, calfw_issame),
#     }
#     val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
#
#     return val_dataset


# def test_dataset(data_root, val_data_path, concat_mem_file_name):
#     val_data = evaluate_utils.get_val_data(os.path.join(data_root, val_data_path))
#     # theses datasets are already normalized with mean 0.5, std 0.5
#     age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
#     val_data_dict = {
#         'agedb_30': (age_30, age_30_issame),
#         "cfp_fp": (cfp_fp, cfp_fp_issame),
#         "lfw": (lfw, lfw_issame),
#         "cplfw": (cplfw, cplfw_issame),
#         "calfw": (calfw, calfw_issame),
#     }
#     val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
#     return val_dataset


class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 class_num=10000
                 ):
        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.class_num = class_num
        self.low_res_augmentation_prob = low_res_augmentation_prob
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

        self.tot_rot_try = 0
        self.rot_success = 0

    # def __len__(self):
    #     return self.class_num
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        #Gil
        person_id = target
        template = [x for x in self.samples if x[1] == person_id]
        template_samples = []
        for template_sample in template:
            path, _ = template_sample
            template_image = self.loader(path)

            if 'WebFace' in self.root or 'webface' in self.root:
                # swap rgb to bgr since image is in rgb for webface
                template_image = Image.fromarray(np.asarray(template_image)[:, :, ::-1])

            template_image, _ = self.augment(template_image)
#            gil - remove remark after debug
#             if self.transform is not None:
#                 template_image = self.transform(template_image)
            if self.target_transform is not None:
                template_image = self.target_transform(template_image)

            template_samples.append(template_image)

        #template_samples = self.enchance_template(template)

        return template_samples, person_id


        return sample, target

    def augment(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            new = np.zeros_like(np.array(sample))
            #orig_W, orig_H = F._get_image_size(sample)
            orig_W, orig_H = sample.size
            i, j, h, w = self.random_resized_crop.get_params(sample,
                                                            self.random_resized_crop.scale,
                                                            self.random_resized_crop.ratio)
            cropped = F.crop(sample, i, j, h, w)
            new[i:i+h,j:j+w, :] = np.array(cropped)
            sample = Image.fromarray(new.astype(np.uint8))
            crop_ratio = min(h, w) / max(orig_H, orig_W)
        else:
            crop_ratio = 1.0

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))
        else:
            resize_ratio = 1

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                                  self.photometric.saturation, self.photometric.hue)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    sample = F.adjust_brightness(sample, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    sample = F.adjust_contrast(sample, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    sample = F.adjust_saturation(sample, saturation_factor)

        information_score = resize_ratio * crop_ratio
        return sample, information_score


class FiveValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name):
        '''
        concatenates all validation datasets from emore
        val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
        }
        agedb_30: 0
        cfp_fp: 1
        lfw: 2
        cplfw: 3
        calfw: 4
        '''
        self.dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}

        self.val_data_dict = val_data_dict
        # concat all dataset
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
            for same in issame:
                dup_issame.append(same)
                dup_issame.append(same)
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            key_orders.append(key)
        assert key_orders == ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']

        if isinstance(all_imgs[0], np.memmap):
            self.all_imgs = evaluate_utils.read_memmap(concat_mem_file_name)
        else:
            self.all_imgs = np.concatenate(all_imgs)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

        assert len(self.all_imgs) == len(self.all_issame)
        assert len(self.all_issame) == len(self.all_dataname)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = torch.tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)


def low_res_augmentation(img):
    # resize the image to a small size and enlarge it back
    img_shape = img.shape
    side_ratio = np.random.uniform(0.2, 1.0)
    small_side = int(side_ratio * img_shape[0])
    interpolation = np.random.choice(
        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
    interpolation = np.random.choice(
        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

    return aug_img, side_ratio