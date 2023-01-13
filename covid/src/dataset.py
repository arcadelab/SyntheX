# Dataloading utilities from preprocessed HDF5 files.
import math
import random

import h5py as h5

import torch
import torch.utils.data

import torchvision.transforms.functional as TF

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import PIL
import PIL.Image as Image
import cv2

import matplotlib.pyplot as plt

from util import *

def calc_pad_amount(padded_img_dim, cur_img_dim):
    # new pad dimension should be larger
    assert(padded_img_dim > cur_img_dim)

    # first calculate the amount to pad along the borders
    pad = (padded_img_dim - cur_img_dim)/ 2

    # handle odd sized input
    if pad != int(pad):
        pad = int(pad) + 1
    else:
        # needs to be integral
        pad = int(pad)

    return pad

class COVIDRandomDataAugDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, data_num, proj_pad_dim=0, ds_factor=8, num_classes=2, data_aug=False, heavy_aug=False, valid=False):
        self.data_path = data_path
        self.data_num = data_num
        self.ds_factor = ds_factor
        self.num_classes = num_classes
        self.valid = valid

        crop_size = 160
        self.randomcrop_output_size = [crop_size, crop_size]
        self.prob_of_aug = 0.5 if data_aug else 0.0
        self.do_heavy_aug = True if heavy_aug else False

        self.do_invert = True
        self.do_gamma  = True
        self.do_noise  = True
        self.do_affine = True
        self.do_erase  = True

        self.erase_prob = 0.25

        self.pad_data_for_affine = True

        self.do_norm_01_scale = True

        self.print_aug_info = False

        self.proj_pad_dim = proj_pad_dim
        self.extra_pad = 0
        self.need_to_pad_proj = False

        self.debug_label = False

    def __len__(self):
        return self.data_num

    def _invert_fun(self, p,s):
        #print('invert fun')
        p_max = p.max()
        #p_min = p.min()
        p = p_max - p

        return (p,s)

    def _noise_fun(self, p,s):
        #print('noise fun')
        # normalize to [0,1] to apply noise
        p_min = p.min()
        p_max = p.max()

        p = (p - p_min) / (p_max - p_min)

        cur_noise_sigma = random.uniform(0.005, 0.01)
        p += torch.randn(p.shape) * cur_noise_sigma

        p = (p * (p_max - p_min)) + p_min

        return (p,s)

    def _gamma_fun(self, p,s):
        #print('gamma fun')
        # normalize to [0,1] to apply gamma
        p_min = p.min()
        p_max = p.max()

        p = (p - p_min) / (p_max - p_min)

        gamma = random.uniform(0.7,1.3)
        p.pow_(gamma)

        p = (p * (p_max - p_min)) + p_min

        return (p,s)

    def _affine_fun(self, p,s):
        #print('affine fun')
        # data needs to be in [0,1] for PIL functions
        p_min = p.min()
        p_max = p.max()

        p = (p - p_min) / (p_max - p_min)

        orig_p_shape = p.shape
        self.need_to_pad_proj = self.extra_pad > 0
        if self.pad_data_for_affine:
            pad1 = int(math.ceil(orig_p_shape[1] / 2.0))
            pad2 = int(math.ceil(orig_p_shape[2] / 2.0))
            if self.need_to_pad_proj:
                pad1 += self.extra_pad
                pad2 += self.extra_pad
                self.need_to_pad_proj = False

            p = torch.from_numpy(np.pad(p.numpy(),
                                        ((0,0), (pad1,pad1), (pad2,pad2)),
                                        'reflect'))

        p_il = TF.to_pil_image(p)

        # this uniformly samples the direction
        rand_trans = torch.randn(2)
        rand_trans /= rand_trans.norm()

        # now uniformly sample the magnitdue
        rand_trans *= random.random() * 20

        rot_ang = random.uniform(-5, 5)
        trans_x = rand_trans[0]
        trans_y = rand_trans[1]
        shear   = random.uniform(-2, 2)

        scale_factor = random.uniform(0.9, 1.1)

        if self.print_aug_info:
            print('Rot: {:.2f}'.format(rot_ang))
            print('Trans X: {:.2f} , Trans Y: {:.2f}'.format(trans_x, trans_y))
            print('Shear: {:.2f}'.format(shear))
            print('Scale: {:.2f}'.format(scale_factor))

        p = TF.to_tensor(TF.affine(TF.to_pil_image(p),
                         rot_ang,
                         (trans_x, trans_y),
                         scale_factor,
                         shear,
                         resample=PIL.Image.BILINEAR))

        if self.pad_data_for_affine:
            # pad can be zero
            pad_shape = (orig_p_shape[-2] + (2 * self.extra_pad), orig_p_shape[-1] + (2 * self.extra_pad))
            p = center_crop(p, pad_shape)

        p = (p * (p_max - p_min)) + p_min

        if s is not None:
            orig_s_shape = s.shape
            if self.pad_data_for_affine:
                pad1 = int(math.ceil(orig_s_shape[1] / 2.0))
                pad2 = int(math.ceil(orig_s_shape[2] / 2.0))
                s = torch.from_numpy(np.pad(s.numpy(),
                                            ((0,0), (pad1,pad1), (pad2,pad2)),
                                            'reflect'))

            # warp each class separately, I don't want any wacky color
            # spaces assumed by PIL
            for c in range(s.shape[0]):
                s[c,:,:] = TF.to_tensor(TF.affine(TF.to_pil_image(s[c,:,:]),
                                                  rot_ang,
                                                  (trans_x, trans_y),
                                                  scale_factor,
                                                  shear))
            if self.pad_data_for_affine:
                s = center_crop(s, orig_s_shape)

        return (p,s)

    def _erase_fun(self, p,s):
        #print('erase fun')
        p_2d_shape = [p.shape[-2], p.shape[-1]]
        box_mean_dim = torch.Tensor([p_2d_shape[0] * 0.15, p_2d_shape[1] * 0.15])

        num_boxes = random.randint(1,5)

        if self.print_aug_info:
            print('  Random Corrupt: num. boxes: {}'.format(num_boxes))

        for box_idx in range(num_boxes):
            box_valid = False

            while not box_valid:
                # First sample box dims
                box_dims = torch.round((torch.randn(2) * (box_mean_dim)) + box_mean_dim).long()

                if (box_dims[0] > 0) and (box_dims[1] > 0) and \
                        (box_dims[0] <= p_2d_shape[0]) and (box_dims[1] <= p_2d_shape[1]):
                    # Next sample box location
                    start_row = random.randint(0, p_2d_shape[0] - box_dims[0])
                    start_col = random.randint(0, p_2d_shape[1] - box_dims[1])

                    box_valid = True

            p_roi = p[0,start_row:(start_row+box_dims[0]),start_col:(start_col+box_dims[1])]

            sigma_noise = (p_roi.max() - p_roi.min()) * 0.2

            p_roi += torch.randn(p_roi.shape) * sigma_noise

        return (p,s)

    def _heavy_fun(self, p,s):
        #print('heavy fun')
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        p_array = p.numpy()

        #max_val = np.max(p_array)
        median_val = np.median(p_array)
        scale_val = 0.1 * median_val

        seq = iaa.Sequential([
                #
                # Execute 0 to 2 of the following (less important) augmenters per
                # image. Don't execute all of them, as that would often be way too
                # strong.
                #

                iaa.SomeOf((1, 2),
                [
                    # Blur
                    sometimes(
                    iaa.OneOf([
                        # Blur each image with varying strength using
                        # gaussian blur (sigma between 0 and 3.0),
                        # average/uniform blur (kernel size between 2x2 and 7x7)
                        # median blur (kernel size between 3x3 and 11x11).
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(4, 6)),
                    ])
                    ),

                    # Additive Noise Injection
                    # sometimes(
                    # iaa.OneOf([
                    #     iaa.AdditiveLaplaceNoise(scale=(0, scale_val)),
                    #     iaa.AdditivePoissonNoise(scale_val),
                    # ])
                    # ),

                    # Dropout
                    sometimes(
                    iaa.OneOf([
                        # Either drop randomly 1 to 10% of all pixels (i.e. set
                        # them to black) or drop them on an image with 2-5% percent
                        # of the original size, leading to large dropped
                        # rectangles.
                        iaa.Dropout((0.1, 0.15)),
                        iaa.CoarseDropout(
                            (0.1, 0.15), size_percent=(0.1, 0.15)
                        ),
                    ])
                    ),

                    # Convolutional
                    sometimes(
                    iaa.OneOf([
                        # Sharpen each image, overlay the result with the original
                        # image using an alpha between 0 (no sharpening) and 1
                        # (full sharpening effect).
                        iaa.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.5)),

                        # Same as sharpen, but for an embossing effect.
                        iaa.Emboss(alpha=(0.5, 1.0), strength=(1.0, 2.0))
                    ])
                    ),

                    # Pooling
                    sometimes(
                    iaa.OneOf([
                        iaa.AveragePooling([2, 4]),
                        iaa.MaxPooling([2, 4]),
                        iaa.MinPooling([2, 4]),
                        iaa.MedianPooling([2, 4]),
                        ])
                    ),

                    # Multiply
                    sometimes(
                        iaa.OneOf([
                            # Change brightness of images (50-150% of original value).
                            iaa.Multiply((0.5, 1.5)),
                            iaa.MultiplyElementwise((0.5, 1.5))
                        ])
                    ),

                    # Replace 10% of all pixels with either the value 0 or max_val
                    # sometimes(
                    #     iaa.ReplaceElementwise(0.1, [0, scale_val])
                    # ),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    )
                ],
                # do all of the above augmentations in random order
                random_order=True
                )
            ],
            # do all of the above augmentations in random order
            random_order=True)

        p = seq(images=p_array)

        p = torch.from_numpy(p)

        return (p,s)

    def _peppersalt_fun(self, p,s):
        #print('peppersalt fun')
        p_array = p.numpy()
        # Perform Pepper/Salt noise injection
        saltpepper_seq = iaa.Sequential([
            iaa.OneOf([
                iaa.ImpulseNoise(0.1),
                iaa.SaltAndPepper(0.1),
                iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
                ])
            ])

        p_min = p_array.min()
        p_max = p_array.max()

        p_array = 255*(p_array - p_min) / (p_max - p_min)
        p_array = saltpepper_seq(images=p_array)
        p_array = (p_array * (p_max - p_min)/255) + p_min
        p = torch.from_numpy(p_array)

        return (p,s)


    def _contrast_fun(self, p,s):
        #print('contrast fun')
        # Perform contrast randomization
        p_array = p.numpy()
        contrast_seq = iaa.Sequential([
                    iaa.OneOf([
                        # Improve or worsen the contrast of images.
                        iaa.LinearContrast((0.5, 2.0)),
                        iaa.GammaContrast((0.5, 2.0)),
                        iaa.LogContrast(gain=(0.6, 1.4)),
                        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                        ])
            ])

        p_min = p_array.min()
        p_max = p_array.max()

        p_array = (p_array - p_min) / (p_max - p_min)
        p_array = contrast_seq(images=p_array)
        p_array = (p_array * (p_max - p_min)) + p_min
        p = torch.from_numpy(p_array)

        return (p,s)

    def _randomcrop_fun(self, p,s):
        h, w = p.shape[1:]
        new_h, new_w = self.randomcrop_output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        p = p[:, top: top + new_h,
                 left: left + new_w]

        s = s[:, top: top + new_h,
                 left: left + new_w]

        return (p,s)

    def __getitem__(self, i):
        assert(type(i) is int)
        ### Load data from disk lazily
        ### Comment: the following methods are not implemented:
        ### minmax, scale_data, dup_data_w_left_right_flip

        load_batch = 1
        orig_img_shape = None

        ## Need to re-work!!!
        cur_img_path = self.data_path + '/proj/' + str(i).zfill(5) + '.png'
        cur_seg_path = self.data_path + '/seg/' + str(i).zfill(5) + '.png'

        cur_img_np = cv2.imread(cur_img_path, 0)
        cur_projs_np = np.zeros((load_batch, cur_img_np.shape[0], cur_img_np.shape[1])).astype(np.float32)
        cur_projs_np[0, :, :] = cur_img_np.astype(np.float32)
        cur_projs = torch.from_numpy(cur_projs_np)

        cur_seg_np = cv2.imread(cur_seg_path, 0) / 255
        cur_segs_np = np.zeros((load_batch, cur_seg_np.shape[0], cur_seg_np.shape[1])).astype(np.float32)
        cur_segs_np[0, :, :] = cur_seg_np.astype(np.float32)
        cur_segs = torch.from_numpy(cur_segs_np)

        if self.debug_label:
            plt.figure()
            plt.imshow(cur_seg_np)
            plt.show()

        assert(cur_seg_np.shape[0]==cur_img_np.shape[0])

        if orig_img_shape is None:
            orig_img_shape = (cur_img_np.shape[0], cur_img_np.shape[1])
        else:
            assert(orig_img_shape[0] == cur_img_np.shape[0])
            assert(orig_img_shape[1] == cur_img_np.shape[1])

        if self.proj_pad_dim > 0:
            # only support square images for now
            assert(cur_projs.shape[-1] == cur_projs.shape[-2])
            self.extra_pad = calc_pad_amount(self.proj_pad_dim, cur_projs.shape[-1])

        p = cur_projs
        cur_segs_dice = torch.zeros(load_batch, self.num_classes, cur_segs.shape[1], cur_segs.shape[2])

        for c in range(self.num_classes):
            cur_segs_dice[0,c,:,:] = cur_segs[:,:] == c

        s = cur_segs_dice.clone().detach()[0,:,:,:]

        self.need_to_pad_proj = self.extra_pad > 0

        if (self.prob_of_aug > 0) and (random.random() < 0.5):
            # print('data-aug...')
            if (random.random() < 0.5):
                # normalize to [0,1] to apply noise
                p_min = p.min()
                p_max = p.max()

                p = (p - p_min) / (p_max - p_min)

                cur_noise_sigma = random.uniform(0.005, 0.01)
                p += torch.randn(p.shape) * cur_noise_sigma

                p = (p * (p_max - p_min)) + p_min

                if self.print_aug_info:
                    print('noise sigma: {:.3f}'.format(cur_noise_sigma))

            if (random.random() < 0.5):
                # normalize to [0,1] to apply gamma
                p_min = p.min()
                p_max = p.max()

                p = (p - p_min) / (p_max - p_min)

                gamma = random.uniform(0.7,1.3)
                p.pow_(gamma)

                p = (p * (p_max - p_min)) + p_min

                if self.print_aug_info:
                    print('gamma = {:.2f}'.format(gamma))

        # end data aug

        if (self.do_heavy_aug) and (random.random() < 0.5):
            #print('heavy augmenting...')
            if (random.random() < 0.5):
                (p,s) = self._invert_fun(p,s)

            if (random.random() < 0.5):
                (p,s) = self._peppersalt_fun(p,s)

            if (random.random() < 0.5):
                (p,s) = self._contrast_fun(p,s)

            if (random.random() < 0.5):
                (p,s) = self._affine_fun(p,s)

            (p,s) = self._heavy_fun(p,s)

        if self.need_to_pad_proj:
            p = torch.from_numpy(np.pad(p.numpy(),
                                 ((0, 0), (self.extra_pad, self.extra_pad), (self.extra_pad, self.extra_pad)),
                                 'reflect'))

        if self.do_norm_01_scale:
            p = (p - p.mean()) / p.std()

        #if not self.valid:
        #    (p,s,cur_lands) = self._randomcrop_fun(p,s,cur_lands)

        # plt.imshow(p.numpy()[0])
        # plt.show()

        return (p,s)

def get_orig_img_shape(h5_file_path, pat_ind):
    f = h5.File(h5_file_path, 'r')

    s = f['{:02d}/projs'.format(pat_ind)].shape

    assert(len(s) == 3)

    return (s[1], s[2])
