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

class SurgicalToolRandomDataAugDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, data_num, proj_pad_dim=0, ds_factor=8, num_classes=7, data_aug=False, heavy_aug=False, valid=False):
        self.data_path = data_path
        self.data_num = data_num
        self.ds_factor = ds_factor
        self.num_classes = num_classes
        self.valid = valid

        crop_size = 160 if ds_factor==8 else 320 if ds_factor==4 else 416
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

        self.include_heat_map = True

        self.print_aug_info = False

        self.proj_pad_dim = proj_pad_dim
        self.extra_pad = 0
        self.need_to_pad_proj = False

        self.debug_label = False

    def __len__(self):
        return self.data_num

    def _invert_fun(self, p,s,cur_lands):
        #print('invert fun')
        p_max = p.max()
        #p_min = p.min()
        p = p_max - p

        return (p,s,cur_lands)

    def _noise_fun(self, p,s,cur_lands):
        #print('noise fun')
        # normalize to [0,1] to apply noise
        p_min = p.min()
        p_max = p.max()

        p = (p - p_min) / (p_max - p_min)

        cur_noise_sigma = random.uniform(0.005, 0.01)
        p += torch.randn(p.shape) * cur_noise_sigma

        p = (p * (p_max - p_min)) + p_min

        return (p,s,cur_lands)

    def _gamma_fun(self, p,s,cur_lands):
        #print('gamma fun')
        # normalize to [0,1] to apply gamma
        p_min = p.min()
        p_max = p.max()

        p = (p - p_min) / (p_max - p_min)

        gamma = random.uniform(0.7,1.3)
        p.pow_(gamma)

        p = (p * (p_max - p_min)) + p_min

        return (p,s,cur_lands)

    def _affine_fun(self, p,s,cur_lands):
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
        shear_x = random.uniform(-2, 2)
        shear_y = random.uniform(-2, 2)

        scale_factor = random.uniform(0.9, 1.1)

        if self.print_aug_info:
            print('Rot: {:.2f}'.format(rot_ang))
            print('Trans X: {:.2f} , Trans Y: {:.2f}'.format(trans_x, trans_y))
            print('Shear X: {:.2f} , Shear Y: {:.2f}'.format(shear_x, shear_y))
            print('Scale: {:.2f}'.format(scale_factor))

        p = TF.to_tensor(TF.affine(TF.to_pil_image(p),
                         rot_ang,
                         (trans_x, trans_y),
                         scale_factor,
                         [shear_x, shear_y],
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
                                                  [shear_x, shear_y]))
            if self.pad_data_for_affine:
                s = center_crop(s, orig_s_shape)

        if cur_lands is not None:
            shape_for_center_of_rot = s.shape if s is not None else p.shape

            center_of_rot = ((shape_for_center_of_rot[-2] / 2.0) + 0.5,
                             (shape_for_center_of_rot[-1] / 2.0) + 0.5)

            A_inv = TF._get_inverse_affine_matrix(center_of_rot, rot_ang, (trans_x, trans_y), scale_factor, [shear_x, shear_y])
            A = np.matrix([ [A_inv[0], A_inv[1], A_inv[2]], [A_inv[3], A_inv[4], A_inv[5]], [0,0,1]]).I

            for pt_idx in range(cur_lands.shape[-1]):
                cur_land = cur_lands[:,pt_idx]
                if (not math.isinf(cur_land[0])) and (not math.isinf(cur_land[1])):
                    tmp_pt = A * np.asmatrix(np.pad(cur_land.numpy(), (0,1), mode='constant', constant_values=1).reshape(3,1))
                    xform_l = torch.from_numpy(np.squeeze(np.asarray(tmp_pt))[0:2])
                    if (s is not None) and \
                       ((xform_l[0] < 0) or (xform_l[0] > (orig_s_shape[1] - 1)) or \
                        (xform_l[1] < 0) or (xform_l[1] < (orig_s_shape[0] - 1))):
                        xform_l[0] = math.inf
                        xform_l[1] = math.inf

                    cur_lands[:,pt_idx] = xform_l

        return (p,s,cur_lands)

    def _erase_fun(self, p,s,cur_lands):
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

        return (p,s,cur_lands)

    def _heavy_fun(self, p,s,cur_lands):
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

        return (p,s,cur_lands)

    def _peppersalt_fun(self, p,s,cur_lands):
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

        return (p,s,cur_lands)


    def _contrast_fun(self, p,s,cur_lands):
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

        return (p,s,cur_lands)

    def _randomcrop_fun(self, p,s,cur_lands):
        h, w = p.shape[1:]
        new_h, new_w = self.randomcrop_output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        p = p[:, top: top + new_h,
                 left: left + new_w]

        s = s[:, top: top + new_h,
                 left: left + new_w]

        cur_lands[0,:] = cur_lands[0,:] - left
        cur_lands[1,:] = cur_lands[1,:] - top

        for l_idx in range(cur_lands.shape[-1]):
            cur_l = cur_lands[:, l_idx]

            if not math.isinf(cur_l[0]):
                if (cur_l[0] < 0) or (cur_l[0] > (new_h-1)) or \
                   (cur_l[1] < 0) or (cur_l[1] > (new_w-1)):
                       cur_l[0] = math.inf
                       cur_l[1] = math.inf

        return (p,s,cur_lands)

    def __getitem__(self, i):
        assert(type(i) is int)
        ### Load data from disk lazily
        ### Comment: the following methods are not implemented:
        ### minmax, scale_data, dup_data_w_left_right_flip

        load_batch = 1
        orig_img_shape = None

        ## Need to re-work!!!
        cur_img_path = self.data_path + '/proj/' + str(i).zfill(5) + '.tiff'
        cur_seg_path = self.data_path + '/seg/' + str(i).zfill(5) + '.tiff'
        cur_lands_h5path = self.data_path + '/lands/' + str(i).zfill(5) + '.h5'

        cur_img_PIL = Image.open(cur_img_path)
        cur_img_np = np.asarray(cur_img_PIL)
        cur_projs_np = np.zeros((load_batch, cur_img_np.shape[0], cur_img_np.shape[1])).astype(np.float32)
        cur_projs_np[0, :, :] = cur_img_np.astype(np.float32)
        cur_projs = torch.from_numpy(cur_projs_np)

        cur_seg_PIL = Image.open(cur_seg_path)
        cur_seg_np = np.asarray(cur_seg_PIL).astype(np.float32)
        cur_segs_np = np.zeros((load_batch, cur_seg_np.shape[0], cur_seg_np.shape[1])).astype(np.float32)
        cur_segs_np[0, :, :] = cur_seg_np.astype(np.float32)
        cur_segs = torch.from_numpy(cur_segs_np)

        f_lands = h5.File(cur_lands_h5path, 'r')
        ld1 = np.expand_dims(f_lands['ld1'][:], axis=0)
        ld2 = np.expand_dims(f_lands['ld2'][:], axis=0)
        cur_lands_np = np.transpose(np.concatenate((ld1, ld2), axis=0))
        cur_lands = torch.from_numpy(np.expand_dims(cur_lands_np, axis=0)).to(dtype=torch.float16)

        if self.debug_label:
            plt.figure()
            plt.imshow(cur_seg_np)
            plt.plot(cur_lands_np[0, :], cur_lands_np[1, :], 'r+')
            plt.show()

        assert(cur_seg_np.shape[0]==cur_img_np.shape[0])
        assert(torch.all(torch.isfinite(cur_lands)))  # all inputs should be finite

        if orig_img_shape is None:
            orig_img_shape = (cur_img_np.shape[0], cur_img_np.shape[1])
        else:
            assert(orig_img_shape[0] == cur_img_np.shape[0])
            assert(orig_img_shape[1] == cur_img_np.shape[1])

        for l_idx in range(cur_lands.shape[-1]):
            cur_l = cur_lands[0, :,l_idx]

            if (cur_l[0] < 0) or (cur_l[0] > (orig_img_shape[1]-1)) or \
               (cur_l[1] < 0) or (cur_l[1] > (orig_img_shape[0]-1)):
                   cur_l[0] = math.inf
                   cur_l[1] = math.inf


        if self.proj_pad_dim > 0:
            # only support square images for now
            assert(cur_projs.shape[-1] == cur_projs.shape[-2])
            self.extra_pad = calc_pad_amount(self.proj_pad_dim, cur_projs.shape[-1])

        p = cur_projs
        cur_segs_dice = torch.zeros(load_batch, self.num_classes, cur_segs.shape[1], cur_segs.shape[2])

        for c in range(self.num_classes):
            cur_segs_dice[0,c,:,:] = cur_segs[:,:] == c

        s = cur_segs_dice.clone().detach()[0,:,:,:]
        cur_lands = cur_lands.clone().detach()[0,:,:]

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
            # print('heavy augmenting...')
            if (random.random() < 0.5):
                (p,s,cur_lands) = self._invert_fun(p,s,cur_lands)

            if (random.random() < 0.5):
                (p,s,cur_lands) = self._peppersalt_fun(p,s,cur_lands)

            if (random.random() < 0.5):
                (p,s,cur_lands) = self._contrast_fun(p,s,cur_lands)

            if (random.random() < 0.5):
                (p,s,cur_lands) = self._affine_fun(p,s,cur_lands)

            (p,s,cur_lands) = self._heavy_fun(p,s,cur_lands)

        if self.need_to_pad_proj:
            p = torch.from_numpy(np.pad(p.numpy(),
                                 ((0, 0), (self.extra_pad, self.extra_pad), (self.extra_pad, self.extra_pad)),
                                 'reflect'))

        if self.do_norm_01_scale:
            p = (p - p.mean()) / p.std()

        h = None
        if self.include_heat_map:
            assert(s is not None)
            assert(cur_lands is not None)

            num_lands = cur_lands.shape[-1]

            h = torch.zeros(num_lands, 1, s.shape[-2], s.shape[-1])

            # "FH-l", "FH-r", "GSN-l", "GSN-r", "IOF-l", "IOF-r", "MOF-l", "MOF-r", "SPS-l", "SPS-r", "IPS-l", "IPS-r"
            #sigma_lut = [ 2.5, 2.5, 7.5, 7.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
            sigma_lut = torch.full([num_lands], 2.5 * 8 / self.ds_factor)

            (Y,X) = torch.meshgrid(torch.arange(0, s.shape[-2]),
                                   torch.arange(0, s.shape[-1]))
            Y = Y.float()
            X = X.float()

            for land_idx in range(num_lands):
                sigma = sigma_lut[land_idx]

                cur_land = cur_lands[:,land_idx]

                mu_x = cur_land[0]
                mu_y = cur_land[1]

                if not math.isinf(mu_x) and not math.isinf(mu_y):
                    pdf = torch.exp(((X - mu_x).pow(2) + (Y - mu_y).pow(2)) / (sigma * sigma * -2)) / (2 * math.pi * sigma * sigma)
                    #pdf /= pdf.sum() # normalize to sum of 1
                    h[land_idx,0,:,:] = pdf
            #assert(torch.all(torch.isfinite(h)))

        if (random.random() < 0.5):
            rot_times = random.randint(1,3)

            p = torch.rot90(p, rot_times, [-2, -1])
            s = torch.rot90(s, rot_times, [-2, -1])
            h = torch.rot90(h, rot_times, [-2, -1])

        return (p,s,cur_lands,h)

def get_orig_img_shape(h5_file_path, pat_ind):
    f = h5.File(h5_file_path, 'r')

    s = f['{:02d}/projs'.format(pat_ind)].shape

    assert(len(s) == 3)

    return (s[1], s[2])

def get_num_lands_from_dataset(h5_file_path):
    f = h5.File(h5_file_path, 'r')

    num_lands = int(f['land-names/num-lands'].value)

    f.close()

    return num_lands

def get_land_names_from_dataset(h5_file_path):
    f = h5.File(h5_file_path, 'r')

    num_lands = int(f['land-names/num-lands'].value)

    land_names = []

    for l in range(num_lands):
        s = f['land-names/land-{:02d}'.format(l)].value
        if (type(s) is bytes) or (type(s) is np.bytes_):
            s = s.decode()
        assert(type(s) is str)

        land_names.append(s)

    f.close()

    return land_names


def get_rand_dataset(h5_file_path, label_h5_file_path, pat_inds, num_classes,
                pad_img_dim=0, no_seg=False,
                minmax=None,
                data_aug=False,
                heavy_aug=False,
                train_valid_split=None,
                train_valid_idx=None,
                dup_data_w_left_right_flip=False,
                ds_factor=8,
                load_disk_img=False):
    # classes:
    # 0 --> BG
    # 1 --> Pelvis
    # 2 --> Left Femur
    # 3 --> Right Femur

    need_to_scale_data   = False
    need_to_find_min_max = False

    if minmax is not None:
        if (type(minmax) is bool) and minmax:
            need_to_scale_data = True
            print('need to find min/max for preprocessing...')
            need_to_find_min_max = True
            minmax_min =  math.inf
            minmax_max = -math.inf
        elif type(minmax) is tuple:
            minmax_min = minmax[0]
            minmax_max = minmax[1]
            need_to_scale_data = True
            print('using provided min/max for preprocessing: ({}, {})'.format(minmax_min, minmax_max))

    f_label = h5.File(label_h5_file_path, 'r')
    if not load_disk_img:
        f_drr = h5.File(h5_file_path, 'r')

    all_projs = None
    all_segs  = None
    all_lands = None

    orig_img_shape = None

    for pat_idx in pat_inds:
        pat_g = f_label[pat_idx]
        cur_segs_np = pat_g['segs'][:]
        cur_segs = torch.from_numpy(cur_segs_np)
        assert(len(cur_segs.shape) == 3)

        if not load_disk_img:
            pat_g_drr = f_drr[pat_idx]
            cur_projs_np = pat_g_drr['projs'][:]
            assert(len(cur_projs_np.shape) == 3)
        else:
            pat_num_projs = cur_segs.shape[0]
            cur_projs_np = np.zeros((cur_segs.shape[0], cur_segs.shape[1], cur_segs.shape[2]))
            for img_idx in range(pat_num_projs):
                cur_img_path = h5_file_path + '/' + str(pat_idx) + '/' + str(img_idx).zfill(4) + '.tiff'
                cur_img_PIL = Image.open(cur_img_path)
                cur_projs_np[img_idx] = np.asarray(cur_img_PIL)

        if orig_img_shape is None:
            orig_img_shape = (cur_projs_np.shape[1], cur_projs_np.shape[2])
        else:
            assert(orig_img_shape[0] == cur_projs_np.shape[1])
            assert(orig_img_shape[1] == cur_projs_np.shape[2])

        cur_lands = torch.from_numpy(pat_g['lands'][:])
        print('pat idx:', pat_idx, 'cur_lands:', cur_lands.shape[0])
        print('pat idx:', pat_idx, 'cur_projs:', cur_projs_np.shape[0])
        assert(cur_lands.shape[0] == cur_projs_np.shape[0])
        assert(torch.all(torch.isfinite(cur_lands)))  # all inputs should be finite

        # mark out of bounds landmarks with inf's
        for img_idx in range(cur_lands.shape[0]):
            for l_idx in range(cur_lands.shape[-1]):
                cur_l = cur_lands[img_idx,:,l_idx]

                if (cur_l[0] < 0) or (cur_l[0] > (orig_img_shape[1]-1)) or \
                   (cur_l[1] < 0) or (cur_l[1] > (orig_img_shape[0]-1)):
                       cur_l[0] = math.inf
                       cur_l[1] = math.inf

        if need_to_find_min_max:
            minmax_min = min(minmax_min, cur_projs_np.min())
            minmax_max = max(minmax_max, cur_projs_np.max())

        cur_projs = torch.from_numpy(cur_projs_np)

        # Need a singleton dimension to represent grayscale data
        cur_projs = cur_projs.view(cur_projs.shape[0], 1, cur_projs.shape[1], cur_projs.shape[2])

        if all_projs is None:
            all_projs = cur_projs
        else:
            all_projs = torch.cat((all_projs, cur_projs))

        cur_segs_dice = torch.zeros(cur_segs.shape[0], num_classes, cur_segs.shape[1], cur_segs.shape[2])

        for i in range(cur_segs.shape[0]):
            for c in range(num_classes):
                cur_segs_dice[i,c,:,:] = cur_segs[i,:,:] == c

        if all_segs is None:
            all_segs = cur_segs_dice.clone().detach()
        else:
            all_segs = torch.cat((all_segs, cur_segs_dice))

        if all_lands is None:
            all_lands = cur_lands.clone().detach()
        else:
            all_lands = torch.cat((all_lands, cur_lands))

        if dup_data_w_left_right_flip:
            all_projs = torch.cat((all_projs, torch.flip(cur_projs, [3])))

            # left/right flip the segmentations
            cur_segs_dice = torch.flip(cur_segs_dice, [3])

            assert(cur_segs_dice.shape[1] == 7)  # TODO: allow for a mapping to be passed
            # update l/r labels
            # 0 BG stays the same
            # 1 left hemipelvis <--> 2 right hemipelvis
            # 3 vertebrae stays the same
            # 4 upper sacrum stays the smae
            # 5 left femur <--> 6 left femur

            def swap_classes(c1, c2):
                tmp_copy  = cur_segs_dice[:,c1,:,:].clone().detach()
                cur_segs_dice[:,c1,:,:] = cur_segs_dice[:,c2,:,:]
                cur_segs_dice[:,c2,:,:] = tmp_copy

            swap_classes(1,2)
            swap_classes(5,6)

            # flip lands and update, etc
            for img_idx in range(cur_lands.shape[0]):
                # do the l/r flip for each landmark
                for l_idx in range(cur_lands.shape[-1]):
                    cur_l = cur_lands[img_idx,:,l_idx]
                    if math.isfinite(cur_l[0]) and math.isfinite(cur_l[1]):
                        cur_l[0] = (orig_img_shape[-1] - 1) - cur_l[0]

                # now swap the l/r landmarks
                assert((cur_lands.shape[-1] % 2) == 0)
                for l_idx in range(cur_lands.shape[-1] // 2):
                    tmp_land = cur_lands[img_idx,:,l_idx].clone().detach()
                    cur_lands[img_idx,:,l_idx] = cur_lands[img_idx,:,l_idx+1]
                    cur_lands[img_idx,:,l_idx] = tmp_land

            all_segs = torch.cat((all_segs, cur_segs_dice))
            all_lands = torch.cat((all_lands, cur_lands))

    # end loop over patients
    f_label.close()
    if not load_disk_img:
        f_drr.close()

    # scale to [0,1] if needed
    if need_to_scale_data:
        assert((minmax_max - minmax_min) > 1.0e-6)
        print('scaling data using min/max: {} , {}'.format(minmax_min, minmax_max))
        all_projs = (all_projs - minmax_min) / (minmax_max - minmax_min)

    def set_helper_vars(ds, do_data_aug, do_heavy_aug):
        ds.prob_of_aug = 0.5 if do_data_aug else 0.0
        ds.do_heavy_aug = True if do_heavy_aug else False
        # stuff in some custom vars
        ds.rob_orig_img_shape = orig_img_shape

        ds.rob_data_is_scaled = need_to_scale_data
        if need_to_scale_data:
            ds.rob_minmax = (minmax_min, minmax_max)

    if (train_valid_split is not None) and (train_valid_split > 0):
        print('split dataset into train/validation')
        assert((0.0 < train_valid_split) and (train_valid_split < 1.0))
        num_train = int(math.ceil(train_valid_split * all_projs.shape[0]))
        num_valid = all_projs.shape[0] - num_train

        all_inds = list(range(all_projs.shape[0]))

        if (train_valid_idx is None) or (train_valid_idx[0] is None) or (train_valid_idx[1] is None):
            print('  randomly splitting all complete tensors into training/validation...')
            random.shuffle(all_inds)

            train_inds = all_inds[:num_train]
            valid_inds = all_inds[num_train:]
        else:
            print('  use previously specified split')
            train_inds = train_valid_idx[0]
            valid_inds = train_valid_idx[1]
            assert(len(train_inds) == num_train)
            assert(len(valid_inds) == num_valid)

        train_ds = RandomDataAugDataSet(all_projs[train_inds,:,:,:], all_segs[train_inds,:,:,:], all_lands[train_inds,:,:], proj_pad_dim=pad_img_dim, ds_factor=ds_factor)
        set_helper_vars(train_ds, data_aug, heavy_aug)

        valid_ds = RandomDataAugDataSet(all_projs[valid_inds,:,:,:], all_segs[valid_inds,:,:,:], all_lands[valid_inds,:,:], proj_pad_dim=pad_img_dim, ds_factor=ds_factor)
        set_helper_vars(valid_ds, False, False)

        return (train_ds, valid_ds, train_inds, valid_inds)
    else:
        ds = RandomDataAugDataSet(all_projs, all_segs, all_lands, proj_pad_dim=pad_img_dim, ds_factor=ds_factor)
        set_helper_vars(ds, data_aug, heavy_aug)

        return ds
