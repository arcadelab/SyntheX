import numpy as np
from PIL import Image
import h5py
import os
import matplotlib.pyplot as plt
import argparse
import torch
from ncc import ncc_2d
from util import get_gaussian_2d_heatmap

def load_landmark_annotation_fromh5(annot_file):
        hf = h5py.File(annot_file, 'r')
        ld1 = hf.get("ld1")
        ld2 = hf.get("ld2")

        ld_annot_arr = np.array([ld1[0], ld1[1], ld2[0], ld2[1]])

        return ld_annot_arr

def landmark_error(ld1_ind, ld2_ind, ld_annot_arr, pixel_size = 0.5804):
        ld1_err = pixel_size * np.linalg.norm(np.array([ld1_ind[0] - ld_annot_arr[1], ld1_ind[1] - ld_annot_arr[0]]), ord=2)
        ld2_err = pixel_size * np.linalg.norm(np.array([ld2_ind[0] - ld_annot_arr[3], ld2_ind[1] - ld_annot_arr[2]]), ord=2)
        # ld_err = (ld1_err + ld2_err)/2

        return ld1_err, ld2_err

def dice_score(pred, grd):
        eps = 1.0e-4
        numerator = 2 * np.sum(np.multiply(pred, grd)) + eps
        denominator = np.sum(np.multiply(pred, pred)) + np.sum(np.multiply(grd, grd)) + eps
        dice = numerator / denominator
        # print('    dice error:', dice)

        return dice

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Compute numeric results for surgical tool detection task.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('root_fold_path', help='Path to the root 5-fold folder', type=str)

        parser.add_argument('net_det_folder', help='Folder name of network detection. Choices are sim2real_network_detection, and real2real_network_detection', type=str)

        args = parser.parse_args()

        ROOT_FOLD_PATH = args.root_fold_path

        NET_DET_FOLDER = args.net_det_folder

        OUTPUT_FOLDER = "output"

        all_dice_score_list = []
        all_ld1_error_list = []
        all_ld2_error_list = []

        ld_threshold = 0.9
        ds_factor = 2.8125
        patch_size = int(25 * 8 / ds_factor)
        pad_size = int(patch_size / 2)
        sigma = 2.5 * 8 / ds_factor

        landmark_local_template = get_gaussian_2d_heatmap(patch_size, patch_size, sigma)

        for fold_ID in range(1, 6):
                print('Running Fold ', fold_ID)
                FOLD_PATH = ROOT_FOLD_PATH + "/Fold" + str(fold_ID)
                img_folder = FOLD_PATH + "/test"
                output_folder = FOLD_PATH + "/" + OUTPUT_FOLDER + "/" + NET_DET_FOLDER
                total_count = len(os.listdir(img_folder + "/proj"))

                dice_score_list = []
                ld1_error_list = []
                ld2_error_list = []
                for count in range(total_count):
                        ld1_folder = output_folder + '/ld1'
                        ld2_folder = output_folder + '/ld2'
                        pred_mask_folder = output_folder + '/pred_mask'

                        ld1_img = np.load(ld1_folder + '/ld1_' + str(count).zfill(3) + ".npy")
                        ld2_img = np.load(ld2_folder + '/ld2_' + str(count).zfill(3) + ".npy")
                        pred_mask = np.load(pred_mask_folder + '/pred_mask_' + str(count).zfill(3) + ".npy")
                        pred_mask_arr = (pred_mask[0] > 0.5).astype(int)
                        max_ld1_ind = np.unravel_index(np.argmax(ld1_img, axis=None), ld1_img.shape)
                        max_ld2_ind = np.unravel_index(np.argmax(ld2_img, axis=None), ld2_img.shape)

                        grd_land_folder = img_folder + '/lands'
                        ld_annot_arr = load_landmark_annotation_fromh5(grd_land_folder + '/' + str(count).zfill(5) + '.h5')

                        grd_seg_folder = img_folder + '/seg'
                        annot_mask_file = grd_seg_folder + '/' + str(count).zfill(5) + '.tiff'
                        annot_mask_PIL = Image.open(annot_mask_file)
                        annot_mask_arr = np.array(annot_mask_PIL)
                        annot_mask_arr = (annot_mask_arr > 0).astype(int)

                        # Compute ncc based thresholding:
                        ld1_heat_tensor = torch.from_numpy(ld1_img)
                        ld2_heat_tensor = torch.from_numpy(ld2_img)
                        ld1_heat_pad = torch.from_numpy(
                                np.pad(ld1_img, ((pad_size, pad_size), (pad_size, pad_size)), 'reflect'))
                        ld2_heat_pad = torch.from_numpy(
                                np.pad(ld2_img, ((pad_size, pad_size), (pad_size, pad_size)), 'reflect'))

                        ld1_start_roi_row = max_ld1_ind[0]
                        ld1_start_roi_col = max_ld1_ind[1]
                        ld2_start_roi_row = max_ld2_ind[0]
                        ld2_start_roi_col = max_ld2_ind[1]

                        ld1_heat_roi = ld1_heat_pad[ld1_start_roi_row:(ld1_start_roi_row + patch_size),
                                       ld1_start_roi_col:(ld1_start_roi_col + patch_size)]
                        ld2_heat_roi = ld2_heat_pad[ld2_start_roi_row:(ld2_start_roi_row + patch_size),
                                       ld2_start_roi_col:(ld2_start_roi_col + patch_size)]

                        ld1_ncc = ncc_2d(landmark_local_template, ld1_heat_roi)
                        ld2_ncc = ncc_2d(landmark_local_template, ld2_heat_roi)

                        if not (-1 in ld_annot_arr):
                                ld1_err, ld2_err = landmark_error(max_ld1_ind, max_ld2_ind, ld_annot_arr)
                                ld_dist = np.sqrt((max_ld1_ind[0] - max_ld2_ind[0])*(max_ld1_ind[0] - max_ld2_ind[0]) + (max_ld1_ind[1] - max_ld2_ind[1])*(max_ld1_ind[1] - max_ld2_ind[1]))*0.58
                                # This is the physical landmark distance contraints because of the tool size:
                                if ( ld_dist > 15 and ld_dist < 200):
                                        if ld1_ncc > ld_threshold:
                                                ld1_error_list.append(ld1_err)
                                                all_ld1_error_list.append(ld1_err)

                                        if ld2_ncc > ld_threshold:
                                                ld2_error_list.append(ld2_err)
                                                all_ld2_error_list.append(ld2_err)

                        dice = dice_score(pred_mask_arr, annot_mask_arr)
                        dice_score_list.append(dice)
                        all_dice_score_list.append(dice)

        all_dice_score_arr = np.array(all_dice_score_list)
        all_dice_score_std = np.std(all_dice_score_arr)
        all_dice_score_CI = np.round(1.96 * all_dice_score_std / np.sqrt(np.size(all_dice_score_arr)), 2)
        print('Mean DICE score: {:.2f} +/- {:.2f} CI: {:.2f}'.format(np.mean(all_dice_score_arr), all_dice_score_std, all_dice_score_CI))

        all_ld1_error_arr = np.array(all_ld1_error_list)
        all_ld1_error_std = np.std(all_ld1_error_arr)
        all_ld1_error_CI = np.round(1.96 * all_ld1_error_std / np.sqrt(np.size(all_ld1_error_arr)), 2)
        print('Mean Base landmark error: {:.2f} +/- {:.2f} CI: {:.2f}'.format(np.mean(all_ld1_error_arr), all_ld1_error_std, all_ld1_error_CI))

        all_ld2_error_arr = np.array(all_ld2_error_list)
        all_ld2_error_std = np.std(all_ld2_error_arr)
        all_ld2_error_CI = np.round(1.96 * all_ld2_error_std / np.sqrt(np.size(all_ld2_error_arr)), 2)
        print('Mean Tip landmark error: {:.2f} +/- {:.2f} CI: {:.2f}'.format(np.mean(all_ld2_error_arr), all_ld2_error_std, all_ld2_error_CI))

        all_ld_error_arr = np.concatenate((all_ld1_error_arr, all_ld2_error_arr))
        all_ld_error_std = np.std(all_ld_error_arr)
        all_ld_error_CI = np.round(1.96 * all_ld_error_std / np.sqrt(np.size(all_ld_error_arr)), 2)
        print('Mean All landmark error: {:.2f} +/- {:.2f} CI: {:.2f}'.format(np.mean(all_ld_error_arr), all_ld_error_std, all_ld_error_CI))

