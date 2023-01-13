import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import exists
from TransUNet.transunet import VisionTransformer as ViT_seg
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tqdm import tqdm
import argparse

def load_network_from_checkpoint(checkpoint_filename, pretrained_path):
        print('loading state from checkpoint...')
        prev_state = torch.load(checkpoint_filename)

        print('creating network')
        vit_name = 'R50-ViT-B_16'
        vit_patches_size = 16
        img_size = 224
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16'] #args.vit_name
        config_vit.n_classes = 2
        config_vit.n_skip = 3 #args.n_skip
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size)) #args.img_size
        net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes,
                      batch_norm=False, padding=False, n_classes=2, num_lands=0)
        net.load_from(weights=np.load(pretrained_path))

        net.load_state_dict(prev_state['model-state-dict'])

        return net

def dice_score(pred, grd):
        eps = 1.0e-4
        numerator = 2 * np.sum(np.multiply(pred, grd)) + eps
        denominator = np.sum(np.multiply(pred, pred)) + np.sum(np.multiply(grd, grd)) + eps
        dice = numerator / denominator

        return dice

def compute_confusion_scores(pred, grd):
        eps = 1.0e-4
        TP = np.sum(np.multiply(pred, grd))
        TN = np.sum(np.multiply(np.invert(pred), np.invert(grd)))
        FP = np.sum(np.multiply(pred, np.invert(grd)))
        FN = np.sum(np.multiply(np.invert(pred), grd))

        sensitivity = (TP + eps) / (TP + FN + eps)
        specificity = (TN + eps) / (TN + FP + eps)
        precision = (TP + eps) / (TP + FP + eps)
        accuracy = (TP + TN + eps) / (TP + TN + FP + FN + eps)

        beta = 1
        F1 = (1 + beta*beta) * (precision * sensitivity + eps) / (beta*beta*precision + sensitivity + eps)

        beta = 2
        F2 = (1 + beta*beta) * (precision * sensitivity + eps) / (beta*beta*precision + sensitivity + eps)

        return sensitivity, specificity, precision, accuracy, F1, F2

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Test and Result Calculation for COVID-19 Lesion Segmentation Task.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # network model is first positional arg
        parser.add_argument('root_folder', help='Root folder that contains QaTa-COV19 real data', type=str)

        parser.add_argument('--synthex_checkpoint_filename', help='SyntheX simulation training checkpoint file name', type=str)

        parser.add_argument('--vit_pretrain_file', help='Google pretrained ViT model for transUnet', type=str, default="../data/google_ViTmodels/imagenet21k_R50+ViT-B_16.npz")

        parser.add_argument('--real2real', help='Perform real2real testing.', action='store_true')

        parser.add_argument('--visualize_pred', help='Write visualization figures to disk.', action='store_true')

        parser.add_argument('--create_visplot', help='Create overlay visualization images.', action='store_true')

        args = parser.parse_args()

        root_folder = args.root_folder
        synthex_checkpoint_filename = args.synthex_checkpoint_filename
        pretrained_path = args.vit_pretrain_file
        visualize_pred = args.visualize_pred
        create_visplot = args.create_visplot
        real2real = args.real2real
        img_dim = 224
        seg_threshold = 0.5

        output_folder_prefix = 'real2real_result' if real2real else 'imageng_synthex_result'

        test_folders = [root_folder + "/QaTa-COV19/Fold1/test",
                        root_folder + "/QaTa-COV19/Fold2/test",
                        root_folder + "/QaTa-COV19/Fold3/test",
                        root_folder + "/QaTa-COV19/Fold4/test",
                        root_folder + "/QaTa-COV19/Fold5/test"]

        img_folder = root_folder + "/QaTa-COV19/Images"
        grd_folder = root_folder + "/QaTa-COV19/Ground-truths"
        output_folder = root_folder + "/QaTa-COV19/NetworkDetection"
        count = 0
        dice_coeff_list = []
        sensitivity_list = []
        specificity_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []
        F2_list = []

        for test_folder in test_folders:
                print('Testing...', test_folder)
                proj_folder = test_folder + '/proj'
                grd_folder = test_folder + '/seg'
                output_folder = test_folder + '/' + output_folder_prefix

                if create_visplot and (not os.path.exists(output_folder)):
                        os.makedirs(output_folder)

                if real2real:
                        checkpoint_filename = test_folder + '/../checkpoint/bestvalid_net.pt'
                else:
                        checkpoint_filename = synthex_checkpoint_filename

                net = load_network_from_checkpoint(checkpoint_filename, pretrained_path)

                print('moving network to device...')
                dev = 'cuda'
                net.to(dev)

                for img_file in tqdm(sorted(os.listdir(proj_folder))):
                        # print('Running ...', img_file)
                        if(exists(os.path.join(proj_folder, img_file)) and exists(os.path.join(grd_folder, img_file))):
                                proj_arr = cv2.imread(os.path.join(proj_folder, img_file), 0)
                                # proj_arr = cv2.resize(proj_arr, (img_dim, img_dim))
                                proj_arr = np.expand_dims(proj_arr, axis=0)
                                proj_arr = np.expand_dims(proj_arr, axis=0)
                                proj_torch = torch.from_numpy(proj_arr).type(torch.float)

                                grd_arr = cv2.imread(os.path.join(grd_folder, img_file), 0)
                                # grd_arr = cv2.resize(grd_arr, (img_dim, img_dim), interpolation=cv2.INTER_NEAREST)

                                proj_torch = (proj_torch - proj_torch.mean()) / proj_torch.std()

                                projs = proj_torch.to(dev)
                                net_out = net(projs)

                                pred_masks = net_out

                                pred_mask = pred_masks.detach().cpu()[:, 1, :, :]

                                pred_arr = pred_mask.numpy()[0, :, :]

                                # cv2.imwrite(os.path.join(output_folder, img_file), pred_mask)
                                # np.save(output_folder + '/pred_mask/pred_mask_' + str(count).zfill(3) + ".npy", pred_mask)

                                dice_coeff = dice_score(pred_arr > seg_threshold, grd_arr > 0)
                                dice_coeff_list.append(dice_coeff)

                                cur_fold_dice_coeff_arr = np.array(dice_coeff_list)
                                # print('cur fold mean dice score:', np.mean(cur_fold_dice_coeff_arr), ' +/- ', np.std(cur_fold_dice_coeff_arr))

                                confus_scores = compute_confusion_scores(pred_arr > seg_threshold, grd_arr > 0)
                                sensitivity = confus_scores[0]
                                specificity = confus_scores[1]
                                precision   = confus_scores[2]
                                accuracy    = confus_scores[3]
                                F1          = confus_scores[4]
                                F2          = confus_scores[5]

                                sensitivity_list.append(sensitivity)
                                specificity_list.append(specificity)
                                precision_list.append(precision)
                                accuracy_list.append(accuracy)
                                F1_list.append(F1)
                                F2_list.append(F2)

                                if create_visplot:
                                        img_arr = proj_arr[0,0,:,:]
                                        if img_arr.dtype != np.uint8:
                                                img_arr = 255 * img_arr
                                        if pred_arr.dtype != np.uint8:
                                                pred_arr = 255 * pred_arr
                                        if grd_arr.dtype != np.uint8:
                                                grd_arr = 255 * grd_arr

                                        alpha = 0.7
                                        img_rgb = np.repeat(np.expand_dims(img_arr, axis=-1), 3, axis=-1)
                                        pred_rgb = np.concatenate((np.expand_dims(pred_arr, axis=-1), np.zeros((pred_arr.shape[0], pred_arr.shape[1], 2))), axis=-1)
                                        grd_rgb = np.concatenate((np.expand_dims(grd_arr, axis=-1), np.zeros((grd_arr.shape[0], grd_arr.shape[1], 2))), axis=-1)
                                        img_pred_blended = cv2.addWeighted(img_rgb.astype(np.uint8), alpha, pred_rgb.astype(np.uint8), 1-alpha, 0.0)
                                        img_grd_blended = cv2.addWeighted(img_rgb.astype(np.uint8), alpha, grd_rgb.astype(np.uint8), 1-alpha, 0.0)

                                        pred_fig, (pred_ax1, pred_ax2, pred_ax3) = plt.subplots(1, 3, figsize=(21, 7))
                                        pred_fig.suptitle('Dice: {:.2f}   Sensitivity: {:.2f}   Specificity: {:.2f}   Precision: {:.2f}   Accuracy: {:.2f}'
                                                          .format(dice_coeff, sensitivity, specificity, precision, accuracy), fontsize=16)
                                        pred_ax1.imshow(img_arr, cmap='gray')
                                        pred_ax1.set_title('Input X-ray')
                                        pred_ax2.imshow(img_pred_blended)
                                        pred_ax2.set_title('Network Segmentation')
                                        pred_ax3.imshow(img_grd_blended)
                                        pred_ax3.set_title('Groundtruth Mask')
                                        if visualize_pred:
                                                plt.show()
                                        plt.savefig(os.path.join(output_folder, img_file))
                                        plt.close()

                print('deleting network...')
                del net

        sensitivity_arr = np.array(sensitivity_list)
        specificity_arr = np.array(specificity_list)
        precision_arr   = np.array(precision_list)
        accuracy_arr    = np.array(accuracy_list)
        F1_arr          = np.array(F1_list)
        F2_arr          = np.array(F2_list)

        print('mean sensitivity: {:.4f} +/- {:.4f} '.format(np.mean(sensitivity_arr), np.std(sensitivity_arr)))
        print('mean specificity: {:.4f} +/- {:.4f} '.format(np.mean(specificity_arr), np.std(specificity_arr)))
        print('mean precision: {:.4f} +/- {:.4f} '.format(np.mean(precision_arr), np.std(precision_arr)))
        print('mean accuracy: {:.4f} +/- {:.4f} '.format(np.mean(accuracy_arr), np.std(accuracy_arr)))
        print('mean F1: {:.4f} +/- {:.4f} '.format(np.mean(F1_arr), np.std(F1_arr)))
        print('mean F2: {:.4f} +/- {:.4f} '.format(np.mean(F2_arr), np.std(F2_arr)))

