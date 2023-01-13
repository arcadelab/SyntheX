from TransUNet.transunet import VisionTransformer as ViT_seg
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from scaledup_dataset import *
from util             import *
from dice             import *
import os
import cv2
import argparse

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='5-fold Real2Real Testing for Surgical Tool Detection Task.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('root_fold_path', help='Path to the root 5-fold folder', type=str)

        args = parser.parse_args()

        ROOT_FOLD_PATH = args.root_fold_path

        proj_unet_dim = 512

        num_classes = 2
        unet_num_lvls = 6
        unet_init_feats_exp = 5
        unet_batch_norm = True
        unet_padding = True
        unet_no_max_pool = True
        num_lands = 2
        unet_use_res = True
        unet_block_depth = 2
        load_from_checkpoint = True

        # Image padding
        extra_pad = calc_pad_amount(proj_unet_dim, 480)

        for fold_ID in range(1, 6):
                print('Running Fold ', fold_ID)
                FOLD_PATH = ROOT_FOLD_PATH + "/Fold" + str(fold_ID)
                checkpoint_filename = FOLD_PATH + "/real2real_model/yy_checkpoint_net_50.pt"

                print('loading state from checkpoint...')
                prev_state = torch.load(checkpoint_filename)

                print('creating network')
                vit_name = 'R50-ViT-B_16'
                vit_patches_size = 16
                img_size = proj_unet_dim
                config_vit = CONFIGS_ViT_seg['R50-ViT-B_16'] #args.vit_name
                config_vit.n_classes = num_classes
                config_vit.n_skip = 3 #args.n_skip
                if vit_name.find('R50') != -1:
                        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size)) #args.img_size

                net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes,
                          batch_norm=unet_batch_norm, padding=unet_padding, n_classes=num_classes, num_lands=num_lands)

                if load_from_checkpoint:
                    net.load_state_dict(prev_state['model-state-dict'])

                print('moving network to device...')
                dev = 'cuda'
                net.to(dev)

                img_folder = FOLD_PATH + "/test/proj"

                output_folder = FOLD_PATH + "/output/real2real_network_detection"
                if not os.path.exists(output_folder):
                        os.mkdir(output_folder)

                if not os.path.exists(output_folder + "/ld1"):
                        os.mkdir(output_folder + "/ld1")

                if not os.path.exists(output_folder + "/ld2"):
                        os.mkdir(output_folder + "/ld2")

                if not os.path.exists(output_folder + "/pred_mask"):
                        os.mkdir(output_folder + "/pred_mask")

                if not os.path.exists(output_folder + "/visualization"):
                        os.mkdir(output_folder + "/visualization")

                count = 0
                for img_file in sorted(os.listdir(img_folder)):
                        print('Running ...', img_file)
                        proj_PIL = Image.open(os.path.join(img_folder, img_file))
                        proj_arr = np.array(proj_PIL)
                        proj_arr = np.expand_dims(proj_arr, axis=0)
                        proj_arr = np.pad(proj_arr, ((0, 0), (extra_pad, extra_pad), (extra_pad, extra_pad)), 'reflect')
                        proj_arr = np.expand_dims(proj_arr, axis=0)
                        proj = torch.from_numpy(proj_arr)

                        projs = proj.to(dev)

                        with torch.no_grad():
                                net_out = net(projs)

                        pred_masks     = net_out[0]
                        pred_heat_maps = net_out[1]

                        proj = center_crop(proj, [1, 1, 480, 480])
                        pred_masks = center_crop(pred_masks, [1, 1, 480, 480])
                        pred_heat_maps = center_crop(pred_heat_maps, [1, 1, 480, 480])

                        for batch_idx in range(pred_heat_maps.shape[0]):
                            for land_idx in range(pred_heat_maps.shape[1]):
                                pred_heat_maps[batch_idx, land_idx, :, :] = pred_heat_maps[batch_idx, land_idx, :, :] / torch.max(pred_heat_maps[batch_idx, land_idx, :, :])

                        heat_sum = torch.sum(pred_heat_maps.detach().cpu(), dim=1)
                        ld1_img = pred_heat_maps.detach().cpu()[0, 0, :, :]
                        ld2_img = pred_heat_maps.detach().cpu()[0, 1, :, :]
                        pred_mask = pred_masks.detach().cpu()[:, 1, :, :]

                        ld1_arr = np.array(ld1_img)
                        ld2_arr = np.array(ld2_img)
                        ld1_arr = (ld1_arr - np.min(ld1_arr))/(np.max(ld1_arr) - np.min(ld1_arr))
                        ld2_arr = (ld2_arr - np.min(ld2_arr))/(np.max(ld2_arr) - np.min(ld2_arr))

                        cv2.imwrite(output_folder + '/ld1/ld1_' + str(count).zfill(3) + ".png", (ld1_arr * 255).astype(np.uint8))
                        cv2.imwrite(output_folder + '/ld2/ld2_' + str(count).zfill(3) + ".png", (ld2_arr * 255).astype(np.uint8))

                        np.save(output_folder + '/ld1/ld1_' + str(count).zfill(3) + ".npy", ld1_img)
                        np.save(output_folder + '/ld2/ld2_' + str(count).zfill(3) + ".npy", ld2_img)
                        np.save(output_folder + '/pred_mask/pred_mask_' + str(count).zfill(3) + ".npy", pred_mask)

                        pred_heat = heat_sum
                        proj_heat = proj + 5*heat_sum
                        mask_sum = torch.zeros([pred_masks.shape[0], pred_masks.shape[2], pred_masks.shape[3]])
                        for idx in range(num_classes):
                            mask_sum += pred_masks.detach().cpu()[:,idx,:,:] * idx * 255/num_classes

                        if True:
                                pred_fig, (pred_ax1, pred_ax2, pred_ax3) = plt.subplots(1, 3, figsize=(21, 7))
                                pred_ax1.imshow(proj[0,0,:,:], cmap='gray')
                                pred_ax1.set_title('Input X-ray')
                                pred_ax2.imshow(mask_sum[0,:,:] * 255/num_classes, cmap='gray')
                                pred_ax2.set_title('Network Segmentation')
                                # pred_ax3.imshow(pred_heat[0,:,:], cmap='gray')
                                pred_ax3.imshow(proj_heat[0,0,:,:], cmap='gray')
                                pred_ax3.set_title('Network Heatmap')
                                # plt.show()
                                plt.savefig(output_folder + '/visualization/' + str(count).zfill(3) + ".png")
                                plt.close()

                        count = count + 1
                        torch.cuda.empty_cache()
