import argparse
import shutil
import os.path
import torch.optim as optim

from TransUNet.transunet import VisionTransformer as ViT_seg
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from scaledup_dataset import *
from dataset          import *
from util             import *
from dice             import *

# Fix random seed:
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hip Imaging Training Script for Scaled-up Experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_data_path', help='Path to the training datafile containing projections, segmentations and landmarks', type=str)

    parser.add_argument('valid_data_path', help='Path to the validation datafile containing projections, segmentations and landmarks', type=str)

    parser.add_argument('output_path', help='Path to output folder, containing loss text, models, debug files', type=str)

    parser.add_argument('--vit_pretrain_file', help='Google pretrained ViT model for transUnet', type=str, default="../data/google_ViTmodels/imagenet21k_R50+ViT-B_16.npz")

    parser.add_argument('--debug-freq', help='Number of epochs to save debug data', type=int, default=-1)

    parser.add_argument('--num-lands', help='The number of landmarks', type=int, default=14)

    parser.add_argument('--num-classes', help='The number of label classes to be identified', type=int, default=7)

    parser.add_argument('--batch-size', help='Number of images each minibatch', type=int, default=5)

    parser.add_argument('--ds-factor', help='Downsample factor', type=int, default=8)

    parser.add_argument('--unet-img-dim', help='Dimension to adjust input images to before inputting into U-Net', type=int, default=384)

    parser.add_argument('--checkpoint-net', help='Path to network saved as checkpoint', type=str, default='zz_checkpoint.pt')

    parser.add_argument('--checkpoint-freq', help='Frequency (in terms of epochs) at which to save the network checkpoint to disk.', type=int, default=2)

    parser.add_argument('--no-save-best-valid', help='Do not save best validation netowrk to disk.', action='store_true')

    parser.add_argument('--max-num-epochs', help='Maximum number of epochs', type=int, default=50)

    parser.add_argument('--steplr-patience', help='Number of patient epochs for step lr', type=int, default=2)

    parser.add_argument('--steplr-stepsize', help='Step size epochs for step lr', type=int, default=2)

    parser.add_argument('--loss-fig', help='figure plot for training and validation loss', type=str, default='loss_fig.png')

    parser.add_argument('--no-gpu', help='Only use CPU - do not use GPU even if it is available', action='store_true')

    parser.add_argument('--max-hours', help='Maximum number of hours to run for; terminates when the program does not expect to be able to complete another epoch. A non-positive value indicates no maximum limit.', type=float, default=-1.0)

    parser.add_argument('--unet-num-lvls', help='Number of levels in the U-Net', type=int, default=6)

    parser.add_argument('--unet-init-feats-exp', help='Number of initial features used in the U-Net, two raised to this power.', type=int, default=5)

    parser.add_argument('--unet-batch-norm', help='Use Batch Normalization in U-Net', action='store_true')

    parser.add_argument('--unet-padding', help='Add padding to preserve image sizes for U-Net', action='store_true')

    parser.add_argument('--unet-no-max-pool', help='Learn downsampling weights instead of max-pooling', action='store_true')

    parser.add_argument('--unet-block-depth', help='Depth of the blocks of convolutions at each level', type=int, default=2)

    parser.add_argument('--data-aug', help='Randomly augment the data', action='store_true')
    parser.add_argument('--heavy-aug', help='Perform augmentation using imgaug', action='store_true')
    parser.add_argument('--heat-coeff', help='Weighting applied to heatmap loss - dice gets one minus this.', type=float, default=0.5)

    parser.add_argument('--dice-valid', help='Use only dice validation loss even when training with dice + heatmap loss', action='store_true')
    parser.add_argument('--unet-no-res', help='Do not use residual connections in U-Net blocks', action='store_true')
    parser.add_argument('--train-valid-split', help='Ratio of training data to keep as training, one minus this is used for validation. Enabled when a value in [0,1] is provided, and overrides the valid-pats flag.', type=float, default=-1)
    parser.add_argument('--adv-loss', help='Perform first layer reverse gradient adversarial loss training', action='store_true')

    args = parser.parse_args()

    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    vit_pretrain_file = args.vit_pretrain_file

    output_path = args.output_path
    debug_grd_path = output_path + "/grd"
    if not os.path.exists(debug_grd_path):
        os.mkdir(debug_grd_path)

    debug_pred_path = output_path + "/pred"
    if not os.path.exists(debug_pred_path):
        os.mkdir(debug_pred_path)

    debug_runs_path = output_path + "/runs"
    if not os.path.exists(debug_runs_path):
        os.mkdir(debug_runs_path)

    debug_freq = args.debug_freq

    save_debug_data = debug_freq > 0

    num_classes = args.num_classes

    batch_size = args.batch_size

    ds_factor = args.ds_factor

    proj_unet_dim = args.unet_img_dim

    checkpoint_freq = args.checkpoint_freq

    # Optimizer and Learning Rate Scheduler
    optim_type = 'sgd'
    init_lr = 0.1
    patient_epoch = args.steplr_patience
    steplr_step_size = args.steplr_stepsize
    nesterov = True
    momentum = 0.9
    wgt_decay = 0.0001
    lr_sched_meth = 'step'

    num_epochs = args.max_num_epochs

    train_loss_txt_path = output_path + "/train_loss.txt"
    valid_loss_txt_path = output_path + "/valid_loss.txt"
    loss_fig_file = args.loss_fig

    max_hours = args.max_hours
    enforce_max_hours = max_hours > 0

    cpu_dev = torch.device('cpu')

    if args.no_gpu:
        dev = cpu_dev
    else:
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    unet_num_lvls = args.unet_num_lvls
    unet_init_feats_exp = args.unet_init_feats_exp
    unet_batch_norm = args.unet_batch_norm
    unet_padding = args.unet_padding
    unet_no_max_pool = args.unet_no_max_pool
    unet_use_res = not args.unet_no_res
    unet_block_depth = args.unet_block_depth

    data_aug = args.data_aug
    heavy_aug = args.heavy_aug

    num_lands = args.num_lands
    print('num. lands read from file: {}'.format(num_lands))
    assert(num_lands > 0)

    heat_coeff = args.heat_coeff

    use_dice_valid = args.dice_valid

    adv_loss = args.adv_loss

    train_idx = None
    valid_idx = None

    lrs_is_cos  = lr_sched_meth == 'cos'
    lrs_none    = lr_sched_meth == 'none'
    lrs_plateau = lr_sched_meth == 'plateau'

    print('initializing training dataset/dataloader')
    train_data_num = len([name for name in os.listdir(train_data_path + '/proj')])
    train_ds = ScaledupRandomDataAugDataSet(train_data_path, train_data_num, proj_pad_dim=proj_unet_dim, ds_factor=ds_factor, num_classes=num_classes, data_aug=data_aug, heavy_aug=heavy_aug)

    num_data_workers = 0
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_data_workers)

    train_ds_len = len(train_ds)
    print('Length of training dataset: {}'.format(train_ds_len))

    print('initializing validation dataset')
    valid_data_num = len([name for name in os.listdir(valid_data_path + '/proj')])
    valid_ds = ScaledupRandomDataAugDataSet(valid_data_path, valid_data_num, proj_pad_dim=proj_unet_dim, ds_factor=ds_factor, num_classes=num_classes, data_aug=data_aug, heavy_aug=heavy_aug, valid=True)

    print('Length of validation dataset: {}'.format(len(valid_ds)))

    epoch = 0

    print('creating network')
    vit_name = 'R50-ViT-B_16'
    vit_patches_size = 16
    img_size = proj_unet_dim
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16'] #args.vit_name
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3 #args.n_skip
    pretrained_path = vit_pretrain_file
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size)) #args.img_size
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes,
                  batch_norm=unet_batch_norm, padding=unet_padding, n_classes=num_classes, num_lands=num_lands)
    net.load_from(weights=np.load(pretrained_path))

    print('moving network to device...')
    net.to(dev)

    print('creating loss function')
    if num_lands > 0:
        print('  Dice + Heatmap Loss...')
        criterion = DiceAndHeatMapLoss2D(skip_bg=False, heatmap_wgt=heat_coeff)
    else:
        print('  Dice only...')
        criterion = DiceLoss2D(skip_bg=False)

    print('creating SGD optimizer and LR scheduler')
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=momentum, weight_decay=wgt_decay, nesterov=nesterov)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=steplr_step_size, gamma=0.5)

    train_iter_loss_out = RunningFloatWriter(train_loss_txt_path, new_file=True)

    valid_loss_out = RunningFloatWriter(valid_loss_txt_path, new_file=True)

    train_iter_loss_list = []
    valid_iter_loss_list = []

    tot_time_this_session_hours = 0.0
    num_epochs_completed_this_session = 0

    print('Start Training...')

    keep_training = True

    while keep_training:
        epoch_start_time = time.time()

        print('Epoch: {:03d}'.format(epoch))

        net.train()

        num_batches = 0
        avg_loss    = 0.0

        running_loss = 0.0
        running_loss_num_iters = int(0.05 * train_ds_len)
        running_loss_iter = 0

        num_examples_run = 0

        for (i, data) in enumerate(train_dl, 0):
            (proj, mask, lands, heat) = data

            projs = proj.to(dev)
            masks = mask.to(dev)

            if num_lands > 0:
                if len(heat.shape) > 4:
                    assert(len(heat.shape) == 5)
                    assert(heat.shape[2] == 1)
                    heat = heat.view(heat.shape[0], heat.shape[1], heat.shape[3], heat.shape[4])
                heats = heat.to(dev)

            optimizer.zero_grad()

            net_out = net(projs)
            if num_lands > 0:
                pred_masks     = net_out[0]
                pred_heat_maps = net_out[1]
            else:
                pred_masks = net_out

            if pred_masks.shape[-1] > masks.shape[-1]:
                pred_masks = center_crop(pred_masks, masks.shape)

            if num_lands > 0:
                if pred_heat_maps.shape[-1] > heats.shape[-1]:
                    pred_heat_maps = center_crop(pred_heat_maps, heats.shape)

                loss = criterion((pred_masks, pred_heat_maps), (masks, heats))
            else:
                loss = criterion(pred_masks, masks)

            loss.backward()
            optimizer.step()

            num_examples_run += projs.shape[0]

            l = loss.item()

            train_iter_loss_out.write(l)

            avg_loss    += l
            num_batches += 1

            running_loss      += l
            running_loss_iter += 1
            if running_loss_iter == running_loss_num_iters:
                print('    Running Avg. Loss: {:.6f}'.format(running_loss / running_loss_num_iters))

                running_loss_iter = 0
                running_loss      = 0.0

        avg_loss /= num_batches

        print('  Running validation')
        (avg_valid_loss, std_valid_loss) = test_dataset(valid_ds, net, dev=dev,
                                                        num_lands=0 if use_dice_valid else num_lands, adv_loss=adv_loss)

        valid_loss_out.write(avg_valid_loss)

        print('  Avg. Training Loss: {:.6f}'.format(avg_loss))
        print('  Validation Loss: {:.6f} +/- {:.6f}'.format(avg_valid_loss, std_valid_loss))

        train_iter_loss_list.append(avg_loss)
        valid_iter_loss_list.append(avg_valid_loss)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(train_iter_loss_list, 'bo-')
        ax1.set_title('Training Loss')
        ax2.plot(valid_iter_loss_list, 'ro-')
        ax2.set_title('Validation Loss')
        plt.savefig(loss_fig_file)
        plt.close(f)

        if lr_sched is not None:
            if lrs_plateau:
                lr_sched.step(avg_valid_loss)
            elif epoch >= patient_epoch:
                lr_sched.step()

        epoch += 1

        def save_net(net_path):
            tmp_name = '{}.tmp'.format(net_path)
            torch.save({ 'epoch'                : epoch,
                         'model-state-dict'     : net.state_dict(),
                         'optim-type'           : optim_type,
                         'optimizer-state-dict' : optimizer.state_dict(),
                         'scheduler-state-dict' : lr_sched.state_dict() if lr_sched is not None else None,
                         'loss'                 : loss,
                         'num-classes'          : num_classes,
                         'depth'                : unet_num_lvls,
                         'init-feats-exp'       : unet_init_feats_exp,
                         'batch-norm'           : unet_batch_norm,
                         'padding'              : unet_padding,
                         'no-max-pool'          : unet_no_max_pool,
                         'pad-img-size'         : proj_unet_dim,
                         'batch-size'           : batch_size,
                         'data-aug'             : data_aug,
                         'opt-nesterov'         : nesterov,
                         'opt-momentum'         : momentum,
                         'opt-wgt-decay'        : wgt_decay,
                         'num-lands'            : num_lands,
                         'heat-coeff'           : heat_coeff,
                         'use-dice-valid'       : use_dice_valid,
                         'unet-use-res'         : unet_use_res,
                         'unet-block-depth'     : unet_block_depth,
                         'lrs-meth'             : lr_sched_meth,
                         'checkpoint-freq'      : checkpoint_freq,
                         'train-idx'            : train_idx,
                         'valid-idx'            : valid_idx },
                       tmp_name)
            shutil.move(tmp_name, net_path)

        checkpoint_filename = output_path + "/yy_checkpoint_net_" + str(epoch).zfill(2) + ".pt"
        net_saved_this_epoch_path = None
        if (epoch % checkpoint_freq) == 0:
            print('  Saving checkpoint')
            save_net(checkpoint_filename)
            net_saved_this_epoch_path = checkpoint_filename

        epoch_end_time = time.time()

        this_epoch_hours = (epoch_end_time - epoch_start_time) / (60.0 * 60.0)
        print('  This epoch took {:.4f} hours!'.format(this_epoch_hours))

        tot_time_this_session_hours += this_epoch_hours

        num_epochs_completed_this_session += 1

        avg_epoch_time_hours = tot_time_this_session_hours / num_epochs_completed_this_session

        print('  Current average epoch runtime: {:.4f} hours'.format(avg_epoch_time_hours))

        if enforce_max_hours:
            if (tot_time_this_session_hours + avg_epoch_time_hours) > max_hours:
                print('  Exiting - did not expect to be able to complete next expoch within time limit!')
                keep_training = False
        if epoch >= num_epochs:
            keep_training = False
            print('  Exiting - maximum number of epochs performed!')

        if not keep_training:
            print('    saving checkpoint before exit!')

            if net_saved_this_epoch_path is None:
                save_net(checkpoint_filename)
                net_saved_this_epoch_path = checkpoint_filename
            elif net_saved_this_epoch_path != checkpoint_filename:
                shutil.copy(net_saved_this_epoch_path, checkpoint_filename)

    print('Training Hours: {:.4f}'.format(tot_time_this_session_hours))
