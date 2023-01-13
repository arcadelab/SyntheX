# Implementation referred to https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/adda.py

import argparse
import shutil
import torch.optim as optim

from TransUNet.transunet import VisionTransformer as ViT_seg
from TransUNet.transunet import transunet_feature_extractor
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
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
    parser = argparse.ArgumentParser(description='Hip Imaging Training Script with ADDA for Controlled Study.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     # input data file is first positional arg
    parser.add_argument('source_data_file_path', help='Path to source domain datafile containing projections', type=str)

    parser.add_argument('source_label_file_path', help='Path to source domain datafile containing segmentations and landmarks', type=str)

    parser.add_argument('target_data_file_path', help='Path to target domain datafile containing projections', type=str)

    parser.add_argument('target_label_file_path', help='Path to target domain datafile containing segmentations and landmarks', type=str)

    parser.add_argument('--vit_pretrain_file', help='Google pretrained ViT model for transUnet', type=str, default="../data/google_ViTmodels/imagenet21k_R50+ViT-B_16.npz")

    parser.add_argument('--train-pats', help='comma delimited list of patient IDs used for training', type=str)

    parser.add_argument('--valid-pats', help='comma delimited list of patient IDs used for validation', type=str)

    parser.add_argument('--num-classes', help='The number of label classes to be identified', type=int, default=7)

    parser.add_argument('--batch-size', help='Number of images each minibatch', type=int, default=3)

    parser.add_argument('--ds-factor', help='Downsample factor', type=int, default=8)

    parser.add_argument('--unet-img-dim', help='Dimension to adjust input images to before inputting into U-Net', type=int, default=192)

    parser.add_argument('--checkpoint-net', help='Path to network saved as checkpoint', type=str, default='zz_checkpoint.pt')

    parser.add_argument('--best-net', help='Path to network saved with best score on the validation data', type=str, default='zz_best_valid.pt')

    parser.add_argument('--checkpoint-freq', help='Frequency (in terms of epochs) at which to save the network checkpoint to disk.', type=int, default=50)

    parser.add_argument('--no-save-best-valid', help='Do not save best validation netowrk to disk.', action='store_true')

    parser.add_argument('--max-num-epochs', help='Maximum number of epochs', type=int, default=50)

    parser.add_argument('--train-loss-txt', help='output file for training loss', type=str, default='train_iter_loss.txt')

    parser.add_argument('--valid-loss-txt', help='output file for validation loss', type=str, default='valid_loss.txt')

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

    parser.add_argument('--use-lands', help='Learn landmark heatmaps', action='store_true')

    parser.add_argument('--heat-coeff', help='Weighting applied to heatmap loss - dice gets one minus this.', type=float, default=0.5)

    parser.add_argument('--dice-valid', help='Use only dice validation loss even when training with dice + heatmap loss', action='store_true')

    parser.add_argument('--unet-no-res', help='Do not use residual connections in U-Net blocks', action='store_true')

    parser.add_argument('--train-valid-split', help='Ratio of training data to keep as training, one minus this is used for validation. Enabled when a value in [0,1] is provided, and overrides the valid-pats flag.', type=float, default=-1)

    parser.add_argument('--model-file', help='Network model file trained on source dlmain', type=str, default='/Users/gaocong/Documents/Research/Generalization/mac/output/yy_best_net_drr1_sf_p5.pt')

    parser.add_argument('--k-disc', help='discriminator training iterations', type=int, default=20)

    parser.add_argument('--k-clf', help='clasifier training iterations', type=int, default=5)

    args = parser.parse_args()

    source_data_file_path = args.source_data_file_path
    source_label_file_path = args.source_label_file_path
    target_data_file_path = args.target_data_file_path
    target_label_file_path = args.target_label_file_path

    vit_pretrain_file = args.vit_pretrain_file

    assert(args.train_pats is not None)
    train_pats = [int(i) for i in args.train_pats.split(',')]
    assert(len(train_pats) > 0)

    if args.train_valid_split < 0:
        assert(args.valid_pats is not None)
        valid_pats = [int(i) for i in args.valid_pats.split(',')]
        assert(len(valid_pats) > 0)

    save_best_valid = not args.no_save_best_valid

    num_classes = args.num_classes

    batch_size = args.batch_size

    ds_factor = args.ds_factor

    proj_unet_dim = args.unet_img_dim

    best_valid_filename = args.best_net
    checkpoint_filename = args.checkpoint_net

    checkpoint_freq = args.checkpoint_freq

    # Optimizer and Learning Rate Scheduler
    optim_type = 'sgd'
    init_lr = 0.1
    patient_epoch = 20
    nesterov = True
    momentum = 0.9
    wgt_decay = 0.0001
    lr_sched_meth = 'step'#'plateau'
    lr_patience = 10
    lr_cooldown = 10

    num_epochs = args.max_num_epochs

    train_loss_txt_path = args.train_loss_txt
    valid_loss_txt_path = args.valid_loss_txt
    loss_fig_file = args.loss_fig

    max_hours = args.max_hours
    enforce_max_hours = max_hours > 0

    cpu_dev = torch.device('cpu')

    if args.no_gpu:
        dev = cpu_dev
    else:
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_valid_split = args.train_valid_split

    unet_num_lvls = args.unet_num_lvls
    unet_init_feats_exp = args.unet_init_feats_exp
    unet_batch_norm = args.unet_batch_norm
    unet_padding = args.unet_padding
    unet_no_max_pool = args.unet_no_max_pool
    unet_use_res = not args.unet_no_res
    unet_block_depth = args.unet_block_depth

    unet_model_file = args.model_file

    data_aug = args.data_aug
    heavy_aug = args.heavy_aug

    num_lands = get_num_lands_from_dataset(source_data_file_path)
    print('num. lands read from file: {}'.format(num_lands))
    assert(num_lands > 0)

    heat_coeff = args.heat_coeff

    use_dice_valid = args.dice_valid

    k_disc = args.k_disc
    k_clf = args.k_clf

    train_idx = None
    valid_idx = None

    lrs_is_cos  = lr_sched_meth == 'cos'
    lrs_none    = lr_sched_meth == 'none'
    lrs_plateau = lr_sched_meth == 'plateau'

    print('initializing source training dataset/dataloader')
    source_train_ds = get_dataset(source_data_file_path, source_label_file_path, train_pats, num_classes=num_classes,
                           pad_img_dim=proj_unet_dim,
                           data_aug=data_aug, heavy_aug=heavy_aug, train_valid_split=train_valid_split,
                           train_valid_idx=(train_idx,valid_idx),
                           dup_data_w_left_right_flip=False, ds_factor=ds_factor)
    if train_valid_split >= 0:
        assert(type(source_train_ds) is tuple)
        (source_train_ds, source_valid_ds, train_idx, valid_idx) = source_train_ds

    #data_minmax = train_ds.rob_minmax

    num_data_workers = 8 if data_aug else 0

    source_train_dl = DataLoader(source_train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_data_workers)

    train_ds_len = len(source_train_ds)

    print('Length of source training dataset: {}'.format(train_ds_len))

    print('initializing target training dataset/dataloader')
    target_train_ds = get_dataset(target_data_file_path, target_label_file_path, train_pats, num_classes=num_classes,
                           pad_img_dim=proj_unet_dim,
                           data_aug=data_aug, heavy_aug=heavy_aug, train_valid_split=train_valid_split,
                           train_valid_idx=(train_idx,valid_idx),
                           dup_data_w_left_right_flip=False, ds_factor=ds_factor)
    if train_valid_split >= 0:
        assert(type(target_train_ds) is tuple)
        (target_train_ds, target_valid_ds, train_idx, valid_idx) = target_train_ds

    target_train_dl = DataLoader(target_train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_data_workers)

    if train_valid_split < 0:
        print('initializing target validation dataset')
        target_valid_ds = get_dataset(target_data_file_path, target_label_file_path, valid_pats, num_classes=num_classes,
                               pad_img_dim=proj_unet_dim, ds_factor=ds_factor, valid=True)

    print('Length of validation dataset: {}'.format(len(target_valid_ds)))

    best_valid_loss = None
    epoch = 0

    print('creating source network')
    vit_name = 'R50-ViT-B_16'
    vit_patches_size = 16
    img_size = proj_unet_dim
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16'] #args.vit_name
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3 #args.n_skip
    pretrained_path = vit_pretrain_file
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size)) #args.img_size

    net_src = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes,
                  batch_norm=unet_batch_norm, padding=unet_padding, n_classes=num_classes, num_lands=num_lands)

    print('creating target network')
    net_tar = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes,
                  batch_norm=unet_batch_norm, padding=unet_padding, n_classes=num_classes, num_lands=num_lands)

    print('moving target model network to device...')
    target_model = net_tar.to(dev)
    print('loading target model state from pretained ', unet_model_file)  #load from pretrain model: yy_best
    load_map_location = torch.device('cuda')
    load_model_state = torch.load(unet_model_file, map_location=load_map_location)
    target_model.load_state_dict(load_model_state['model-state-dict'])

    discriminator = nn.Sequential(  # discriminator network
        nn.Conv2d(16, 16, 3, 1, 1),
        nn.ReLU(),
        #nn.MaxPool2d(3, stride=2), # remove maxpooling
        nn.Conv2d(16, 8, 3, 1, 1),
        nn.ReLU(),
        #nn.MaxPool2d(3, stride=2), # remove maxpooling
        nn.Conv2d(8, 4, 3, 1, 1),
        nn.ReLU(),
        #nn.MaxPool2d(3, stride=2), # remove maxpooling
        nn.Conv2d(4, 1, 3, 1, 1),
        nn.Sigmoid())

    discriminator = discriminator.to(dev)

    print('creating loss function')
    if num_lands > 0:
        print('  Dice + Heatmap Loss...')
        criterion = DiceAndHeatMapLoss2D(skip_bg=False, heatmap_wgt=heat_coeff)
    else:
        print('  Dice only...')
        criterion = DiceLoss2D(skip_bg=False)

    lr_sched = None

    print('creating target model optimizer...')
    optimizer_tar = optim.SGD(target_model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wgt_decay, nesterov=nesterov)
    lr_sched = optim.lr_scheduler.StepLR(optimizer_tar, step_size=10, gamma=0.5)

    print('creating discriminative model optimizer...')
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001, weight_decay=wgt_decay) # discriminator optimizer

    # discriminator learning rate scheduler
    lr_sched_disc = optim.lr_scheduler.StepLR(discriminator_optim, 500, gamma=0.5)

    # target model learning rate scheduler
    lr_sched_tar = optim.lr_scheduler.StepLR(optimizer_tar, 500, gamma=0.5)

    print('Setup discriminator & reconstruction loss')
    disc_criterion = nn.BCEWithLogitsLoss()

    train_iter_loss_out = RunningFloatWriter(train_loss_txt_path, new_file=True)

    valid_loss_out = RunningFloatWriter(valid_loss_txt_path, new_file=True)

    train_iter_loss_list = []
    valid_iter_loss_list = []

    tot_time_this_session_hours = 0.0
    num_epochs_completed_this_session = 0

    print('Start Training...')

    keep_training = True

    iteration = int( train_ds_len / (batch_size*(k_clf+k_disc)) )
    count_train = 0
    count_valid = 0
    Sigmoid_fun = torch.nn.Sigmoid()
    while keep_training:
        epoch_start_time = time.time()

        print('Epoch: {:03d}'.format(epoch))

        target_model.train()

        num_batches = 0
        avg_loss    = 0.0

        total_loss = 0.0

        running_loss = 0.0
        running_loss_num_iters = int(0.05 * train_ds_len)
        running_loss_iter = 0

        num_examples_run = 0

        batch_iterator = zip(loop_iterable(source_train_dl), loop_iterable(target_train_dl))

        for iter in range(iteration):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True) # True for discriminator
            for iter_disc in range(k_disc):
                (source_x, _, _, _), (target_x, _, _, _) = next(batch_iterator)

                source_x, target_x = source_x.to(dev), target_x.to(dev)

                source_features = transunet_feature_extractor(target_model, source_x)
                target_features = transunet_feature_extractor(target_model, target_x)

                disc_output_source = discriminator(source_features)  # apply relativistic discriminator
                disc_output_target = discriminator(target_features)  # apply relativistic discriminator

                loss = -torch.sum(torch.log(Sigmoid_fun(disc_output_target-disc_output_source)) + torch.log(1-Sigmoid_fun(disc_output_source-disc_output_target)))/(2 * disc_output_source.shape[0] * disc_output_source.shape[2] * disc_output_source.shape[3])  # loss objective for relativistic

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()
                lr_sched_disc.step()


            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for iter_clf in range(k_clf):
                (source_x, mask, lands, heat), (target_x, _, _, _) = next(batch_iterator)

                source_x = source_x.to(dev)
                target_x = target_x.to(dev)
                masks = mask.to(dev)

                if num_lands > 0:
                    if len(heat.shape) > 4:
                        assert(len(heat.shape) == 5)
                        assert(heat.shape[2] == 1)
                        heat = heat.view(heat.shape[0], heat.shape[1], heat.shape[3], heat.shape[4])
                    heats = heat.to(dev)

                target_features = transunet_feature_extractor(target_model, target_x)
                source_features = transunet_feature_extractor(target_model, source_x)

                disc_output_source = discriminator(source_features)  # apply relativistic discriminator
                disc_output_target = discriminator(target_features)  # apply relativistic discriminator

                loss_adv = -torch.sum(torch.log(1-Sigmoid_fun(disc_output_target-disc_output_source)) + torch.log(Sigmoid_fun(disc_output_source-disc_output_target)))/(2 * disc_output_source.shape[0] * disc_output_source.shape[2] * disc_output_source.shape[3])  # unchanged loss function

                # Model Segmentation & Landmark loss
                net_out = target_model(source_x)
                # need to visualize prediction

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

                    loss_model = criterion((pred_masks, pred_heat_maps), (masks, heats))
                else:
                    loss_model = criterion(pred_masks, masks)

                loss = loss_model + 0.01*loss_adv# + 0.01*loss_recon # total loss function

                count_train += 1

                l = loss.item()
                train_iter_loss_out.write(l)
                total_loss += loss.detach().cpu().item()

                optimizer_tar.zero_grad()
                loss.backward()
                optimizer_tar.step()
                lr_sched_tar.step()

        avg_loss = total_loss / (iteration*k_clf)

        print('  Running validation')
        (avg_valid_loss, std_valid_loss) = test_dataset(target_valid_ds, target_model, dev=dev,
                                                        num_lands=0 if use_dice_valid else num_lands, adv_loss=False)

        count_valid += 1

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
            else:
                lr_sched.step()

        epoch += 1

        new_best_valid = False
        if (best_valid_loss is None) or (avg_valid_loss < best_valid_loss):
            best_valid_loss = avg_valid_loss
            new_best_valid = True

        def save_net(net_path):
            tmp_name = '{}.tmp'.format(net_path)
            torch.save({ 'epoch'                : epoch,
                         'model-state-dict'     : target_model.state_dict(),
                         'optim-type'           : optim_type,
                         'optimizer-state-dict' : optimizer_tar.state_dict(),
                         'scheduler-state-dict' : lr_sched.state_dict() if lr_sched is not None else None,
                         'loss'                 : loss,
                         'save-best-valid'      : save_best_valid,
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

        checkpoint_filename = args.checkpoint_net + str(epoch).zfill(2) + ".pt"
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
