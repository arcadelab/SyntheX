import argparse

from TransUNet.transunet import VisionTransformer as ViT_seg
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from dataset import *
from util    import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ensemble segmentation and heatmap estimation for hip imaging application.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_data_file_path', help='Path to the datafile containing projections', type=str)

    parser.add_argument('input_label_file_path', help='Path to the datafile containing groundtruth segmentations and landmarks', type=str)

    parser.add_argument('output_data_file_path', help='Path to the output datafile containing segmentations', type=str)

    parser.add_argument('--nets', help='Paths to the networks used to perform segmentation - specify this after the positional arguments', type=str, nargs='+')

    parser.add_argument('--pats', help='comma delimited list of patient IDs used for testing', type=str)

    parser.add_argument('--no-gpu', help='Only use CPU - do not use GPU even if it is available', action='store_true')

    parser.add_argument('--times', help='Path to file storing runtimes for each image', type=str, default='')

    parser.add_argument('--rand', help='Run test on rand data', action='store_true')

    args = parser.parse_args()

    network_paths = args.nets

    src_data_file_path = args.input_data_file_path
    src_label_file_path = args.input_label_file_path
    dst_data_file_path = args.output_data_file_path

    rand = args.rand

    assert(args.pats is not None)
    test_pats = [i for i in args.pats.split(',')] if rand else [int(i) for i in args.pats.split(',')]
    assert(len(test_pats) > 0)

    cpu_dev = torch.device('cpu')

    torch_map_loc = None

    if args.no_gpu:
        dev = cpu_dev
        torch_map_loc = 'cpu'
    else:
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nets = []
    for net_path in network_paths:
        print('  loading state from disk for: {}'.format(net_path))

        state = torch.load(net_path, map_location=torch_map_loc)

        print('  loading unet params from checkpoint state dict...')
        num_classes         = state['num-classes']
        unet_num_lvls       = state['depth']
        unet_init_feats_exp = state['init-feats-exp']
        unet_batch_norm     = state['batch-norm']
        unet_padding        = state['padding']
        unet_no_max_pool    = state['no-max-pool']
        unet_use_res        = state['unet-use-res']
        unet_block_depth    = state['unet-block-depth']
        proj_unet_dim       = state['pad-img-size']
        batch_size          = state['batch-size']
        num_lands           = state['num-lands']

        print('             num. classes: {}'.format(num_classes))
        print('                    depth: {}'.format(unet_num_lvls))
        print('        init. feats. exp.: {}'.format(unet_init_feats_exp))
        print('              batch norm.: {}'.format(unet_batch_norm))
        print('         unet do pad img.: {}'.format(unet_padding))
        print('              no max pool: {}'.format(unet_no_max_pool))
        print('    reflect pad img. dim.: {}'.format(proj_unet_dim))
        print('            unet use res.: {}'.format(unet_use_res))
        print('         unet block depth: {}'.format(unet_block_depth))
        print('               batch size: {}'.format(batch_size))
        print('              num. lands.: {}'.format(num_lands))

        print('    creating network')
        vit_name = 'R50-ViT-B_16'
        vit_patches_size = 16
        img_size = proj_unet_dim
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes,
                      batch_norm=unet_batch_norm, padding=unet_padding, n_classes=num_classes, num_lands=num_lands)

        net.load_state_dict(state['model-state-dict'])

        del state

        print('  moving network to device...')
        net.to(dev)

        nets.append(net)

    land_names = None
    if num_lands > 0:
        land_names = get_land_names_from_dataset(src_data_file_path)
        assert(len(land_names) == num_lands)

    print('initializing testing dataset')
    test_ds = get_rand_dataset(src_data_file_path, src_label_file_path, test_pats, num_classes=num_classes, pad_img_dim=proj_unet_dim, no_seg=True) if rand else get_dataset(src_data_file_path, src_label_file_path, test_pats, num_classes=num_classes, pad_img_dim=proj_unet_dim, no_seg=True, valid=True)

    print('Length of testing dataset: {}'.format(len(test_ds)))

    print('opening destination file for writing')
    f = h5.File(dst_data_file_path, 'w')

    # save off the landmark names
    if land_names:
        land_names_g = f.create_group('land-names')
        land_names_g['num-lands'] = num_lands

        for l in range(num_lands):
            land_names_g['land-{:02d}'.format(l)] = land_names[l]

    times = []

    print('running network on projections')
    seg_dataset_ensemble(test_ds, nets, f, dev=dev, num_lands=num_lands, times=times, adv_loss=False)

    print('closing file...')
    f.flush()
    f.close()

    if args.times:
        times_out = open(args.times, 'w')
        for t in times:
            times_out.write('{:.6f}\n'.format(t))
        times_out.flush()
        times_out.close()
