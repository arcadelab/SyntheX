import argparse

from dataset import *
from util import *

from ncc import ncc_2d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='estimate landmark locations and write to CSV',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('heat_file_path', help='Path to dataset file containing labelings.', type=str)

    parser.add_argument('heats_group_path', help='H5 group path to heat maps', type=str)

    parser.add_argument('--out', help='output image path', type=str, default='yy_lands_est.csv')

    parser.add_argument('--pat', help='patient index', type=str)

    parser.add_argument('--use-seg', help='Path to segmentation dataset used to assist in detection', type=str, default='')

    parser.add_argument('--no-hdr', help='No CSV header', action='store_true')

    parser.add_argument('--threshold', help='heatmap thresholding', type=int)

    parser.add_argument('--rand', help='Run test on rand data', action='store_true')

    parser.add_argument('--hm-lvl', help='Use heatmap max activation as thresholding otherwise use default cross region', action='store_true')

    parser.add_argument('-d', '--ds-factor', help='Down sample Factor', type=int, default=4)

    args = parser.parse_args()

    heat_file_path = args.heat_file_path
    heats_group_path = args.heats_group_path

    out_csv_path = args.out

    rand = args.rand

    hm_lvl = args.hm_lvl

    ds_factor = args.ds_factor

    pat_ind = args.pat if rand else int(args.pat)

    no_csv_hdr = args.no_hdr

    seg_ds_path = args.use_seg

    threshold = args.threshold

    land_names = get_land_names_from_dataset(heat_file_path)

    num_lands = len(land_names)

    seg_labels_to_use_for_lands = { 'FH-l'   : 5,
                                    'FH-r'   : 6,
                                    'GSN-l'  : 1,
                                    'GSN-r'  : 2,
                                    'IOF-l'  : 1,
                                    'IOF-r'  : 2,
                                    'MOF-l'  : 1,
                                    'MOF-r'  : 2,
                                    'SPS-l'  : 1,
                                    'SPS-r'  : 2,
                                    'IPS-l'  : 1,
                                    'IPS-r'  : 2,
                                    'ASIS-l' : 1,
                                    'ASIS-r' : 2,
                                    'PSIS-l' : 1,
                                    'PSIS-r' : 2,
                                    'PIIS-l' : 1,
                                    'PIIS-r' : 2 }

    csv_out = open(out_csv_path, 'w')
    if not no_csv_hdr:
        csv_out.write('pat,proj,land,row,col,time\n')

    print('reading heatmaps...')
    f = h5.File(heat_file_path, 'r')
    heats = torch.from_numpy(f[heats_group_path][:])
    segs = None
    if seg_ds_path:
        segs = torch.from_numpy(f[seg_ds_path][:])
    f.close()

    patch_size = int(25 * 8 / ds_factor)
    pad_size = int(patch_size/2)
    sigma = 2.5 * 8 / ds_factor

    landmark_local_template = get_gaussian_2d_heatmap(patch_size, patch_size, sigma)

    threshold_list = [0.01, 0.5, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995]

    print('detecting landmark locations...')
    for i in range(heats.shape[0]):
        for land_ind in range(num_lands):
            seg_label_to_use = seg_labels_to_use_for_lands[land_names[land_ind]]

            start_time = time.time()

            cur_heat = heats[i,land_ind,:,:]

            cur_heat_pad = torch.from_numpy(np.pad(cur_heat.cpu().numpy(), ((pad_size, pad_size), (pad_size, pad_size)), 'reflect'))

            def rule_3():
                max_ind = None

                if (segs is None) or (seg_label_to_use is None):
                    max_ind = np.unravel_index(torch.argmax(cur_heat).item(), cur_heat.shape)
                else:
                    tmp_heat = cur_heat.clone().detach()
                    tmp_heat[segs[i,:,:] != seg_label_to_use] = -math.inf

                    max_ind = np.unravel_index(torch.argmax(tmp_heat).item(), cur_heat.shape)
                    if tmp_heat[max_ind[0], max_ind[1]] == -math.inf:
                        max_ind = None

                if max_ind is not None:
                    # Since this index was first computed in the un-padded image, we do not need to subtract
                    # the padding amount to get the start location in the padded image (it implicitly has -12)
                    start_roi_row = max_ind[0]
                    start_roi_col = max_ind[1]

                    heat_roi = cur_heat_pad[start_roi_row:(start_roi_row+patch_size), start_roi_col:(start_roi_col+patch_size)]

                    cut_thrd = threshold_list[threshold-1] if threshold else 0.9
                    # print(ncc_2d(landmark_local_template, heat_roi))
                    if ncc_2d(landmark_local_template, heat_roi) < cut_thrd:
                        max_ind = None

                return max_ind

            def rule_2():
                max_ind = None

                if (segs is None) or (seg_label_to_use is None):
                    max_hm_response = torch.max(cur_heat).item()
                    max_ind = np.unravel_index(torch.argmax(cur_heat).item(), cur_heat.shape)
                else:
                    tmp_heat = cur_heat.clone().detach()
                    tmp_heat[segs[i,:,:] != seg_label_to_use] = -math.inf

                    max_hm_response = torch.max(tmp_heat).item()
                    max_ind = np.unravel_index(torch.argmax(tmp_heat).item(), cur_heat.shape)
                    if tmp_heat[max_ind[0], max_ind[1]] == -math.inf:
                        max_ind = None

                cut_thrd = threshold_list[threshold-1] if threshold else 0.1
                if max_hm_response < cut_thrd:
                    max_ind = None

                return max_ind

            max_ind = rule_2() if hm_lvl else rule_3()
            if max_ind is None:
                # use -1 to indicate landmark not found
                max_ind = [-1, -1]

            stop_time = time.time()

            csv_line = '{},{},{},{},{},{:3f}'.format(pat_ind, i, land_ind, max_ind[0], max_ind[1], stop_time - start_time)
            csv_out.write('{}\n'.format(csv_line))
