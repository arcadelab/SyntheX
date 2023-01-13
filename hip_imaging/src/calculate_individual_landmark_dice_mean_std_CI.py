import h5py
import numpy as np

import argparse

def landmark_accuracy_test(gt_landmark_path, patients, test_landmark_csv, all_landmark_list, landmark_name_list, img_size):
    f = h5py.File(gt_landmark_path, 'r')

    gt_landmark = f[patients + '/lands/'][:, :, :]
    gt_landmark = np.asarray(gt_landmark)
    ## load test landmark results from csv
    csv_info = np.genfromtxt(test_landmark_csv, delimiter=',')
    csv_info = np.asarray(csv_info).copy()
    col = csv_info[1:, 4].flatten()
    row = csv_info[1:, 3].flatten()
    col = col.reshape(-1, 14)
    row = row.reshape(-1, 14)
    test_landmark = np.stack((col, row), axis=1)  # shape the landmark matrix same as gt_landmark,
                                                  # e.g, patient 1: (111, 2, 14)

    ## Find pixel wise distance
    projs, coords, lands = gt_landmark.shape
    for i in range(lands):
        current_list = all_landmark_list[i]
        for j in range(projs):
            x_gt = gt_landmark[j, 0, i]
            x_test = test_landmark[j, 0, i]
            y_gt = gt_landmark[j, 1, i]
            y_test = test_landmark[j, 1, i]
            if x_gt > 0 and y_gt > 0 and x_gt < img_size and y_gt < img_size and x_test > 0 and y_test > 0 and x_test < img_size and y_test < img_size:
                x_difference = (x_gt - x_test) ** 2
                y_difference = (y_gt - y_test) ** 2
                pixel_difference = np.sqrt(x_difference + y_difference)
                current_list.append(pixel_difference)

def dice_score_test(patients, dice_score_csv, all_label_list):
    ## load dice score results from csv
    csv_info = np.genfromtxt(dice_score_csv, delimiter=',')
    current_label_list = [[] for _ in range(6)]
    csv_info = np.asarray(csv_info).copy()
    label_col = csv_info[1:, 2]
    dice_col = csv_info[1:, 3]
    size = len(label_col)
    for i in range(size):
        if label_col[i] == 1:
            current_label_list[0].append(dice_col[i])
        elif label_col[i] == 2:
            current_label_list[1].append(dice_col[i])
        elif label_col[i] == 3:
            current_label_list[2].append(dice_col[i])
        elif label_col[i] == 4:
            current_label_list[3].append(dice_col[i])
        elif label_col[i] == 5:
            current_label_list[4].append(dice_col[i])
        else:
            current_label_list[5].append(dice_col[i])

    for j in range(len(all_label_list)):
        for dice_score in current_label_list[j]:
            all_label_list[j].append(dice_score)

# Landmark names for display
num_landmarks = 14
num_labels = 6
landmark_name_list = ['L.FH', 'R.FH', 'L.GSN', 'R.GSN', 'L.IOF', 'R.IOF', 'L.MOF', 'R.MOF', 'L.SPS', 'R.SPS',
                          'L.IPS', 'R.IPS', 'L.ASIS', 'R.ASIS']
label_name_list = ['Left Hemipelvis', 'Right Hemipelvis', 'Vertebrae', 'Upper Sacrum', 'Left Femur', 'Right Femur']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hip Imaging Evaluation: Compute landmark and segmentation metrics from exported CSV files.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('real_label_file_path', help='Real label file path to datafile containing segmentations and landmarks', type=str)
    parser.add_argument('-o', '--output-csv-path', help='Output CSV file folder path', type=str, default='../../output')
    parser.add_argument('-r', '--result-csv-path', help='Result CSV file path', type=str)
    parser.add_argument('-n', '--exp-name', help='Experiment name as file prefix', type=str, default='drr_nodr')
    parser.add_argument('-t', '--date', help='Experiment Date', type=str, default='')
    parser.add_argument('-d', '--ds-factor', help='Down sample Factor', type=int, default=4)
    parser.add_argument('-s', '--img-size', help='Image size after downsampling', type=int, default=360)

    args = parser.parse_args()
    print_latex = True

    real_label_file_path = args.real_label_file_path
    csv_output_path = args.output_csv_path
    result_csv = args.result_csv_path
    exp_name = args.exp_name
    date = args.date
    ds_factor = args.ds_factor
    img_size = args.img_size

    pixel_size = 0.194

    # Using a list which has 14 elements, and each element is a list with accuracy error
    all_landmark_list = [[] for _ in range(num_landmarks)]
    # Landmark names for display
    test_landmark_csv_path1 = csv_output_path + '/spec_1_sim2real_lands.csv'
    test_landmark_csv_path2 = csv_output_path + '/spec_2_sim2real_lands.csv'
    test_landmark_csv_path3 = csv_output_path + '/spec_3_sim2real_lands.csv'
    test_landmark_csv_path4 = csv_output_path + '/spec_4_sim2real_lands.csv'
    test_landmark_csv_path5 = csv_output_path + '/spec_5_sim2real_lands.csv'
    test_landmark_csv_path6 = csv_output_path + '/spec_6_sim2real_lands.csv'

    landmark_accuracy_test(real_label_file_path, '01', test_landmark_csv_path1, all_landmark_list, landmark_name_list, img_size)
    landmark_accuracy_test(real_label_file_path, '02', test_landmark_csv_path2, all_landmark_list, landmark_name_list, img_size)
    landmark_accuracy_test(real_label_file_path, '03', test_landmark_csv_path3, all_landmark_list, landmark_name_list, img_size)
    landmark_accuracy_test(real_label_file_path, '04', test_landmark_csv_path4, all_landmark_list, landmark_name_list, img_size)
    landmark_accuracy_test(real_label_file_path, '05', test_landmark_csv_path5, all_landmark_list, landmark_name_list, img_size)
    landmark_accuracy_test(real_label_file_path, '06', test_landmark_csv_path6, all_landmark_list, landmark_name_list, img_size)

    ## display landmark result
    print("****************** Landmark Results ********************")
    for i in range(num_landmarks):
        current_list = all_landmark_list[i].copy()
        current_array = np.asarray(current_list)
        current_landmark_mean = np.round(0.194 * ds_factor * np.average(current_array), 2)
        current_landmark_std = np.round(0.194 * ds_factor * np.std(current_array), 2)
        current_landmark_CI = np.round(1.96*current_landmark_std/np.sqrt(np.size(current_array)), 2)
        print('Landmark ' + str(i) + ' ' + landmark_name_list[i] + ':')
        print('   Mean: {:.2f}, STD: {:.2f}, CI: {:.2f}'.format(current_landmark_mean, current_landmark_std, current_landmark_CI))

    # Display Result
    landmark_sim2real_result = []
    for i in range(14):
        current_list = all_landmark_list[i].copy()
        for element in current_list:
            landmark_sim2real_result.append(element)
    landmark_sim2real_result_array = np.array(landmark_sim2real_result)

    landmark_sim2real_mean = np.round(0.194 * ds_factor * np.average(landmark_sim2real_result_array), 2)
    landmark_sim2real_std = np.round(0.194 * ds_factor * np.std(landmark_sim2real_result_array), 2)
    landmark_sim2real_CI = np.round(1.96 * landmark_sim2real_std / np.sqrt(np.size(landmark_sim2real_result_array)),
                                    2)

    print('***********************************')
    print('Sim2Real Overall Landmark Result: ')
    print('Average error in mm:', landmark_sim2real_mean, ' std: ', landmark_sim2real_std, ' CI: ', landmark_sim2real_CI)
    print('***********************************')

    dice_score_csv_path1 = csv_output_path + '/spec_1_sim2real_dice.csv'
    dice_score_csv_path2 = csv_output_path + '/spec_2_sim2real_dice.csv'
    dice_score_csv_path3 = csv_output_path + '/spec_3_sim2real_dice.csv'
    dice_score_csv_path4 = csv_output_path + '/spec_4_sim2real_dice.csv'
    dice_score_csv_path5 = csv_output_path + '/spec_5_sim2real_dice.csv'
    dice_score_csv_path6 = csv_output_path + '/spec_6_sim2real_dice.csv'

    # Segmentation scores
    all_label_list = [[] for _ in range(num_labels)]

    dice_score_test('01', dice_score_csv_path1, all_label_list)
    dice_score_test('02', dice_score_csv_path2, all_label_list)
    dice_score_test('03', dice_score_csv_path3, all_label_list)
    dice_score_test('04', dice_score_csv_path4, all_label_list)
    dice_score_test('05', dice_score_csv_path5, all_label_list)
    dice_score_test('06', dice_score_csv_path6, all_label_list)

    ## display segmentation dice result
    print("****************** Segmentation Results ********************")
    for i in range(num_labels):
        current_list = all_label_list[i].copy()
        current_array = np.asarray(current_list)
        current_dice_mean = np.round(np.average(current_array), 2)
        current_dice_std = np.round(np.std(current_array), 2)
        current_dice_CI = np.round(1.96 * current_dice_std / np.sqrt(np.size(current_array)), 2)
        print('Dice ' + str(i) + ' ' + label_name_list[i] + ':')
        print('   Mean: {:.2f}, STD: {:.2f}, CI: {:.2f}'.format(current_dice_mean, current_dice_std,
                                                                    current_dice_CI))

    # Display Result
    dice_sim2real_result = []
    for i in range(6):
        current_list = all_label_list[i].copy()
        for element in current_list:
            dice_sim2real_result.append(element)
    dice_sim2real_result_array = np.array(dice_sim2real_result)

    dice_sim2real_mean = np.round(np.average(dice_sim2real_result_array), 2)
    dice_sim2real_std = np.round(np.std(dice_sim2real_result_array), 2)
    dice_sim2real_CI = np.round(1.96 * dice_sim2real_std / np.sqrt(np.size(dice_sim2real_result_array)), 2)

    print('***********************************')
    print('Sim2Real Overall Dice Result: ')
    print('Average Dice error in:', dice_sim2real_mean, ' std: ', dice_sim2real_std, ' CI: ', dice_sim2real_CI)
    print('***********************************')
