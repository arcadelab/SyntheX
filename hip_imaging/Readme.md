# Hip Imaging

This folder contains training and testing scripts of the hip imaging application. The training, testing and evaluation scripts are located in folder `src/`.

## Training
### Precisely Controlled Study
#### Data Introduction
The paired images and labels (segmentation and landmark) are structured in HDF5 files. They are located in `data/hip_imaging/controlled_study`.
The real images and labels are `real.h5` and `real_label.h5`, respectively. The additional BrainLab LoopX real X-ray images and labels are `real_loopx.h5` and `real_loopx_label.h5`.
The synthetic images include:
```text
realistic.h5 heuristic.h5 naive.h5 cyc-realistic.h5 
```
`cyc-realistic.h5` saves data after CycleGAN model training and inference. The simulation labels are stored in `sim_label.h5`.

#### Training script
The task network training script is `/src/hip_imaging/train_test_code/train.py`. It can be run with the following arguments:
```bash
# Set path to image and label h5 files:
DataH5=/data/hip_imaging/controlled_study/realistic.h5
LabelH5=/data/hip_imaging/controlled_study/sim_label.h5

# Folder to export trainig output files
OutputFolder=/data/hip_imaging/controlled_study/output

# ViT pre-train model
ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

# Patient ID
PatID=1

python train.py ${DataH5} ${LabelH5} 
--train-pats 2,3,4,5,6 
--valid-pats 1 
--checkpoint-net ${OutputFolder}/yy_check_net${PatID}_ 
--train-loss-txt ${OutputFolder}/yy_train_loss${PatID}.txt
--valid-loss-txt ${OutputFolder}/yy_valid_loss${PatID}.txt
--loss-fig ${OutputFolder}/loss_fig${PatID}.png
--vit_pretrain_file ${ViTpretrain}
--batch-size 5
--ds-factor 4
--unet-img-dim 384
--unet-padding
--heavy-aug
```
The domain randomization methods are implemented in dataset.py as functions in class `RandomDataAugDataSet`  (also in scaledup_dataset.py class `ScaledupRandomDataAugDataSet`). `--heavy-aug` refers to strong domain randomization. If using regular domain randomization, it needs to be replaced with `--data-aug`.

This example command refers to 1 fold training. We performed 6-fold training/testing by alternating the patient ID from 1 to 6 as leave-one-out. The training patient IDs are updated accordingly. 

#### Training with Adversarial Discriminative Domain Adaptation (ADDA)
The task network training script is `/src/hip_imaging/train_test_code/train_adda.py`. We provide an example task network pre-train using realistic images with strong domain randomization in `/data/hip_imaging/controlled_study/sim2real_example/yy_check_net1_50.pt` 
It can be run with the following arguments:
```bash
# Set path to image and label h5 files:
SimDataH5=/data/hip_imaging/controlled_study/realistic.h5
SimLabelH5=/data/hip_imaging/controlled_study/sim_label.h5
RealDataH5=/data/hip_imaging/controlled_study/real.h5
RealLabelH5=/data/hip_imaging/controlled_study/real_label.h5

# Point to pre-trained task network model
ModelCheckpoint=/data/hip_imaging/controlled_study/sim2real_example/yy_check_net1_50.pt

# Folder to export trainig output files
OutputFolder=/data/hip_imaging/controlled_study/output

# ViT pre-train model
ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

# Patient ID
PatID=1

python train_adda.py ${SimDataH5} ${SimLabelH5} ${RealDataH5} ${RealLabelH5} 
--model-file ${ModelCheckpoint}
--train-pats 2,3,4,5,6 
--valid-pats 1 
--checkpoint-net ${OutputFolder}/yy_check_net${PatID}_ 
--train-loss-txt ${OutputFolder}/yy_train_loss${PatID}.txt
--valid-loss-txt ${OutputFolder}/yy_valid_loss${PatID}.txt
--loss-fig ${OutputFolder}/loss_fig${PatID}.png
--vit_pretrain_file ${ViTpretrain}
--batch-size 5
--ds-factor 4
--unet-img-dim 384
--unet-padding
--heavy-aug
```

### Scaled-up Experiment
#### Data Introduction
The scaled-up experiments were conducted using large-scale synthetic data generated using DeepDRR from the New Mexico Decedent Datbase. 
Due to the excessive large amount of data, we provide a subset of 1k images and labels in `/home/cong/Research/SyntheX/GitHub/data/hip_imaging/scaledup_experiment/nmd_20CT_1k_ds_4x`.
The images and labels are saved as individual files in each sub folders.

#### Training script
The task network training script is `/src/hip_imaging/train_test_code/train_scaledup.py`. It can be run with the following arguments:
```bash
# Set path to training and validation data folders:
TrainDataFolder=/home/cong/Research/SyntheX/GitHub/data/hip_imaging/scaledup_experiment/nmd_20CT_1k_ds_4x/train_data
LabelH5=/home/cong/Research/SyntheX/GitHub/data/hip_imaging/scaledup_experiment/nmd_20CT_1k_ds_4x/valid_data

# Folder to export trainig output files
OutputFolder=/data/hip_imaging/controlled_study/output

# ViT pre-train model
ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

python train_scaledup.py ${TrainDataFolder} ${ValidDataFolder} ${OutputFolder}
--vit_pretrain_file ${ViTpretrain}
--batch-size 5
--ds-factor 4
--unet-img-dim 384
--max-num-epochs 20
--unet-padding
--heavy-aug
```
## Testing
### Model Inference Results to File
We use the following script to write network model inference results to a h5 file:
```bash
# This example testing is performed on the controlled real data
RealDataH5=/data/hip_imaging/controlled_study/real.h5
RealLabelH5=/data/hip_imaging/controlled_study/real_label.h5

# Patient ID
PatID=1

# Inference result output h5
InferenceH5=/data/hip_imaging/controlled_study/sim2real_example/spec_${PatID}_sim2real.h5

# Network Checkpoint Model
ModelCheckpoint=/data/hip_imaging/controlled_study/sim2real_example/yy_check_net${PatID}_50.pt

python test_ensemble.py ${RealDataH5} ${RealLabelH5} ${InferenceH5}
--pats ${PatID}
--nets ${ModelCheckpoint}
```
This example command refers to 1 fold testing. We performed 6-fold training/testing by alternating the patient ID from 1 to 6 as leave-one-out. The patient IDs are updated accordingly. 
### Export Detection Results to CSV Files
#### Landmark
```bash
# Patient ID
PatID=1

# Inference result output h5
InferenceH5=/data/hip_imaging/controlled_study/sim2real_example/spec_${PatID}_sim2real.h5

# Output Landmark CSV File
LandmarkCSV=/data/hip_imaging/controlled_study/sim2real_example/spec_${patID}_sim2real_lands.csv

python est_lands_csv.py ${InferenceH5}
nn-heats
--use-seg nn-segs
--pat ${PatID}
--out ${LandmarkCSV}
```
#### Segmentation
```bash
# Patient ID
PatID=1

RealLabelH5=/data/hip_imaging/controlled_study/real_label.h5

# Inference result output h5
InferenceH5=/data/hip_imaging/controlled_study/sim2real_example/spec_${PatID}_sim2real.h5

# Output Segmentation Dice CSV File
DiceCSV=/data/hip_imaging/controlled_study/sim2real_example/spec_${patID}_sim2real_dice.csv

python compute_actual_dice_on_test.py ${RealLabelH5} ${InferenceH5}
nn-segs
${DiceCSV}
${PatID}
```
### Compute Numeric Results
```bash
RealLabelH5=/data/hip_imaging/controlled_study/real_label.h5

# Result Output Folder
ResultFolder=/data/hip_imaging/controlled_study/sim2real_example/result

# Result Output CSV
ResultCSV=/data/hip_imaging/controlled_study/sim2real_example/result/result.csv

python calculate_individual_landmark_dice_mean_std_CI.py ${RealLabelH5}
-o ${ResultFolder}
-r ${ResultCSV}
-n realistic
-t Oct12
-s 360
-d 4
```
Due to the excessively large sizes of the network models, we only provide the csv result files for the controlled 6-fold realistic training with strong domain randomization
as an example case to compute the final numeric results. The files are located in `/data/hip_imaging/controlled_study/sim2real_example/result`
The expected commandline outputs are as follows:
```commandline
****************** Landmark Results ********************
Landmark 0 L.FH:
   Mean: 4.25, STD: 2.37, CI: 0.32
Landmark 1 R.FH:
   Mean: 9.69, STD: 5.76, CI: 0.85
Landmark 2 L.GSN:
   Mean: 5.89, STD: 3.54, CI: 0.48
Landmark 3 R.GSN:
   Mean: 9.19, STD: 7.91, CI: 1.09
Landmark 4 L.IOF:
   Mean: 7.59, STD: 10.12, CI: 1.38
Landmark 5 R.IOF:
   Mean: 5.11, STD: 6.64, CI: 0.93
Landmark 6 L.MOF:
   Mean: 6.38, STD: 11.81, CI: 1.51
Landmark 7 R.MOF:
   Mean: 8.40, STD: 4.48, CI: 0.58
Landmark 8 L.SPS:
   Mean: 5.40, STD: 5.15, CI: 0.62
Landmark 9 R.SPS:
   Mean: 5.90, STD: 5.50, CI: 0.67
Landmark 10 L.IPS:
   Mean: 4.39, STD: 3.05, CI: 0.40
Landmark 11 R.IPS:
   Mean: 4.14, STD: 3.16, CI: 0.40
Landmark 12 L.ASIS:
   Mean: 9.38, STD: 6.54, CI: 1.12
Landmark 13 R.ASIS:
   Mean: 7.35, STD: 13.31, CI: 2.43
***********************************
Sim2Real Overall Landmark Result: 
Average error in mm: 6.44  std:  7.05  CI:  0.26
***********************************
****************** Segmentation Results ********************
Dice 0 Left Hemipelvis:
   Mean: 0.84, STD: 0.20, CI: 0.02
Dice 1 Right Hemipelvis:
   Mean: 0.87, STD: 0.14, CI: 0.01
Dice 2 Vertebrae:
   Mean: 0.78, STD: 0.17, CI: 0.02
Dice 3 Upper Sacrum:
   Mean: 0.59, STD: 0.14, CI: 0.01
Dice 4 Left Femur:
   Mean: 0.91, STD: 0.18, CI: 0.02
Dice 5 Right Femur:
   Mean: 0.81, STD: 0.33, CI: 0.03
***********************************
Sim2Real Overall Dice Result: 
Average Dice error in: 0.8  std:  0.23  CI:  0.01
***********************************

Process finished with exit code 0

Landmark 6 L.MOF:
   Mean: 6.38, STD: 11.81, CI: 1.51
Landmark 7 R.MOF:
   Mean: 8.40, STD: 4.48, CI: 0.58
Landmark 8 L.SPS:
   Mean: 5.40, STD: 5.15, CI: 0.62
Landmark 9 R.SPS:
   Mean: 5.90, STD: 5.50, CI: 0.67
Landmark 10 L.IPS:
   Mean: 4.39, STD: 3.05, CI: 0.40
Landmark 11 R.IPS:
   Mean: 4.14, STD: 3.16, CI: 0.40
Landmark 12 L.ASIS:
   Mean: 9.38, STD: 6.54, CI: 1.12
Landmark 13 R.ASIS:
   Mean: 7.35, STD: 13.31, CI: 2.43
***********************************
Sim2Real Overall Landmark Result: 
Average error in mm: 6.44  std:  7.05  CI:  0.26
***********************************
****************** Segmentation Results ********************
Dice 0 Left Hemipelvis:
   Mean: 0.84, STD: 0.20, CI: 0.02
Dice 1 Right Hemipelvis:
   Mean: 0.87, STD: 0.14, CI: 0.01
Dice 2 Vertebrae:
   Mean: 0.78, STD: 0.17, CI: 0.02
Dice 3 Upper Sacrum:
   Mean: 0.59, STD: 0.14, CI: 0.01
Dice 4 Left Femur:
   Mean: 0.91, STD: 0.18, CI: 0.02
Dice 5 Right Femur:
   Mean: 0.81, STD: 0.33, CI: 0.03
***********************************
Sim2Real Overall Dice Result: 
Average Dice error in: 0.8  std:  0.23  CI:  0.01
***********************************

Process finished with exit code 0

```
All of our controlled experiments were performed following the above pipeline and the results were calculated using the same routine.
