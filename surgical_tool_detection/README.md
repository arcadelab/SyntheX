# Surgical Tool Detection
## Data Introduction
### Simulation
Due to the excessive size of the full simulation dataset, we only provide a subset of the simulation training data, located in `/data/surgical_tool_detection/sim/train_data`.
### Real
The 5-fold real data can be found in `/data/surgical_tool_detection/real/Fold*`.

## Train and Test
### Training
Train and test scripts locate in folder `src`. Main training script is `train.py`. It can be run using the following commands:
```bash
# Simulation data train and validation folder:
TrainData=/data/surgical_tool_detection/sim/train_data
ValidData=/data/surgical_tool_detection/sim/valid_data
OutputFolder=/data/surgical_tool_detection/sim/output

ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

python train.py ${TrainData} ${ValidData} ${OutputFolder} --vit_pretrain_file ${ViTpretrain} --heavy-aug --unet-padding
```
For 5-fold real data training, the train/valid folder needs to be passed accordingly.

### Testing
Sim2Real testing is done with the following script. We provided our model checkpoint trained using the full synthetic dataset: `/data/surgical_tool_detection/sim/yy_checkpoint_net_10.pt`.
```bash
# Path to real data folder
RealDataFolder=/data/surgical_tool_detection/real
ModelCheckpoint=/data/surgical_tool_detection/sim/yy_checkpoint_net_10.pt
python test_5fold_sim2real.py ${RealDataFolder} ${ModelCheckpoint}
```
The output results will be saved to each sub Fold folder's `Fold*/output/sim2real_network_detection`.

Real2Real testing is done with the following script.
```bash
# Path to real data folder
RealDataFolder=/data/surgical_tool_detection/real

python test_5fold_real2real.py ${RealDataFolder}
```
The program will automatically search and find each fold's model. However, due to the excessive model sizes, we do not provide all 5 fold real2real models. 
We have attached all real2real network detection results in the subfolders: `Fold*/output/real2real_network_detection`. Especially, the visualizations are also attached.

## Numeric Results Calculation
Sim2Real results:
```bash
RealDataFolder=/data/surgical_tool_detection/real

python calculate_5fold_metric_error.py ${RealDataFolder} sim2real_network_detection
```
The following outputs are expected:
```commandline
Mean DICE score: 0.92 +/- 0.07 CI: 0.01
Mean Base landmark error: 1.09 +/- 0.69 CI: 0.09
Mean Tip landmark error: 1.12 +/- 1.04 CI: 0.13
Mean All landmark error: 1.10 +/- 0.88 CI: 0.08
```

Real2Real results:
```bash
RealDataFolder=/data/surgical_tool_detection/real

python calculate_5fold_metric_error.py ${RealDataFolder} real2real_network_detection
```
The following outputs are expected:
```commandline
Mean DICE score: 0.41 +/- 0.23 CI: 0.03
Mean Base landmark error: 1.09 +/- 0.89 CI: 0.11
Mean Tip landmark error: 1.29 +/- 3.40 CI: 0.43
Mean All landmark error: 1.19 +/- 2.49 CI: 0.22
```
