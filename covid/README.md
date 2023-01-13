# COVID-19 Lesion Segmentation

## Data Introduction
### Simulation
We provide part of the synthetic data from the ImagEng lab CT data as example in folder: `/data/covid/imageng_data`. Training and validation images and labels are inside `train` and `valid`, respectively.

### Real
All QaTa 5-fold real data are located in folder: `/data/covid/QaTa-COV19`. Training, validation and testing images and labels are inside `train`, `valid`, and `test`, respectively.

## Train and Test
### Training
Train and test scripts locate in folder `src`. Main training script is `train.py`. It can be run using the following commands:
```bash
SimDataFolder=/data/covid/imageng_data
ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

python train ${SimDataFolder}
--vit_pretrain_file ${ViTpretrain}
--max-num-epochs 20 
--steplr-patience 5 
--steplr-stepsize 5 
--checkpoint-freq 5 
--heavy-aug
```
The outputs will be saved in a subfolder `checkpoint` in the data folder path. For real2real training, the data folder path needs to be adjusted to `Fold*` inside `/data/covid/QaTa-COV19`. Then use the following commands:
```bash
RealDataFolder=/data/covid/QaTa-COV19/Fold1
ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

python train ${RealDataFolder}
--vit_pretrain_file ${ViTpretrain}
--max-num-epochs 50 
--steplr-patience 20 
--steplr-stepsize 10 
--checkpoint-freq 20 
--heavy-aug
```
### Testing
#### Sim2Real
We provide our trained network model on all the simulation data: `/data/covid/checkpoint_net_20.pt`. Sim2Real testing can be done using the following script:
```bash
RootRealDataFolder=/data/covid
ModelCheckpoint=/data/covid/checkpoint_net_20.pt
ViTpretrain=/data/transUnet/imagenet21k_R50+ViT-B_16.npz

python test.py ${RootRealDataFolder} 
--synthex_checkpoint_filename ${ModelCheckpoint}
--vit_pretrain_file ${ViTpretrain}
```
The following commandline outputs are expected:
```commandline
mean sensitivity: 0.8028 +/- 0.1574 
mean specificity: 0.8741 +/- 0.0678 
mean precision: 0.4867 +/- 0.2723 
mean accuracy: 0.8522 +/- 0.0589 
mean F1: 0.5469 +/- 0.2306 
mean F2: 0.6381 +/- 0.2551
```

Due to the excessive model file sizes, we do not provide trained network models on the 5-fold real2real training.
