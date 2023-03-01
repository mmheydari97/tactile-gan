# The pipeline to generate random 2D plots and convert them to tactile style

## Table of contents
- [File Structure](#file-structure)
- [Training](#training)
- [Test](#test)
- [Code References](#code-references)



## File Structure
Below is how the folders and files should be arranged on the project main directory to work by default values. There are also parameters in each executable file to change default paths. 

```
.
├── checkpoints
│   └── [modelname]
│   
├── data
│   ├── test
│   │   ├── source
│   │   └── tactile
│   └── train
│       ├── source
│       └── tactile
│
├── datasets
│   ├── datasets.py
│   ├── __init__.py
│   └── PairedDataset.py
│
├── discriminators
│   ├── discriminators.py
│   ├── __init__.py
│   └── PatchDiscriminator.py
│
├── generators
│   ├── generators.py
│   ├── __init__.py
│   ├── UNet_plusplus.py
│   └── UNet.py
│
├── models
│   └── [modelname]
│       ├── discloss.npy
│       ├── final_model.pth
│       ├── genloss.npy
│       ├── l1loss.npy
│       ├── params.txt
│       └── perloss.npy
│
├── Outputs
│   └── [modelname]
│
├── README.md
├── test.py
├── train.py
└── util.py
```

## Training
Having the dataset prepared we can train a model by running `python train.py`. This file also accepts input arguments but default values work just fine. Below is the list of input arguments for training.
- `--data`: This argument helps the script locate data directory. By default this folder is placed on the main directory so the argument should be "./data" by default.
- `--batch_size`: This argument sets training batch size which is 4 by default. lower batch size will reduce memory consumption, increase time consumption, and might improve the accuracy or destabilize the training.
- `--input_dim`: This argument sets number of input channels which should be left to be 3 for feeding RGB images to the network.
- `--output_dim`: This argument sets number of output channels which should be left to be 3 for generating RGB images out of the network.
- `--initial-epoch`: This argument sets starting epoch counter, useful if we're loading a half trained model.
- `--total_epochs`: This argument sets total number of epochs we're training the network. By default the code runs for 125 iterations.
- `--epoch_constant`: This argument sets the number of epochs in which we keep the learning rate constant. By default it would be 25 starting epochs without learning rate decay.
- `--lr`: This argument sets starting learning rate which is 0.002 by default.
- `--no-label_smoothing`: This argument disables one-sided label smoothing which is a technique to regulate training GANs.
- `--beta1`: This argument sets beta1 for Adam optimizer which is 0.01 by default.
- `--threads`: This argument sets the number of cpu threads for loading the dataset. By default it's set to 8.
- `--lambda_A`: This argument sets the factor of L1 loss term in total loss function which is 5 by default.
- `--lambda_gp`: This argument sets the factor of gradient penalty loss term in total loss function which is 0.1 by default.
- `--lambda_per`: This argument sets the factor of perceptual loss term in total loss function which is 0.2 by default.
- `--w-per`: This argument takes an array of 4 elements, determining the weight of each feature map for perceptual loss
- `--gen`: This argument sets the generator architecture. The options are UNet, and UNet++ which is the default value.
- `--nf`: This argument sets the base number of filters of the generator. It implicitly controls how big the generator should be.
- `--loss`: This argument sets the loss function for GAN loss from Mean Squared Error, Binary Cross-Entropy or Wasserstein loss or hinge loss and we use the first option by default.
- `--no_aug`: This argument disables data augmentation on the dataset when training the network.
- `--target`: This argument determines if the model should work with rgb tactile generation (task 1) or channel-wise tactile generation (task 2). It's set to the first option by default.
- `--version`: This argument sets the version of tactile GAN model. There are 2 models with slightly different loss terms. The second model which is the default, performs slightly better, but slightly more time consuming.
- `--folder_save`: This argument sets the folder where we want to save the model. By default the model will be stored at `./models/pix2obj`.
- `--folder_load`: This argument sets the folder where we want to load the model from when testing. By default the model will be fetched from `./models/pix2obj`.
- `--checkpoint_interval`: This argument sets the number of epochs elapsed to save a checkpoint at `./checkpoints/pix2pix` by default it's set to -1 and we don't save intermediate checkpoints.
- `--continue_training`: If we use this argument the script loads the pretrained model existing in load folder specified above and continues training it.
- `reg_every`: This argument controls how frequently the gradient penalty regularization should be applied to the discriminator.

**_NOTE:_** The model from running the script with default input arguments is already stored and provided so we don't have to run it again.

## Test
Having the dataset and trained model in their corresponding folders, we can run `python test.py --folder [TRAINED_MODEL_DIR]`. This way the model will be loaded, and evaluated with the samples in data folder, using the values listed in `params.txt`. Afterward, you can find the results in `Output` folder under a directory with the same name as `folder_save` argument in training.


## Code References
The authors of Pix2Pix and CycleGAN provided the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Our implementation was inspired by the general structure. The Resnet Generator, PatchDiscriminator and ImagePool was implemented directly following their implementation.
