# The pipeline to generate random 2D plots and convert them to tactile style

## Table of contents
- [File Structure](#file-structure)
- [Dataset Generation](#dataset-generation)
- [Training](#training)
- [Test](#test)
- [Code References](#code-references)



## File Structure
Below is how the folders and files should be arranged on the project main directory to work by default values. There are also parameters in each executable file to change default paths. 

```
.
├── checkpoints
│   └── pix2pix
├── data
│   ├── test
│   │   ├── source
│   │   └── tactile
│   └── train
│       ├── source
│       └── tactile
├── datagen
│   ├── bezier_generator.py
│   ├── draw_plot.py
│   ├── polygon_gen.py
│   └── utils.py
├── datasets
│   ├── datasets.py
│   ├── __init__.py
│   └── PairedDataset.py
├── discriminators
│   ├── discriminators.py
│   ├── GlobalDiscriminator.py
│   ├── __init__.py
│   └── PatchDiscriminator.py
├── generators
│   ├── BCDUNet.py
│   ├── generators.py
│   ├── __init__.py
│   ├── ResnetGen.py
│   ├── UNet_plusplus.py
│   └── UNet.py
├── models
│   └── pix2pix
│       ├── discloss.npy
│       ├── final_model.pth
│       ├── genloss.npy
│       ├── l1loss.npy
│       └── params.txt
├── Outputs
│   └── pix2pix
├── README.md
├── test.py
├── train.py
└── util.py
```

## Dataset Generation
To start from scratch we need to generate training and testing samples for the model.
If we wish to generate more samples the code is provided on `datagen` folder and we should run `python ./datagen/draw_plot.py`. This file takes input arguments to change the number of instances to generate listed below.

- `--cnt_bezier`: The number of bezier curve samples can be changed using this argument.
- `--cnt_scatter`: The number of scatter plot samples can be changed using this argument.
- `--cnt_polygon`: The number of polygon samples can be changed using this argument.
- `--p_figsize`: This argument controls the percentage of squared, vertical and horizontal plots. For instance with [0.5, 0.3, 0.2] results in 50% squared, 30% vertical and 20% horizontal plots.
- `--p_1D`: This argument controls the percentage of bezier curves that strictly have no intersections. 
- `--p_grid`: This argument controls the percentage of samples with gridlines.

As an example of running the code using all of these arguments we can write `python ./datagen/draw_plot.py --cnt_bezier 10 --cnt_scatter 5 --cnt_polygon 6 --p_figsize 0.33 0.33 0.33 --p_1D 0.9 --p_grid 0.1`. After generating samples, we should split them as train and test sets each having source and tactile folders.

**_NOTE:_** By default a dataset of 5000 samples (2000 bezier curves, 1500 polygons and, 1500 scatter plots) has been provided. 90% of the data was assigned to training process and the rest of them belongs to test process. Both of the subsets are balanced and we can find them on `data` directory.

## Training
Having the dataset prepared we can train a model by running `python train.py`. This file also accepts input arguments but default values work just fine. Below is the list of input arguments for training.
- `--dir`: This argument helps the script locate data directory. By default this folder is placed on the main directory so the argument should be "./" by default.
- `--batch_size`: This argument sets training batch size which is 1 by default to reduce memory consumption and improve the accuracy.
- `--test_batch_size`: This argument sets test batch size which is 16 by default. Unless we need to do a process after certain number of tactile generations, this argument has no effect. 
- `--input_dim`: This argument sets number of input channels which should be left to be 3 for feeding RGB images to the network.
- `--output_dim`: This argument sets number of output channels which should be left to be 3 for generating RGB images out of the network.
- `--gen_filters`: This argument sets starting number of convolutional filters for the generator which controls its depth. The default velue of it is 64. 
- ` --disc_filters`: This argument sets starting number of convolutional filters for the discriminator which controls its depth. The default velue of it is 64.
- `--epoch_count`: This argument sets starting epoch counter, useful if we're loading a half trained model.
- `--total_iters`: This argument sets total number of epochs we're training the network. By default the code runs for 125 iterations.
- `--iter_constant`: This argument sets the number of epochs in which we keep the learning rate constant. By default it would be 25 starting epochs without learning rate decay.
- `--lr`: This argument sets staring learning rate which is 0.0002 by default.
- `--label_smoothing`: This argument enables one-sided label smoothing which is a technique to regulate training GANs.
- `--beta1`: This argument sets beta1 for Adam optimizer which is 0.01 by default.
- `--cuda` This argument disables using gpu accelerated training.
- `--threads`: This argument sets the number of cpu threads for loading the dataset. By default it's set to 8.
- `--lambda_A`: This argument sets the factor of L1 loss term in total loss function which is 10 by default.
- `--lambda_per`: This argument sets the factor of perceptual loss term in total loss function which is 0 by default.
- `--lambda_GP`: This argument sets the factor of gradient penalty loss term in total loss function which is 0.1 by default.
- `--norm`: This argument sets normalization technique used in the generator submodules. By default we use instance normalization mode.
- `--gen`: This argument sets the generator we use. The options are Resnet, UNet++, UNet, UNet-no-skips, and BCDUNet and we use the first option by default.
- `--disc`: This argument sets the generator we use from Global or Patch discriminator and we use the first option by default.
- `--loss`: This argument sets the loss function for GAN loss from Mean Squared Error, Binary Cross-Entropy or Wasserstein loss and we use the first option by default.
- `--no_aug`: This argument disables data augmention on the dataset when training the network.
- `--folder_save`: This argument sets the folder where we want to save the model. By default the model will be stored at `./models/pix2pix`.
- `--folder_load`: This argument sets the folder where we want to load the model from when testing. By default the model will be fetched from `./models/pix2pix`.
- `--checkpoint_interval`: This argument sets the number of epoches elapsed to save a checkpoint at `./checkpoints/pix2pix` by default it's set to -1 and we don't save intermediate checkpoints.
- `--continue_training`: If we use this argument the script load the pretrained model existing in load folder specified above and continues training it.
**_NOTE:_** The model from running the script with default input arguments is already stored and provided so we don't have to run it again.

## Test
Having the dataset and trained model in their corresponding folders, we can run `python test.py`. This way the triplets will be stored at `./Outputs/pix2pix` representing source, ground truth and, generated output respectively.


## Code References
The authors of Pix2Pix and CycleGAN provided the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Our implementation was inspired by the general structure. The Resnet Generator, PatchDiscriminator and ImagePool was implemented directly following their implementation.
