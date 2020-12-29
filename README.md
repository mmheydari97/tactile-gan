The implementation of Pix2Pix and CycleGAN.

## File Structure

- Pix2Pix.ipynb and Cyclegan.ipynb are used to train the models
- Test.ipynb loads a model and applies the test set generating images
- Tensorboard_Logger.py contains the necessary functions to write images and losses to a tensorboard to monitor training
- /data contains the training and test images
- /datasets contains the custom dataset pytorch class needed to read in the iamges
- /models is where the saved models go
- /Outputs is where we write images to

Note that a small random subset of Outputs are shown. The contents in /data and /models are placed in [Releases](https://github.com/mkomeili/Image-to-sketch-YanPengGao/releases) due to large file sizes. To duplicate the results, unzip and places the contents from data.zip and models.zip inside the /data and /models folders.

## Code References
The authors of Pix2Pix and CycleGAN provided the [repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Our implementation was inspired by the general structure. The Resnet Generator, PatchDiscriminator and ImagePool was implemented directly following their implementation.
[MTLI Photo-Sketch](https://github.com/mtli/PhotoSketch) was another repository referenced.

The Photo-Sketch dataset was downloaded from the [project page](http://www.cs.cmu.edu/~mengtial/proj/sketch/). The main references from this codebase was creating a custom dataset class and calculating adversarial loss when dealing with multiple real samples.
