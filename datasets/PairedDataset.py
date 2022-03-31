import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import random
import platform


def _is_image(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.svg', '.tiff']
    return any(filename.endswith(extension.lower()) for extension in img_extensions)

def mask_encoder(label_array):
    labels = sorted(np.unique(label_array))
    shape = label_array.shape
    one_hot = np.zeros((*shape, max(labels)+1))
    for i in range(len(labels)):
        one_hot[:,:,labels[i]] = label_array==labels[i]
    return one_hot

class PairedDataset(data.Dataset):
    def __init__(self, img_dir, size=256, flip=0.2, jitter=0.2):
        super(PairedDataset, self).__init__()
        self.size = size
        self.img_dir = img_dir
        self.flip = flip
        self.jitter = jitter
        
        images = []
        for root, _ , fnames in sorted(os.walk(self.img_dir)):
             for fname in fnames:
                if _is_image(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        self.images = images


    def __getitem__(self, i):
        if_flip = random.random() < self.flip
        
        source = Image.open(self.images[i]).convert('RGB')
        
        if if_flip:
            source = source.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.jitter:
            source = transforms.ColorJitter(0.02, 0.02, 0.02, 0.02)(source)
        
        if source.size != (self.size,self.size):
            source = source.resize((self.size, self.size), Image.BICUBIC)
        

        source = transforms.ToTensor()(source)
        source = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(source)

        
        tactile_path = self.images[i].replace("source", "tactile").replace("s_", "t_").replace(".png",".tiff")
        tactile = Image.open(tactile_path).convert(mode="L") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        
        if if_flip:
            tactile = tactile.transpose(Image.FLIP_LEFT_RIGHT)
            
        if tactile.size != (self.size,self.size):
            tactile = tactile.resize((self.size, self.size), Image.BICUBIC)
        tactile = mask_encoder(np.array(tactile))
        tactile = transforms.ToTensor()(tactile)
        return source.float(), tactile.float()
        
    def __len__(self):
        return len(self.images)
        
        
        
