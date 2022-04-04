import os
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data as Dataset
from torchvision import transforms
import torch
import cv2
import albumentations
import albumentations.augmentations as A



class PairedDataset(Dataset):
    def __init__(self, img_dir, size=256, mode='train', aug=False):
        super(PairedDataset, self).__init__()
        self.img_dir = img_dir
        self.size = size
        self.mode = mode
        self.aug = aug
        

        images = []
        for root, _ , fnames in sorted(os.walk(self.img_dir)):
             for fname in fnames:
                if _is_image(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        self.images = images

        if aug:
            self.aug_t = albumentations.Compose([
                            A.transforms.HorizontalFlip(p=0.5),
                            A.transforms.ShiftScaleRotate(shift_limit=0.1,
                                                          scale_limit=0.2,
                                                          rotate_limit=15,
                                                          border_mode=cv2.BORDER_CONSTANT,
                                                          value=(255,255,255),
                                                          mask_value=0,
                                                          p=0.5),])

    
    def _is_image(filename):
        img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.svg', '.tiff']
        return any(filename.endswith(extension.lower()) for extension in img_extensions)

    
    def _mask_encoder(label_array):
        labels = sorted(np.unique(label_array))
        shape = label_array.shape
        one_hot = np.zeros((max(labels)+1, *shape))
        for i in range(len(labels)):
            one_hot[i][label_array==labels[i]] = 1.0 
        return one_hot


    @staticmethod
    def preprocess(img):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        img = image_transform(img)
        return img


    def __getitem__(self, i):
        
        source = Image.open(self.images[i]).convert('RGB')
        tactile_path = self.images[i].replace("source", "tactile").replace("s_", "t_").replace(".png",".tiff")
        tactile = Image.open(tactile_path).convert(mode="L") 
         
        if self.mode=='train' and self.aug:
            augmented = self.aug_t(image=np.array(source), mask=np.array(tactile))
            aug_img_pil = Image.fromarray(augmented['image'])
            # apply pixel-wise transformation
            img_tensor = self.preprocess(aug_img_pil)
            mask_np = np.array(augmented['mask'])

        else:
            img_tensor = self.preprocess(img_pil)
            mask_np = np.array(mask_pil)

        labels = self._mask_labels(mask_np)
        mask_tensor = torch.tensor(labels, dtype=torch.float)
        mask_tensor = (mask_tensor - 0.5) / 0.5
        return img_tensor, mask_tensor
        
    def __len__(self):
        return len(self.images)
        
        
        
