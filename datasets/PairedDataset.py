import os
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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
                if self._is_image(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        self.images = images

        if aug:
            self.aug_t = albumentations.Compose([
                            A.transforms.HorizontalFlip(p=0.5),
                            A.geometric.transforms.ShiftScaleRotate(shift_limit=0.1,
                                                            scale_limit=0.2,
                                                            rotate_limit=15,
                                                            border_mode=cv2.BORDER_CONSTANT,
                                                            value=(255, 255, 255),
                                                            mask_value=(255, 255, 255),
                                                            p=0.5),])    
    @staticmethod
    def _is_image(filename):
        img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.bpm', '.svg', '.tiff']
        return any(filename.lower().endswith(extension) for extension in img_extensions)

    @staticmethod
    def preprocess(img):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5), inplace=True)
            ])
        img = image_transform(img)
        return img

    def __getitem__(self, i):
        source = Image.open(self.images[i]).convert('RGB')
        tactile_path= self.images[i].replace("source", "tactile").replace("s_","t_").replace("png", "tiff")
        tactile = Image.open(tactile_path).convert('RGB')
        
        if self.mode == 'train' and self.aug:
            augmented = self.aug_t(image=np.array(source), mask=np.array(tactile))
            aug_img_pil = Image.fromarray(augmented['image'])
            img_tensor = self.preprocess(aug_img_pil)
            aug_mask_pil = Image.fromarray(augmented['mask'])
            mask_tensor = self.preprocess(aug_mask_pil)
        else:
            img_tensor = self.preprocess(source)
            mask_tensor = self.preprocess(tactile)
        
            # jitter_amount = 0.02
            # img = transforms.ColorJitter(jitter_amount, jitter_amount, jitter_amount, jitter_amount)(img)
            # erased = transforms.RandomErasing(p=0.5, scale=(0.01, 0.08), ratio=(0.5, 2.0), value=(255,255,255))(img=img, sketch=sketch)


        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.images)
