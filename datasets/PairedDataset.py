import os
import numpy as np
from PIL import Image #, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
# import torch
# import cv2
import albumentations as A



class PairedDataset(Dataset):
    def __init__(self, img_dir, size=256, mode='train', aug=False, target='rgb'):
        super(PairedDataset, self).__init__()
        self.img_dir = img_dir
        self.size = size
        self.mode = mode
        self.aug = aug
        self.target = target

        images = []
        for root, _ , fnames in sorted(os.walk(self.img_dir)):
             for fname in fnames:
                if self._is_image(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        self.images = images

        if aug:
            # mask_value = (255,255,255) if self.target=='rgb' else (0,0,0)
            self.aug_t = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent=0.1,     
                    scale=(0.8, 1.2),          
                    rotate=(-15, 15),        
                    fit_output=False,
                    # mode=cv2.BORDER_CONSTANT,
                    # cval=(255, 255, 255),    # uncomment if you want to fill with white
                    # cval_mask=mask_value,    # uncomment if you want to use a specific mask fill
                    p=0.5
                ),
            ])
    @staticmethod
    def _is_image(filename):
        img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.svg', '.tiff']
        return any(filename.lower().endswith(extension) for extension in img_extensions)


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
        tactile_path = self.images[i].replace("source", "tactile").replace("s_", "t_").replace(".png",".tiff").rsplit(".",1)

        if self.target == 'rgb':
            try:
                tactile = np.array(Image.open(f"{tactile_path[0]}.{tactile_path[1]}").convert('RGB'))
            except FileNotFoundError:
                print(f"File {tactile_path} does not exist")
        else:
            try:
                tactile_axes = np.array(Image.open(f"{tactile_path[0]}_axes.{tactile_path[1]}").convert(mode="L"))
                tactile_grid = np.array(Image.open(f"{tactile_path[0]}_grids.{tactile_path[1]}").convert(mode="L"))
                tactile_content = np.array(Image.open(f"{tactile_path[0]}_content.{tactile_path[1]}").convert(mode="L"))
                tactile = np.concatenate([np.expand_dims(tactile_axes,2), np.expand_dims(tactile_grid,2), np.expand_dims(tactile_content,2)], 2)
            except FileNotFoundError:
                print(f"At least one missing component at {tactile_path}")

        if self.mode=='train' and self.aug:
            augmented = self.aug_t(image=np.array(source), mask=tactile)
            aug_img_pil = Image.fromarray(augmented['image'])
            aug_msk_pil = Image.fromarray(augmented['mask'])
            # apply pixel-wise transformation
            img_tensor = self.preprocess(aug_img_pil)
            mask_tensor = transforms.ToTensor()(aug_msk_pil)

        else:
            img_tensor = self.preprocess(source)
            mask_tensor = transforms.ToTensor()(tactile)

        return img_tensor, mask_tensor
        
    def __len__(self):
        return len(self.images)
