import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import random
import platform

if platform.system() == 'Windows':
    IMG_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
    ]
else:
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    
    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



class UnpairedDataset(data.Dataset):
    def __init__(self, a_dir, b_dir,name="sketchy",size=256,erase=False):
        super(UnpairedDataset, self).__init__()
        self.size = size
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.name = name
        self.erase = erase

        a_paths = []
        for root, fold , fnames in sorted(os.walk(a_dir)):
             for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    a_paths.append(path)
        self.a_paths = a_paths
        
        b_paths = []
        for root, fold , fnames in sorted(os.walk(b_dir)):
             for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    b_paths.append(path)
        self.b_paths = b_paths
        
        
        
    
    def __getitem__(self, i):
        photo = Image.open(self.a_paths[i]).convert('RGB')
        if photo.size != (self.size,self.size):
            photo = photo.resize((self.size, self.size), Image.BICUBIC)
        photo = transforms.ToTensor()(photo)
        photo = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(photo)
        if self.erase:
            photo = transforms.RandomErasing(p=0.5, scale=(0.01, 0.08), ratio=(0.5, 2.0))(photo)
            
        b_index = random.randint(0, len(self.b_paths) - 1)
        sketch = Image.open(self.b_paths[b_index]).convert(mode="L")
        if sketch.size != (self.size,self.size):
            sketch = sketch.resize((self.size, self.size), Image.BICUBIC)
        sketch = transforms.ToTensor()(sketch)
        sketch = transforms.Normalize((0.5,), (0.5,))(sketch)
        
        sketch_path= self.a_paths[i].replace("photo", "sketch").replace(".jpg","")
        #return the actual sketch of the photo
        pairedsketches = []
        i = 1
        while(True):
            if self.name == "sketchy":
                temppath = sketch_path + "-"+str(i)+".png"
            elif self.name == "aligned":
                temppath = sketch_path + "_0"+str(i)+".png"
            
            if (not os.path.isfile(temppath)):
                break
            ps = Image.open(temppath).convert(mode="L") 
            if ps.size != (self.size,self.size):
                ps = ps.resize((self.size, self.size), Image.BICUBIC)
                
            ps = transforms.ToTensor()(ps)
            ps = transforms.Normalize((0.5,), (0.5,))(ps)
            pairedsketches.append(ps)
            i +=1
        if len(pairedsketches) > 5:
            pairedsketches = pairedsketches[0:5]
        if len(pairedsketches) == 0:
            print("somethings wrong", sketch_path,"~~~",self.image_path[i])
        pairedsketches = torch.cat(pairedsketches, 0) 
        
        return photo,sketch,pairedsketches
    
    def __len__(self):
        return min(len(self.a_paths),len(self.b_paths)) 