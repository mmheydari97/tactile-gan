import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import random
import platform

#from https://github.com/mtli/PhotoSketch/tree/master/data
if platform.system() == 'Windows':
    IMG_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.svg',
    ]
else:
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG', '.svg', '.SVG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    
    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



class PairedDataset(data.Dataset):
    def __init__(self, a_dir,b_dir,name="sketchy",size=256,flip=True,jitter=True,erase=True, colored_sketch=True): #image_dir should be unused for now
        super(PairedDataset, self).__init__()
        self.size = size
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.name = name
        self.flip = flip
        self.jitter = jitter
        self.erase = erase
        self.colored_sketch = colored_sketch
        
#         path = os.path.abspath(os.getcwd())
#         path = os.path.join(path,"train", "photo")
        image_path = []
        image_name = []
        for root, fold , fnames in sorted(os.walk(self.a_dir)):
             for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    image_path.append(path)
                    image_name.append(fname)
        
        self.image_path = image_path
        self.image_name = image_name
    
    def __getitem__(self, i):
        if_flip = self.flip and random.random() < 0.5
        
        img = Image.open(self.image_path[i]).convert('RGB')
        if if_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.jitter:
            jitter_amount = 0.02
            img = transforms.ColorJitter(jitter_amount, jitter_amount, jitter_amount, jitter_amount)(img)
        
        if img.size != (self.size,self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        

        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        if self.erase:
            img = transforms.RandomErasing(p=0.5, scale=(0.01, 0.08), ratio=(0.5, 2.0))(img)
        

        
        sketch_path= self.image_path[i].replace("photo", "sketch").replace(".jpg","")
        
        sketches = []
        j = 1
        temppath = None
        flag = False
        while(True):
            if self.name == "sketchy":
                temppath = sketch_path + "-"+str(j)+".png"
            elif self.name == "aligned":
                temppath = sketch_path + "_0"+str(j)+".png"
            elif self.name == "tactile":
                temppath = sketch_path.replace("s_", "t_")+".jpg"
                flag = True
            if (not os.path.isfile(temppath)):
                break
            sketch = Image.open(temppath) # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            
            if not self.colored_sketch:
                sketch = sketch.convert(mode="L")
            if if_flip:
                sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)
            
            if sketch.size != (self.size,self.size):
                sketch = sketch.resize((self.size, self.size), Image.BICUBIC)
                
            sketch = transforms.ToTensor()(sketch)
            sketch = transforms.Normalize((0.5,), (0.5,))(sketch)
            sketches.append(sketch)
            j +=1
            if flag:
                break
        if len(sketches) > 5:
            sketches = sketches[0:5]
        if len(sketches) == 0:
            print("somethings wrong", sketch_path,"~~~",self.image_path[i])
        sketches = torch.cat(sketches, 0) #stack vs cat might be better
        return img,sketches
    
    def __len__(self):
        return len(self.image_path) 
    






