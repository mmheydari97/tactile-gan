from operator import concat
import os
import json
import numpy as np
from PIL import Image
from PIL.ImageOps import invert
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, ToPILImage
from generators.generators import create_gen
from datasets.datasets import get_dataset
from util import mkdir


class Opt:
     def __init__(self, dictionary):
        for k, v in dictionary.items():
             setattr(self, k, v)

def load_opt(path):
    with open(path) as json_file:
        opt = json.load(json_file)
    
    opt = Opt(opt)
    return opt

def load_model(model_path, opt,device):
    G = create_gen(opt.gen,opt.input_dim,opt.output_dim,multigpu=False)
    G.to(device)
    
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint["gen"], strict=False)
    return G


def load_data(photo_path,opt):
    data = get_dataset(photo_path, opt, mode='test')
    dataset = DataLoader(dataset=data, batch_size=1, shuffle=False,num_workers=4)
    return dataset

def unnormalize(a):
    return a/2 +0.5

def visualize(out):
    ax = invert(ToPILImage()(out[0])).convert("RGB")
    grid_msk = ToPILImage()(out[1])
    content_msk = ToPILImage()(out[2])

    content = np.expand_dims(np.array(content_msk), axis=2)
    grid = np.expand_dims(np.array(grid_msk), axis=2)

    blk = np.zeros((256,256,2), dtype=np.uint8)
    content = np.concatenate((content, blk), axis=2)
    grid = np.concatenate((blk, grid), axis=2)
    content = Image.fromarray(content)
    grid = Image.fromarray(grid)
    
    ax.paste(grid, (0,0), grid_msk)
    ax.paste(content, (0,0), content_msk)
    return ToTensor()(ax)


def concat_images(photo,sketch,output):
    return torch.cat((photo,sketch,output),2)

def save_images(dataset,path):
    for i, batch in enumerate(dataset):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = Gen(real_A.to(device)).cpu()

        a = unnormalize(real_A[0])
        b = unnormalize(real_B[0])
        out = unnormalize(out[0])

        b_img = visualize(b)
        out_img = visualize(out)
        save_image(concat_images(a, b_img, out_img), os.path.join(path,f"{i+1}.png")) 
        b_elements = concat_images(torch.unsqueeze(b[0],0), torch.unsqueeze(b[1],0), torch.unsqueeze(b[2],0))
        out_elements = concat_images(torch.unsqueeze(out[0],0), torch.unsqueeze(out[1],0), torch.unsqueeze(out[2],0))
        save_image(torch.cat((b_elements, out_elements), 1), os.path.join(path,f"{i+1}_elements.png")) 
        
        print(f"file {i+1}.png saved.")

opt_path = os.path.join(os.getcwd(),"models","pix2seg","params.txt")
opt = load_opt(opt_path)
device = torch.device("cuda:0")

model_path = os.path.join(os.getcwd(),"models",opt.folder_load,"final_model.pth")
Gen = load_model(model_path,opt,device)

photo_path_test= os.path.join(os.getcwd(),"data","test","source")
dataset = load_data(photo_path_test,opt)

output_path = os.path.join(os.getcwd(),"Outputs",opt.folder_load)
mkdir(output_path)
save_images(dataset,output_path)

