import os
import json
from sys import platlibdir
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from generators.generators import create_gen
from datasets.datasets import get_dataset
from util import mkdir


class Opt:
     def __init__(self, dictionary):
        for k, v in dictionary.items():
             setattr(self, k, v)

def load_opt(folder_name):
    path = os.path.join(os.getcwd(),"models",folder_name,"params.txt")
    with open(path) as json_file:
        opt = json.load(json_file)
    
    opt = Opt(opt)
    return opt

def load_model(folder_name,model_name, opt,device):
    G = create_gen(model_name,opt.input_dim,opt.output_dim,multigpu=False)
    G.to(device)
    
    checkpoint = torch.load(os.path.join(os.getcwd(),"models",folder_name,model_name))
    G.load_state_dict(checkpoint["gen"], strict=False) #for cyclegan replace 'gen' with 'genAB'
    return G

def load_data(photo_path,opt):
    data = get_dataset(photo_path, opt, mode='test')
    dataset = DataLoader(dataset=data, batch_size=1, shuffle=False,num_workers=4)
    return dataset

def unnormalize(a):
    return a/2 +0.5

def visualize_mask(img):
    palette = {0:[1.,1.,1.], 1:[0,0,0], 2:[0.3,0.3,0.3], 3:[0.1,0.5,0.7]}
    res = np.zeros((3,256,256), np.float16)
    for k, v in palette.items():
        res[:, img == k] = np.array(v).reshape(-1,1)
    return res


def concat_images(photo,sketch,output):
    return np.swapaxes(np.concatenate((photo,sketch,output),1), 0, 2)

def save_images(dataset,path, reduce_channels=True):
    for i, batch in enumerate(dataset):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = Gen(real_A.to(device))[0].cpu()
        a = unnormalize(real_A[0]).numpy()
        b = unnormalize(real_B[0]).numpy()
        
        if reduce_channels:
            b = np.argmax(b, axis=0)
            out = np.argmax(out, axis=0)

            b = visualize_mask(b)
            out = visualize_mask(out)
        
        file_name = str(i) +".png"
        plt.imsave(os.path.join(path,file_name),concat_images(a,b,out)) 
        print(f"file saved at: {os.path.join(path,file_name)}")

folder= "pix2seg"
opt = load_opt(folder)
device = torch.device("cuda:0")
Gen = load_model(folder,opt.gen,opt,device)

photo_path_train = os.path.join(os.getcwd(),"data","test","source")
dataset = load_data(photo_path_train,opt)

path = os.path.join(os.getcwd(),"Outputs",folder)
mkdir(path)
save_images(dataset,path)

