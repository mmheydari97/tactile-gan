# In[]:


import torch
import torch.nn as nn

import argparse
import os
from math import log10
import json

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


from statistics import mean
  

from torch.nn import init
import functools
import itertools

from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import time
from generators.generators import create_gen
from discriminators.discriminators import create_disc
from datasets.datasets import get_dataset
from util import ImagePool, set_requires_grad,tensor_to_plt,init_weights, mkdir
from Tensorboard_Logger import Logger


# In[]:


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
    G = create_gen(opt.gen,opt.input_dim,opt.output_dim,opt.gen_filters,opt.norm,multigpu=False)
    G.to(device)
    
    checkpoint = torch.load(os.path.join(os.getcwd(),"models",folder_name,model_name))
    G.load_state_dict(checkpoint["gen"]) #for cyclegan replace 'gen' with 'genAB'
    return G

def load_data(photo_path,sketch_path,opt):
    data = get_dataset(photo_path,sketch_path, opt,flip=False,jitter=False,erase= False,colored_s=True)
    dataset = DataLoader(dataset=data, batch_size=1, shuffle=False,num_workers=4)
    return dataset

def unnormalize(a):
    return a/2 +0.5
def onechannel_to_three(a):
    return torch.cat((a,a,a),0)
def concat_images(photo,sketch,output):
    return torch.cat((photo,sketch,output),2)

def save_images(dataset,path):
    for i, batch in enumerate(dataset):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = Gen(real_A.to(device))[0].cpu()


        a = real_A[0]
        a = unnormalize(a)

        b = real_B[0]
        b = unnormalize(b)

        if not colored:
            out = onechannel_to_three(out)
            b = onechannel_to_three(b)
        
        file_name = str(i) +".jpg"
        save_image(concat_images(a,b,out),os.path.join(path,file_name))

# In[]:


folder= "wgan_tactile_unet"
opt = load_opt(folder)
device = torch.device("cuda:0")
Gen = load_model(folder,opt.gen,opt,device)


# In[]:


photo_path_train = os.path.join(os.getcwd(),"data","aligned","test", "photo")
sketch_path_train = os.path.join(os.getcwd(),"data","aligned","test", "sketch")
opt.dataset_name = "aligned"
dataset = load_data(photo_path_train,sketch_path_train,opt)


# In[]:


path = os.path.join(os.getcwd(),"Outputs","UNet++_Test_Images")
mkdir(path)
save_images(dataset,path)




