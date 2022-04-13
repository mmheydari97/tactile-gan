import os
import json
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

def concat_images(photo,sketch,output):
    return torch.cat((photo,sketch,output),2)

def save_images(dataset,path,reduce_channels=True):
    for i, batch in enumerate(dataset):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = Gen(real_A.to(device))[0].cpu()
        a = unnormalize(real_A[0])
        b = unnormalize(real_B[0])

        if reduce_channels:
            out = torch.unsqueeze(torch.argmax(out, dim=0), dim=0)
            b = torch.unsqueeze(torch.argmax(b, dim=0), dim=0) 
            b = torch.cat((b,b,b),0)
            out = torch.cat((out,out,out),0)
        
        file_name = str(i) +".png"
        save_image(concat_images(a,b,out), os.path.join(path,file_name))
        # save_image(a,os.path.join(path,file_name))

folder= "pix2seg"
opt = load_opt(folder)
device = torch.device("cuda:0")
Gen = load_model(folder,opt.gen,opt,device)

photo_path_train = os.path.join(os.getcwd(),"data","test","source")
dataset = load_data(photo_path_train,opt)

path = os.path.join(os.getcwd(),"Outputs",folder)
mkdir(path)
save_images(dataset,path)

