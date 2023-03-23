import argparse
import os
import json
import re
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image, ImageFilter
from PIL.ImageOps import invert
from tqdm import tqdm

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
    gen = create_gen(opt.gen, opt.input_dim, opt.output_dim, opt.nf, multigpu=False)
    gen.to(device)
    
    checkpoint = torch.load(model_path)
    gen.load_state_dict(checkpoint["gen"], strict=False)
    return gen


def load_data(photo_path,opt, mode='test', shuffle=False):
    data = get_dataset(photo_path, opt, mode=mode)
    dataset = DataLoader(dataset=data, batch_size=1, shuffle=shuffle, num_workers=4)
    return dataset

def load_arrays(path):
    gen_loss = np.load(os.path.join(path, "genloss.npy"))
    disc_loss = np.load(os.path.join(path, "discloss.npy"))
    l1_loss = np.load(os.path.join(path, "l1loss.npy"))
    gp_loss = np.load(os.path.join(path, "gploss.npy"))
    per_loss = np.load(os.path.join(path, "perloss.npy"))
    return {"gen":gen_loss, "disc":disc_loss, "l1":l1_loss, "gp":gp_loss, "per": per_loss}


def unnormalize(a):
    return a/2 +0.5

def visualize(out):
    ax_msk = invert(ToPILImage()(out[0]))
    grid_msk = ToPILImage()(out[1])
    content_msk = ToPILImage()(out[2])
    
    ax = np.expand_dims(np.array(ax_msk), axis=2)
    content = np.expand_dims(np.array(content_msk), axis=2)
    grid = np.expand_dims(np.array(grid_msk), axis=2)

    blk = np.zeros((256,256,2), dtype=np.uint8)
    
    ax = np.concatenate((ax,ax,ax), axis=2)
    content = np.concatenate((content, blk), axis=2)
    grid = np.concatenate((blk, grid), axis=2)
    
    ax = Image.fromarray(ax)
    content = Image.fromarray(content)
    grid = Image.fromarray(grid)
    
    ax.paste(grid, (0,0), grid_msk)
    ax.paste(content, (0,0), content_msk)
    
    return ax


def concat_images(*photos, mode="h"):
    #torch.cat((photo,sketch,output),2)
    if mode=="h":
        res = Image.new(photos[0].mode, (photos[0].width*len(photos),photos[0].height))
        for i in range(len(photos)):
            res.paste(photos[i], (photos[i].width*i,0))
    else:
        res = Image.new(photos[0].mode, (photos[0].width, photos[0].height*len(photos)))
        for i in range(len(photos)):
            res.paste(photos[i], (0, photos[i].height*i))

    return res

def save_plot(loss_dict, opt):
	x = np.array(range(opt.initial_epoch, opt.initial_epoch+opt.total_epochs))
	legends = loss_dict.keys()
	for y in loss_dict.values():
		plt.plot(x,y)
	plt.legend(legends)
	plt.xlabel("iteration")
	plt.ylabel("loss")
	plt.savefig(os.path.join(os.getcwd(),"models",opt.folder_save,"loss.png"))

def eval_pair(real, out):
    o = out.detach().cpu().numpy().flatten()
    r = real.detach().cpu().numpy().flatten()
    threshold = (np.min(r)+np.max(r))/2
    o_bin = np.where(o>= threshold, 1, 0)
    r_bin = np.where(r>= threshold, 1, 0)

    accuracy = np.sum(o_bin==r_bin)/o_bin.shape[0]

    intersection = np.logical_and(o_bin,r_bin)
    union = np.logical_or(o_bin,r_bin)

    jaccard = np.sum(intersection)/np.sum(union)
    dice = 2*np.sum(intersection)/(np.sum(o_bin)+np.sum(r_bin))

    return({"accuracy":accuracy, "dice":dice, "jaccard": jaccard})
   

def print_evaluation(accuracy, dice, jaccard, opt):
    a = f"Pixel Accuracy => min:{np.min(accuracy)}, max:{np.max(accuracy)}, avg:{np.mean(accuracy)}, std:{np.std(accuracy)}\n"
    d = f"Dice Coeff => min:{np.min(dice)}, max:{np.max(dice)}, avg:{np.mean(dice)}, std:{np.std(dice)}\n"
    j = f"Jaccard Index => min:{np.min(jaccard)}, max:{np.max(jaccard)}, avg:{np.mean(jaccard)}, std:{np.std(jaccard)}\n"
    with open(os.path.join(os.getcwd(),"models",opt.folder_save,"eval.txt"), 'w') as f:
        f.writelines([a,d,j])
    print (f"Acc: {np.mean(accuracy)}, IoU: {np.mean(jaccard)}, Dice: {np.mean(dice)}")

def test_model(model, dataset, path, evaluation=False):
    accuracy = []
    jaccard = []
    dice = []

    for i, batch in enumerate(tqdm(dataset)):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = model(real_A.to(device)).cpu()

        a = unnormalize(real_A[0])
        b = real_B[0]
        out = out[0]
        
        if evaluation:
            res = eval_pair(b, out)
            accuracy.append(res["accuracy"])
            dice.append(res["dice"])
            jaccard.append(res["jaccard"])

        if opt.target == 'rgb':
            b_img = ToPILImage()(b)
            out_img = ToPILImage()(out)
        
        else:
            b_img = visualize(b)
            out_img = visualize(out)
            
        out_img.save(os.path.join(path,f"o_{i+1}.png"))
        concat_images(ToPILImage()(a), b_img, out_img).save(os.path.join(path,f"sgt_{i+1}.png"))

        if opt.target != 'rgb':        
            b_elements = concat_images(ToPILImage()(b[0]), ToPILImage()(b[1]), ToPILImage()(b[2]))
            out_elements = concat_images(ToPILImage()(out[0]), ToPILImage()(out[1]), ToPILImage()(out[2]))
            concat_images(b_elements,out_elements, mode="v").save(os.path.join(path,f"elm_{i+1}.png"))
    return accuracy, dice, jaccard

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="pix2obj", help="The folder path including params.txt")
    opt = parser.parse_args()

    opt_path = os.path.join(os.getcwd(),"models", opt.folder.split("/")[-1], "params.txt")
    opt = load_opt(opt_path)
    device = torch.device("cuda:0")

    model_path = os.path.join(os.getcwd(),"models",opt.folder_save,"final_model.pth")
    gen = load_model(model_path,opt,device)

    photo_path_test= os.path.join(os.getcwd(),opt.data,"test","source")
    dataset = load_data(photo_path_test,opt, shuffle=False)

    loss_path = os.path.join(os.getcwd(), "models", opt.folder_save)
    losses = load_arrays(loss_path)
    save_plot(losses, opt)

    output_path = os.path.join(os.getcwd(),"Outputs",opt.folder_save)
    mkdir(output_path)
    
    accuracy, dice, jaccard = test_model(gen, dataset, output_path, evaluation=True)
    if len(accuracy)>0:
        print_evaluation(accuracy, dice, jaccard, opt)
