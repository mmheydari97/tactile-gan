import torch
import torch.nn as nn
from generators.UNet import UNet
from generators.UNet_plusplus import UNet_plusplus


def create_gen(name, in_nc, out_nc):
    if name.lower() == "unet":
        netG = UNet(in_channels=in_nc, out_channels=out_nc)
    
    elif name.lower() == "unet++": 
        netG = UNet_plusplus(in_channels=in_nc, out_channels=out_nc)
   
    else:
        msg = name + " not a valid model"
        raise NameError(msg)  
        
    #if we are using multiple GPU's:
    if multigpu and torch.cuda.device_count() > 1:
        netG = nn.DataParallel(netG)
    
    return netG