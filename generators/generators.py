import torch
import torch.nn as nn
from generators.ResnetGen import ResnetGenerator
from generators.UNet import UNet, UNetNoConnects
from generators.UNet_plusplus import UNet_plusplus


def create_gen(name,in_nc,out_nc, gen_filters,normlayer,dropout=False,resblocks=9, multigpu=True):
    if name in ["UNET","unet","U-net","U-Net","UNet"]: 
        netG = UNet(input_dim=in_nc, output_dim=out_nc, num_filter=gen_filters, norm=normlayer)
    
    elif name in ["UNet-no-skips"]:
        netG = UNetNoConnects(input_dim=in_nc, output_dim=out_nc, num_filter=gen_filters, norm=normlayer)
    
    elif name in ["Resnet","resnet","res-net"]:
        netG = ResnetGenerator(input_nc=in_nc, output_nc=out_nc, ngf=gen_filters,
                              use_dropout=dropout, n_blocks =resblocks, padding_type='reflect')
        
    elif name in ["UNet++"]: 
        netG = UNet_plusplus(input_dim=in_nc, output_dim=out_nc, num_filter=gen_filters, norm=normlayer)

    else:
        msg = name + " not a valid model"
        raise NameError(msg)  
        
    #if we are using multiple GPU's:
    if multigpu and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        netG = nn.DataParallel(netG)
    
    return netG