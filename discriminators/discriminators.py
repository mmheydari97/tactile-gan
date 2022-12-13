import torch
import torch.nn as nn
from discriminators.PatchDiscriminator import PatchDiscriminator

def create_disc(name, in_nc, out_nc, ndf, use_sigmoid, return_filter):
    if name.lower() == "patch":
        netD = PatchDiscriminator(in_nc, out_nc, ndf=ndf, use_sigmoid=use_sigmoid, return_filters=return_filter)
    else:
        msg = name + " not a valid model"
        raise NameError(msg)  
        
    if torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
        
    return netD
