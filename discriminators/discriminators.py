import torch
import torch.nn as nn

from discriminators.GlobalDiscriminator import GlobalDiscriminator
from discriminators.PatchDiscriminator import PatchDiscriminator

def create_disc(name,dim, use_sigmoid):
    if name == "Global":
        netD = GlobalDiscriminator(dim,use_sigmoid= use_sigmoid)
        
    elif name == "Patch":
        netD = PatchDiscriminator(dim,use_sigmoid= use_sigmoid)
    else:
        msg = name + " not a valid model"
        raise NameError(msg)  
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netD = nn.DataParallel(netD)
        
    return netD