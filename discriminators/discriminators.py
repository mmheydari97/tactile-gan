import torch
import torch.nn as nn

from discriminators.GlobalDiscriminator import GlobalDiscriminator
from discriminators.PatchDiscriminator import PatchDiscriminator

def create_disc(name, in_nc, out_nc):
    if name.lower() == "global":
        netD = GlobalDiscriminator(in_nc, out_nc)
        
    elif name.lower() == "patch":
        netD = PatchDiscriminator()
    else:
        msg = name + " not a valid model"
        raise NameError(msg)  
        
    if torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
        
    return netD