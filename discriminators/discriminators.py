import torch
import torch.nn as nn
from discriminators.PatchDiscriminator import PatchDiscriminator

def create_disc(name, in_nc, out_nc, num_filter, return_filter, activation=True, multigpu=False):
    if name.lower() == "patch":
        netD = PatchDiscriminator(in_nc, out_nc, num_filter=num_filter, return_filters=return_filter, activation=activation)
    else:
        raise NameError(f"{name} not a valid model")  
        
    if multigpu and torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
        
    return netD
