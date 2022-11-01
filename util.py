import random
import numpy as np
import torch
from torch.nn import init
import os

torch.manual_seed(21)
torch.cuda.manual_seed_all(21)
random.seed(21)
np.random.seed(21)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def perceptual_loss(real_features, fake_features, mode='normal', loss_type='l1', weights=[1, 1, 1, 1]):
    if mode in ['normal', 'gram']:
        pass
    else:
        raise ValueError('mode must be normal or gram')
    if loss_type == 'l1':
        lfunc = torch.nn.L1Loss()
    elif loss_type == 'l2':
        lfunc = torch.nn.MSELoss()
    else:
        raise ValueError('loss_type must be l1 or l2')
    if len(weights) != 4:
        raise ValueError('weights must be a list of 4 numbers')
    weights = np.array(weights)/np.sum(weights)
    loss = 0.0
    for i in range(4):
        # for feature comparison
        if mode == 'normal':
            lo = lfunc(real_features[i], fake_features[i])
            

        # for style comparison
        elif mode == 'gram':
            act_real = real_features[i].reshape(real_features[i].shape[0], real_features[i].shape[1], -1)
            act_fake = fake_features[i].reshape(fake_features[i].shape[0], fake_features[i].shape[1], -1)
            gram_real = act_real @ act_real.permute(0, 2, 1)
            gram_fake = act_fake @ act_fake.permute(0, 2, 1)
            lo = lfunc(gram_real, gram_fake)
        loss += lo * weights[i]
    return loss

