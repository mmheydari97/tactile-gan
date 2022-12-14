import random
import numpy as np
import torch
from torch.nn import init
import torchvision
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


def pan_loss(real_features, fake_features, mode='normal', loss_type='l1', weights=[1, 1, 1, 1]):
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

def gradient_penalty(disc, real_img, real_mask, fake_mask, device, ver=2, type='mixed', constant=1.0, lambda_gp=1.0):
        if lambda_gp > 0.0:
            if type == 'real':
                interpolates = real_mask
            elif type == 'fake':
                interpolates = fake_mask
            elif type == 'mixed':
                alpha = torch.rand(real_mask.size(0), 1, device=device)
                if ver == 2:
                    alpha = (alpha+1)/2
                alpha = alpha.expand(real_mask.shape[0], real_mask.nelement() // real_mask.shape[0]).contiguous().view(*real_mask.shape)
                interpolates = alpha * real_mask + ((1 - alpha) * fake_mask)
            else:
                raise NotImplementedError(f'{type} not implemented')
            interpolates.requires_grad_(True)
            pred = disc(real_img, interpolates)
            gradients = torch.autograd.grad(outputs=pred, inputs=interpolates,
                                            grad_outputs=torch.ones_like(pred).to(device),
                                            create_graph=True,
                                            retain_graph=True)
            gradients = gradients[0].view(real_mask.size(0),-1)                
            res = (((gradients+1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
            
            return res
        else:
            return 0.0


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].cuda().eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].cuda().eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].cuda().eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].cuda().eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], weights=[0.25, 0.25, 0.25, 0.25]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)*weights[i]
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)*weights[i]
        return loss
