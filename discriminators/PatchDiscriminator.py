import torch
import torch.nn as nn
# Defines the PatchGAN discriminator with the specified arguments.
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, use_sigmoid, ndf=64 ):
        super(PatchDiscriminator, self).__init__()
    
        kw = 4
        padw = 1
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf , ndf * 2,kernel_size=kw, stride=2, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf *2, ndf * 4,kernel_size=kw, stride=2, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 4, ndf * 8,kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw),
            
        ]

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)