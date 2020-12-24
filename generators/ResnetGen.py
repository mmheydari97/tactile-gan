import torch
import torch.nn as nn
import functools

# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        
        model = [Inconv(input_nc, ngf),
                     Down(ngf, ngf * 2),
                     Down(ngf * 2, ngf * 4)]

        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type, use_dropout=use_dropout)]
        
        model += [Up(ngf * 4, ngf * 2),
                  Up(ngf * 2, ngf),
                  Outconv(ngf, output_nc)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)



class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch,ks=7,rfpd = 3):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(rfpd),
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=0,
                      bias=False),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x
    
class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch,ks=7,rfpd = 3):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(rfpd),
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=0,bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x



class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, use_dropout):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout)

    def build_conv_block(self, dim, padding_type, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
                       nn.InstanceNorm2d(dim, affine=True, track_running_stats=False),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
                       nn.InstanceNorm2d(dim, affine=True, track_running_stats=False)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




