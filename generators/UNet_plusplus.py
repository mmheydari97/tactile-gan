import torch
import torch.nn as nn


class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel=3,stride=1, padding=1):
        super(ConvBlock,self).__init__()
        
        blocks = [nn.Conv2d(in_channels=in_size, out_channels= out_size, kernel_size=kernel, stride=stride, padding=padding,bias=False),
                  nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
                  nn.ReLU(True),
                  
                  nn.Conv2d(in_channels=out_size, out_channels= out_size, kernel_size=3, stride=1, padding=1,bias=False),
                  nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
                  nn.ReLU(True),
                 ]
        
        self.layer = nn.Sequential(*blocks)

    def forward(self, input):
        return self.layer(input)
    

class UNet_plusplus(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, num_filter=64, norm="instance"):
        super(UNet_plusplus, self).__init__()
#         use_bias = True if norm == "instance" else False
        self.Upsample = nn.Upsample(scale_factor=2)
        self.Downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv0_0 = ConvBlock(3,num_filter)
        self.conv1_0 = ConvBlock(num_filter,num_filter*2)
        self.conv2_0 = ConvBlock(num_filter*2,num_filter*4)
        self.conv3_0 = ConvBlock(num_filter*4,num_filter*8)
        self.conv4_0 = ConvBlock(num_filter*8,num_filter*16)
        
        self.conv0_1 = ConvBlock(num_filter+num_filter*2,num_filter)
        self.conv1_1 = ConvBlock(num_filter*2 + num_filter*4,num_filter*2)
        self.conv2_1 = ConvBlock(num_filter*8+num_filter*4, num_filter*4)
        self.conv3_1 = ConvBlock(num_filter*16 + num_filter*8, num_filter*8)
        
        self.conv0_2 = ConvBlock(num_filter+num_filter+num_filter*2,num_filter)
        self.conv1_2 = ConvBlock(num_filter*2+num_filter*2 + num_filter*4,num_filter*2)
        self.conv2_2 = ConvBlock(num_filter*8+num_filter*4+num_filter*4, num_filter*4)
        
        self.conv0_3 = ConvBlock(num_filter*3+num_filter*2,num_filter)
        self.conv1_3 = ConvBlock(num_filter*2*3+ num_filter*4,num_filter*2)
        
        self.conv0_4 = ConvBlock(num_filter*4+num_filter*2,num_filter)    

        self.downfeature = FeatureMapBlock(num_filter, output_dim)     
        
    def forward(self,input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.Downsample(x0_0))
        x2_0 = self.conv2_0(self.Downsample(x1_0))
        x3_0 = self.conv3_0(self.Downsample(x2_0))
        x4_0 = self.conv4_0(self.Downsample(x3_0))
        
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Upsample(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Upsample(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Upsample(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Upsample(x4_0)], 1))
        
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Upsample(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Upsample(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Upsample(x3_1)], 1))
        
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Upsample(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Upsample(x2_2)], 1))
        
        x0_4  = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Upsample(x1_3)], 1))
        out = self.downfeature(x0_4)
        out = nn.Tanh()(out)
        return out