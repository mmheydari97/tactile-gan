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

class ConvDown(nn.Module):
    def __init__(self, in_size, out_size, kernel=4,stride=2, padding=1):
        super(ConvDown,self).__init__()
        
        blocks = [nn.Conv2d(in_channels=in_size, out_channels= out_size, kernel_size=kernel, stride=stride, padding=padding,bias=False),
                  nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
                  nn.ReLU(True),
                  
                  nn.Conv2d(in_channels=out_size, out_channels= out_size, kernel_size=3, stride=1, padding=1,bias=False),
                  nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
                  nn.ReLU(True)
                 ]
        
        self.layer = nn.Sequential(*blocks)

    def forward(self, input):
        return self.layer(input)
    
    
class DeconvUp(nn.Module):
    def __init__(self, in_size, out_size, kernel=4,stride=2, padding=1):
        super(DeconvUp,self).__init__()
        blocks = [
            nn.ConvTranspose2d(in_size, out_size, kernel, stride, padding,bias=False),
            nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=out_size, out_channels= out_size, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
            nn.ReLU(True),
        ]        
        self.layer = nn.Sequential(*blocks)

    def forward(self, input):
        return self.layer(input)
    

    
class UNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, num_filter=64, norm="instance"):
        super(UNet, self).__init__()
        #256x256x3
        self.conv1 = ConvDown(in_size = input_dim, out_size=num_filter) #128x128x64
        self.conv2 = ConvDown(in_size= num_filter, out_size= num_filter*2) #64x64x128
        self.conv3 = ConvDown(in_size = num_filter *2, out_size= num_filter*4)#32
        self.conv4 = ConvDown(in_size = num_filter *4, out_size= num_filter*8)#16
        self.conv5 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#8
        self.conv6 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#4
        self.conv7 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#2
#         self.conv8 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#1
        
#         self.deconv1 = DeconvUp(in_size=num_filter*8, out_size=num_filter *8) #2x512x512
        self.deconv2 = DeconvUp(num_filter * 8 , num_filter * 8) #4x4x512
        self.deconv3 = DeconvUp(num_filter * 8 * 2, num_filter * 8,) #8x8x512
        self.deconv4 = DeconvUp(num_filter * 8 * 2, num_filter * 8) #16x16x512
        self.deconv5 = DeconvUp(num_filter * 8 * 2, num_filter * 4) #32x32x256
        self.deconv6 = DeconvUp(num_filter * 4 * 2, num_filter * 2) #64x64x128
        self.deconv7 = DeconvUp(num_filter * 2 * 2, num_filter)     #128*128* 64
        self.deconv8 = DeconvUp(num_filter * 2,num_filter)          #256*256 * 64
        self.downfeature = FeatureMapBlock(num_filter, output_dim)     #256*256*1
        
    def forward(self,input):
        c1 = self.conv1(input)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
#         c8 = self.conv8(c7)
        
#         d1 = self.deconv1(c8)

        d2 = self.deconv2(c7)
        d3 = self.deconv3((torch.cat([d2, c6], 1)))
        d4 = self.deconv4((torch.cat([d3, c5], 1)))
        d5 = self.deconv5((torch.cat([d4, c4], 1)))
        d6 = self.deconv6((torch.cat([d5, c3], 1)))
        d7 = self.deconv7((torch.cat([d6, c2], 1)))
        d8 = self.deconv8((torch.cat([d7, c1], 1)))
        
        out = self.downfeature(d8)
        out = nn.Tanh()(out)
        return out
    
    
class UNetNoConnects(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, num_filter=64, norm="instance"):
        super(UNetNoConnects, self).__init__()
        use_bias = True if norm == "instance" else False 
        #256x256x3
        self.conv1 = ConvDown(in_size = input_dim, out_size=num_filter) #128x128x64
        self.conv2 = ConvDown(in_size= num_filter, out_size= num_filter*2) #64x64x128
        self.conv3 = ConvDown(in_size = num_filter *2, out_size= num_filter*4)#32
        self.conv4 = ConvDown(in_size = num_filter *4, out_size= num_filter*8)#16
        self.conv5 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#8
        self.conv6 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#4
        self.conv7 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#2
#         self.conv8 = ConvDown(in_size = num_filter *8, out_size= num_filter*8)#1
        
#         self.deconv1 = DeconvUp(in_size=num_filter*8, out_size=num_filter *8) #2x512x512
        self.deconv2 = DeconvUp(num_filter * 8 , num_filter * 8) #4x4x512
        self.deconv3 = DeconvUp(num_filter * 8 , num_filter * 8) #8x8x512
        self.deconv4 = DeconvUp(num_filter * 8 , num_filter * 8) #16x16x512
        self.deconv5 = DeconvUp(num_filter * 8 , num_filter * 4) #32x32x256
        self.deconv6 = DeconvUp(num_filter * 4 , num_filter * 2) #64x64x128
        self.deconv7 = DeconvUp(num_filter * 2, num_filter)     #128*128* 64
        self.deconv8 = DeconvUp(num_filter,num_filter,)          #256*256 * 64
        self.downfeature = FeatureMapBlock(num_filter, output_dim)     #256*256*1
        
    def forward(self,input):
        c1 = self.conv1(input)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
#         c8 = self.conv8(c7)
        
#         d1 = self.deconv1(c8)

        d2 = self.deconv2(c7)
        d3 = self.deconv3(d2)
        d4 = self.deconv4(d3)
        d5 = self.deconv5(d4)
        d6 = self.deconv6(d5)
        d7 = self.deconv7(d6)
        d8 = self.deconv8(d7)
        
        out = self.downfeature(d8)
        out = nn.Tanh()(out)
        return out

