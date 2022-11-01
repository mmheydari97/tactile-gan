import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.2),

        nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
        nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.2)
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        #256x256x3
        self.down1 = UNetDown(in_channels, 64) #128x128x64
        self.down2 = UNetDown(64, 128) #64x64x128
        self.down3 = UNetDown(128, 256) #32x32x256
        self.down4 = UNetDown(256, 512) #16x16x512
        self.down5 = UNetDown(512, 512) #8x8x512
        self.down6 = UNetDown(512, 512) #4x4x512
        self.down7 = UNetDown(512, 512) #2x2x512
        # self.down8 = UNetDown(512, 512)

        # self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(512, 512) #4x4x512
        self.up3 = UNetUp(1024, 512) #8x8x512
        self.up4 = UNetUp(1024, 512) #16x16x512
        self.up5 = UNetUp(1024, 256) #32x32x256
        self.up6 = UNetUp(512, 128) #64x64x128
        self.up7 = UNetUp(256, 64) #128x128x64
        self.up8 = UNetUp(128, 64) #256x256x64
        self.final = nn.Conv2d(64, out_channels, 1) #256x256x3


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        # d8 = self.down8(d7)
        
        # u1 = self.up1(d8)
        u2 = self.up2(d7)
        u3 = self.up3((torch.cat([u2, d6], 1)))
        u4 = self.up4((torch.cat([u3, d5], 1)))
        u5 = self.up5((torch.cat([u4, d4], 1)))
        u6 = self.up6((torch.cat([u5, d3], 1)))
        u7 = self.up7((torch.cat([u6, d2], 1)))
        u8 = self.up8((torch.cat([u7, d1], 1)))

        return self.final(u8)
