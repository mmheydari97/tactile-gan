import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channel=3, num_filter=64, return_filters=True, activation=True):
        super(PatchDiscriminator, self).__init__()
        self.return_filters = return_filters
        self.kw = 3
        self.padw = 0
        
        def discriminator_block(self, in_filters, out_filters, stride, normalization=True, bias=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=self.kw, stride=stride, padding=self.padw, bias=bias)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True, track_running_stats=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if self.return_filters:
                layers[-1].register_forward_hook(self.add_intermediate_output)
            return layers

        self.model = [
            *discriminator_block(self, in_channels + out_channel, num_filter, stride=2, normalization=False, bias=True),
            *discriminator_block(self, num_filter, num_filter*2, stride=2),
            *discriminator_block(self, num_filter*2, num_filter*4, stride=1),
            *discriminator_block(self, num_filter*4, num_filter*8, stride=1),
            nn.Conv2d(num_filter*8, 1, kernel_size=self.kw, stride=1, padding=self.padw),
        ]
        if activation:
            self.model.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input     
        self.intermediate_outputs = []
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

    def add_intermediate_output(self, lay, inp, outp):
        self.intermediate_outputs.append(outp.detach().clone())
		
    def get_intermediate_output(self):
        return self.intermediate_outputs[:4]

