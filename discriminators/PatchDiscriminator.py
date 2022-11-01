import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channel=3):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(self, in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers[-1].register_forward_hook(self.add_intermediate_output)
            return layers

        self.model = nn.Sequential(
            *discriminator_block(self, in_channels + out_channel, 64, normalization=False),
            *discriminator_block(self, 64, 128),
            *discriminator_block(self, 128, 256),
            *discriminator_block(self, 256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input     
        self.intermediate_outputs = []
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

    def add_intermediate_output(self, lay, inp, outp):
        self.intermediate_outputs.append(torch.autograd.Variable(outp.data, requires_grad=False))
		
    def get_intermediate_output(self):
        return self.intermediate_outputs[:4]

