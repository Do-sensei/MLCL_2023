# './utils/DCGAN.py'
import os
import torch
import torch.nn as nn

# TODO: Make the Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, nc=3, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            # Fill this in # Transposed Convolution with stride 2 and padding 1
            # Fill this in # Batch Normalization
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            # Fill this in # Transposed Convolution with stride 2 and padding 1
            # Fill this in # Batch Normalization
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            # Fill this in # Transposed Convolution with stride 2 and padding 1
            # Fill this in # Batch Normalization
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            # Fill this in # Transposed Convolution with stride 2 and padding 1
            # Fill this in # Hyperbolic Tangent
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    
# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            # Fill this in # Convolution with stride 2 and padding 1
            # Fill this in # Batch Normalization
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            # Fill this in # Convolution with stride 2 and padding 1
            # Fill this in # Batch Normalization
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            # Fill this in # Convolution with stride 2 and padding 1
            # Fill this in # Batch Normalization
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            # Fill this in # Convolution with stride 1 and padding 0
            # Fill this in # Sigmoid
        )

    def forward(self, input):
        return self.main(input)