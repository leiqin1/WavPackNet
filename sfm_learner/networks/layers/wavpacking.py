# Copyright 2025 Valeo Brain.  All rights reserved.

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from sfm_learner.networks.layers.common import ResidualConv, Conv2D


class WavPacking(nn.Module):
    """
    2D convolution with GroupNorm and ELU

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self):
        super().__init__()
        self.xfm = DWTForward(J=1, mode='zero', wave="db1")

    def forward(self, x):
        """Runs the Conv2D layer."""
        Yl, Yh = self.xfm(x)
        b, c, coeffs, h, w = Yh[0].shape
        channels = Yh[0].view(b, c*coeffs, h, w)
        Yl = Yl.view(b, c, h, w)
        return Yl, Yh[0][:,:,0,:,:], Yh[0][:,:,1,:,:], Yh[0][:,:,2,:,:]

class WavPackLayerSeparateConcat(nn.Module):
    def __init__(self, in_channels, kernel_size):

        super().__init__()
        self.conv_low_freq = ResidualConv(in_channels, in_channels, stride=1, kernel_size=kernel_size)
        self.conv_h1 = ResidualConv(in_channels, in_channels, stride=1, kernel_size=kernel_size)
        self.conv_h2 = ResidualConv(in_channels, in_channels, stride=1, kernel_size=kernel_size)
        self.conv_h3 = ResidualConv(in_channels, in_channels, stride=1, kernel_size=kernel_size)

        self.conv_all = ResidualConv(4*in_channels, in_channels, stride=1, kernel_size=kernel_size)

        self.pack = WavPacking()

    def forward(self, x):
        low_frequency, high_frequency_1, high_frequency_2, high_frequency_3 = self.pack(x)

        low_frequency_featuremap = self.conv_low_freq(low_frequency)
        high_frequency_1_featuremap = self.conv_h1(high_frequency_1)
        high_frequency_2_featuremap = self.conv_h2(high_frequency_2)
        high_frequency_3_featuremap = self.conv_h3(high_frequency_3)

        x = torch.cat([low_frequency_featuremap, high_frequency_1_featuremap, high_frequency_2_featuremap, high_frequency_3_featuremap], axis=1)

        x_learned_represenation = self.conv_all(x)

        return x_learned_represenation

class WavUnPacking(nn.Module):
    """
    2D convolution with GroupNorm and ELU

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self):
        super().__init__()
        self.ixfm = DWTInverse(mode='zero', wave="db1")

    def forward(self, x):
        """Runs the Conv2D layer."""
        yh = []
        b, c, h, w = x.shape
        yl = x.narrow(1, 0, c//4)
        yh.append(x.narrow(1, c//4, c//4).view(b, c//4, 1, h, w))
        yh.append(x.narrow(1, 2*c//4, c//4).view(b, c//4, 1, h, w))
        yh.append(x.narrow(1, 3*c//4, c//4).view(b, c//4, 1, h, w))
        Yh = torch.cat(yh, axis=2)

        x_up = self.ixfm((yl, [Yh]))
        return x_up

class WavUnpackLayer(nn.Module):
    """
    Packing layer with 2d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        """
        Initializes a PackLayerConv2d object.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels , out_channels* 4 //d, kernel_size, 1)
        self.unpack = WavUnPacking()
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the PackLayerConv2d layer."""

        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d , h, w)

        x = self.unpack(x)

        return x

class WavUnpack2DLayer(nn.Module):
    """
    UnPacking layer with 2D convolutions. Takes a [B,C,H,W] tensor, convolves it to produce [B,4*C,H,W] and then
    unpacks it into [B,C,2*H,2*W].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        """
        Initializes a PackLayerConv2d object.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels , out_channels* 4, kernel_size, 1)
        self.unpack = WavUnPacking()

    def forward(self, x):
        """Runs the PackLayerConv2d layer."""

        x = self.conv(x)
        x = self.unpack(x)

        return x