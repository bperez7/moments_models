# https://github.com/irhumshafkat/R2Plus1D-PyTorch

import math

import torch.nn as nn
from torch.nn.modules.utils import _triple
import random

class TemporalAvgPool(nn.Module):
    def __init__(self, ks=3, stride=1, padding=1):
        super(TemporalAvgPool, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=ks, stride=stride, padding=padding)

    def forward(self, x):
        x = x.transpose(2,3).transpose(3,4).contiguous()
        b, c, h, w, t = x.size()
        x = x.view(b, c*h*w, t)
        x = self.pool(x)
        t_a = x.size()[-1]
        x = x.view(b, c, h, w, t_a)
        x = x.transpose(3,4).transpose(2,3).contiguous()
        return x


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, tprob=None):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5

        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # intermed_channels = out_channels

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.point = nn.Sequential(
            TemporalAvgPool(ks=temporal_kernel_size[0], stride=temporal_stride[0], padding=temporal_padding[0]),
            nn.Conv3d(intermed_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        )
        self.tprob = tprob

    def forward(self, x):
        temporal = True
        if self.tprob is not None:
            if self.tprob == 1:
                temporal = True
            elif self.tprob == 0:
                temporal = False
            else:
                temporal = (random.random() < self.tprob)
        x = self.relu(self.bn(self.spatial_conv(x)))
        if temporal:
            x = self.temporal_conv(x)
        else:
            x = self.point(x)
        return x
