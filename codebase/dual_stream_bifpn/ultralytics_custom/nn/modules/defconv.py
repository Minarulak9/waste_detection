# ultralytics_custom/nn/modules/defconv.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels,
                         kernel_size, kernel_size)
        )

        # 2 * k^2 offset values (x and y for each point)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # Modulation mask
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self._init_weights()

    def _init_weights(self):
        # Offsets start at zero = behaves like standard conv initially
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        # Mask starts at 0.5 = equal contribution
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 0.5)
        # Standard kaiming init for main weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        offset = self.offset_conv(x)
        mask   = torch.sigmoid(self.mask_conv(x))

        try:
            from torchvision.ops import deform_conv2d
            out = deform_conv2d(
                input   = x,
                offset  = offset,
                weight  = self.weight,
                bias    = self.bias,
                stride  = self.stride,
                padding = self.padding,
                mask    = mask
            )
        except ImportError:
            # Fallback to standard conv if torchvision not available
            print("[DefConv] torchvision not found, using standard conv")
            out = F.conv2d(x, self.weight, self.bias,
                           self.stride, self.padding)
        return out


class DefConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dcn = DeformableConv2d(in_channels, out_channels,
                                    kernel_size, stride, padding)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.dcn(x)))
