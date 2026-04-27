# ultralytics_custom/nn/modules/sobel.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        Gx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
        Gy = torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
        self.register_buffer('Gx', Gx.view(1,1,3,3).repeat(1,3,1,1))
        self.register_buffer('Gy', Gy.view(1,1,3,3).repeat(1,3,1,1))

    def forward(self, x):
        gx = F.conv2d(x, self.Gx, padding=1, groups=1)
        gy = F.conv2d(x, self.Gy, padding=1, groups=1)
        gx = gx.sum(dim=1, keepdim=True)
        gy = gy.sum(dim=1, keepdim=True)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-6)
        return mag


class TextureCNN(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.layer1 = self._block(1,   32)
        self.layer2 = self._block(32,  64)
        self.layer3 = self._block(64, 128)
        self.layer4 = nn.Conv2d(128, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SobelStream(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.sobel       = SobelFilter()
        self.texture_cnn = TextureCNN(out_channels=out_channels)

    def forward(self, x):
        edge_map = self.sobel(x)
        return self.texture_cnn(edge_map)


class StreamFusion(nn.Module):
    def __init__(self, rgb_channels, texture_channels, out_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(rgb_channels + texture_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, texture_feat):
        if rgb_feat.shape[2:] != texture_feat.shape[2:]:
            texture_feat = F.interpolate(
                texture_feat,
                size=rgb_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        combined = torch.cat([rgb_feat, texture_feat], dim=1)
        return self.fusion(combined)
