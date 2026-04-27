# ultralytics_custom/nn/modules/bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNNode(nn.Module):
    def __init__(self, channels, num_inputs=2, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(
            torch.ones(num_inputs, dtype=torch.float32),
            requires_grad=True
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        w = F.relu(self.weights)
        w = w / (w.sum() + self.epsilon)
        out = sum(w[i] * inputs[i] for i in range(len(inputs)))
        return self.conv(out)


class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Top-down nodes
        self.p4_td  = BiFPNNode(channels, num_inputs=2)
        self.p3_td  = BiFPNNode(channels, num_inputs=2)
        # Bottom-up nodes
        self.p4_out = BiFPNNode(channels, num_inputs=3)
        self.p5_out = BiFPNNode(channels, num_inputs=2)
        # Channel projection
        self.p3_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.p4_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.p5_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def _up(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def _down(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

    def forward(self, features):
        p3, p4, p5 = features

        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5 = self.p5_proj(p5)

        # ── Top-down ─────────────────────────────
        p4_td = self.p4_td([self._up(p5, p4.shape[2:]), p4])
        p3_out = self.p3_td([self._up(p4_td, p3.shape[2:]), p3])

        # ── Bottom-up ────────────────────────────
        p4_out = self.p4_out([p4, p4_td, self._down(p3_out)])
        p5_out = self.p5_out([p5, self._down(p4_out)])

        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(channels) for _ in range(num_layers)
        ])

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return features
