"""Compact 2.5D U-Net (3-channel prev/cur/next in, 1-channel lesion logits out).

Standard encoder/decoder with skip connections and BatchNorm. At 640x640 with
base 32 features it trains comfortably in <8 GB VRAM, and the skip connections
preserve the fine spatial detail that tiny lesions need.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    # GroupNorm (not BatchNorm): batch=6 with rare foreground makes BatchNorm's
    # running stats unstable, so eval predictions collapse to an empty mask even
    # while training loss keeps falling. GroupNorm has no batch-statistics
    # train/eval mismatch.
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoders = nn.ModuleList()
        previous = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(previous, feature))
            previous = feature
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, 2, stride=2))
            self.decoders.append(DoubleConv(feature * 2, feature))
        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for index, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skips[-(index + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)
