"""Standard 3D U-Net: four downsamplings, two 3x3x3 convolutions per level.

InstanceNorm3d (or GroupNorm) is used instead of BatchNorm because the
training batch is small. Under mixed precision the normalization itself runs
in float32 so its statistics stay stable. Skip connections are center-aligned
when an upsampling step does not land on the encoder resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Fp32InstanceNorm3d(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(dtype=x.dtype)


def _norm(name: str, channels: int) -> nn.Module:
    if name == "instance":
        return _Fp32InstanceNorm3d(channels, affine=True)
    if name == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unsupported norm {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _match_to(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Center-crop or pad `source` so its spatial size equals `reference`."""
    for axis in range(2, 5):
        delta = source.shape[axis] - reference.shape[axis]
        if delta > 0:
            before = delta // 2
            after = delta - before
            slc = [slice(None)] * 5
            slc[axis] = slice(before, source.shape[axis] - after)
            source = source[tuple(slc)]
        elif delta < 0:
            missing = -delta
            before = missing // 2
            after = missing - before
            # F.pad order is the last axis first: (W_before, W_after, H_before, H_after, D_before, D_after).
            pads = [0, 0, 0, 0, 0, 0]
            pair = {4: 0, 3: 2, 2: 4}[axis]
            pads[pair] = before
            pads[pair + 1] = after
            source = nn.functional.pad(source, pads)
    if source.shape[2:] != reference.shape[2:]:
        raise RuntimeError(f"alignment failed: {source.shape} vs {reference.shape}")
    return source


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: list[int], norm: str):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"expected 5 resolution channels, got {channels}")
        self.channels = list(channels)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for width in channels[:-1]:
            self.encoders.append(ConvBlock(prev, width, norm))
            self.pools.append(nn.MaxPool3d(kernel_size=2))
            prev = width
        self.bottleneck = ConvBlock(prev, channels[-1], norm)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for width in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose3d(prev, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width, norm))
            prev = width
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = _match_to(x, skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        return self.head(x)
