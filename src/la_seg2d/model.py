"""Standard 2D U-Net. Same depth, widths, and normalization as the 3D U-Net.

Each resolution has two 3x3 convolutions. Downsampling is max-pooling, upsampling
is a stride-2 transposed convolution, and skip tensors are center-aligned when
an odd size makes the decoder miss the encoder grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Fp32InstanceNorm2d(nn.InstanceNorm2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(dtype=x.dtype)


def _norm(name: str, channels: int) -> nn.Module:
    if name == "instance":
        return _Fp32InstanceNorm2d(channels, affine=True)
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _match_to(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Center-crop or pad `source` so its spatial size equals `reference`."""
    for axis in range(2, 4):
        delta = source.shape[axis] - reference.shape[axis]
        if delta > 0:
            before = delta // 2
            after = delta - before
            slc = [slice(None)] * 4
            slc[axis] = slice(before, source.shape[axis] - after)
            source = source[tuple(slc)]
        elif delta < 0:
            missing = -delta
            before = missing // 2
            after = missing - before
            # F.pad is last-axis first: (W_before, W_after, H_before, H_after).
            pads = [0, 0, 0, 0]
            pair = {3: 0, 2: 2}[axis]
            pads[pair] = before
            pads[pair + 1] = after
            source = nn.functional.pad(source, pads)
    if source.shape[2:] != reference.shape[2:]:
        raise RuntimeError(f"alignment failed: {source.shape} vs {reference.shape}")
    return source


class UNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: list[int], norm: str):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"expected 5 resolution channels to match the 3D U-Net, got {channels}")
        self.channels = list(channels)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for width in channels[:-1]:
            self.encoders.append(ConvBlock(prev, width, norm))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            prev = width
        self.bottleneck = ConvBlock(prev, channels[-1], norm)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for width in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(prev, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width, norm))
            prev = width
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
