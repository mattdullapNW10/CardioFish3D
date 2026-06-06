"""3D residual U-Net generator (``ResUNet3D``).

Maps a raw, PSF-blurred nuclei patch to a restored (deconvolved) patch on the
*same* voxel grid. The network predicts a global residual that is added to the
input before a final ``tanh`` -- restoration networks converge faster when they
learn the correction rather than the whole signal.

Input/output are normalized to ``[-1, 1]`` with shape ``(B, 1, D, H, W)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DownBlock, UpBlock, ResBlock3D, make_norm


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        norm: str = "instance",
        num_bottleneck_blocks: int = 2,
        anisotropic_z: bool = False,
        slope: float = 0.2,
    ):
        """
        Args:
            base_channels: channel width after the stem; doubles each downsampling.
            depth: number of down/up samplings (3 -> /8 bottleneck).
            norm: ``instance`` | ``group`` | ``none``.
            anisotropic_z: if True, downsample only Y/X (stride ``(1,2,2)``),
                leaving the thin Z axis intact -- useful for very few z-slices.
        """
        super().__init__()
        self.depth = depth
        stride = (1, 2, 2) if anisotropic_z else 2

        # Stem (full resolution) -> skip0
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(norm, base_channels),
            nn.LeakyReLU(slope, inplace=True),
        )

        # Encoder
        self.downs = nn.ModuleList()
        ch = base_channels
        enc_channels = [base_channels]  # channels of each cached skip (stem first)
        for _ in range(depth):
            self.downs.append(DownBlock(ch, ch * 2, norm=norm, stride=stride, slope=slope))
            ch *= 2
            enc_channels.append(ch)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResBlock3D(ch, norm=norm, slope=slope) for _ in range(num_bottleneck_blocks)]
        )

        # Decoder (consumes skips in reverse: skip at depth-1 .. skip0)
        self.ups = nn.ModuleList()
        for level in range(depth):
            skip_ch = enc_channels[depth - 1 - level]
            out_ch = skip_ch
            self.ups.append(
                UpBlock(ch, skip_ch, out_ch, norm=norm, scale=stride, slope=slope)
            )
            ch = out_ch

        # Head: predict residual
        self.head = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        skips = []
        h = self.stem(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        # skips: [stem, d1, d2, ..., d_depth]; bottleneck output == skips[-1]
        for level, up in enumerate(self.ups):
            skip = skips[self.depth - 1 - level]
            h = up(h, skip)

        residual = self.head(h)
        return torch.tanh(identity + residual)
