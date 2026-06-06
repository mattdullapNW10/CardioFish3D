"""Reusable 3D building blocks for the generator and discriminator.

Shape convention everywhere: ``(B, C, D, H, W)`` where ``D`` is the axial (Z)
axis, matching the ZYX ordering used across the rest of the codebase.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_norm(norm: str, num_channels: int) -> nn.Module:
    """Return a normalization layer by name (``instance`` | ``group`` | ``none``)."""
    norm = (norm or "none").lower()
    if norm == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm == "group":
        # 8 groups is a robust default; fall back to fewer if channels are small.
        groups = min(8, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm '{norm}' (expected instance|group|none)")


def maybe_spectral_norm(conv: nn.Module, enabled: bool) -> nn.Module:
    """Wrap a conv layer in spectral norm when ``enabled`` (used by the discriminator)."""
    return nn.utils.spectral_norm(conv) if enabled else conv


class ResBlock3D(nn.Module):
    """Residual block: Conv-Norm-Act-Conv-Norm + identity, then activation.

    Spatial size and channel count are preserved.
    """

    def __init__(self, channels: int, norm: str = "instance", slope: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(norm, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(norm, channels)
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class DownBlock(nn.Module):
    """Strided-conv downsampling (channel doubling) followed by a residual block.

    ``stride`` may be anisotropic, e.g. ``(1, 2, 2)`` to leave the thin Z axis
    untouched while halving Y and X.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "instance",
        stride: int | tuple[int, int, int] = 2,
        slope: float = 0.2,
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
            make_norm(norm, out_ch),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.res = ResBlock3D(out_ch, norm=norm, slope=slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Trilinear upsample -> channel-reducing conv -> concat skip -> fuse -> residual.

    ``scale`` mirrors the matching :class:`DownBlock` stride so spatial sizes line
    up for the skip concatenation.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        norm: str = "instance",
        scale: int | tuple[int, int, int] = 2,
        slope: float = 0.2,
    ):
        super().__init__()
        self.scale = scale
        self.reduce = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            make_norm(norm, out_ch),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            make_norm(norm, out_ch),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.res = ResBlock3D(out_ch, norm=norm, slope=slope)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.res(x)
