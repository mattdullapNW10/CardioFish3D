"""Conditional 3D PatchGAN discriminator (``PatchGAN3D``).

Judges whether a (restored) volume is a plausible deconvolution of a given raw
input. It is *conditional*: the raw input is concatenated channel-wise with the
volume under inspection, so the input has ``in_channels = 2`` by default.

Outputs a 5D patch-logit map (no sigmoid) for use with LSGAN/hinge losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import make_norm, maybe_spectral_norm


class PatchGAN3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        n_layers: int = 3,
        norm: str = "instance",
        spectral_norm: bool = False,
        slope: float = 0.2,
    ):
        """
        Args:
            in_channels: 2 = concat(raw_input, sharp_or_fake).
            n_layers: number of strided conv layers (controls receptive field).
            spectral_norm: wrap every conv in spectral norm for stability.
        """
        super().__init__()
        layers: list[nn.Module] = []

        # First layer: no norm.
        layers += [
            maybe_spectral_norm(
                nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
                spectral_norm,
            ),
            nn.LeakyReLU(slope, inplace=True),
        ]

        ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(ch * 2, base_channels * 8)
            layers += [
                maybe_spectral_norm(
                    nn.Conv3d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                    spectral_norm,
                ),
                make_norm(norm, out_ch),
                nn.LeakyReLU(slope, inplace=True),
            ]
            ch = out_ch

        # Penultimate layer: stride 1 (grows receptive field without shrinking too far).
        out_ch = min(ch * 2, base_channels * 8)
        layers += [
            maybe_spectral_norm(
                nn.Conv3d(ch, out_ch, kernel_size=4, stride=1, padding=1, bias=False),
                spectral_norm,
            ),
            make_norm(norm, out_ch),
            nn.LeakyReLU(slope, inplace=True),
        ]
        ch = out_ch

        # Output: 1-channel patch logits, stride 1, no activation.
        layers += [
            maybe_spectral_norm(
                nn.Conv3d(ch, 1, kernel_size=4, stride=1, padding=1), spectral_norm
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            raw: conditioning input ``(B, 1, D, H, W)``.
            target: real or generated volume ``(B, 1, D, H, W)``.
        Returns:
            patch-logit map ``(B, 1, d, h, w)``.
        """
        x = torch.cat([raw, target], dim=1)
        return self.model(x)
