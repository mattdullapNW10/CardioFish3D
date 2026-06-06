"""Loss functions for the 3D restoration GAN.

Generator objective::

    L_G = lambda_adv * adv + lambda_l1 * L1(fake, target)
          + lambda_freq * FFT-magnitude-L1 + lambda_grad * gradient-L1

Discriminator objective is the adversarial term only (LSGAN or hinge).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """LSGAN (MSE) or hinge adversarial loss over patch-logit maps."""

    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        mode = mode.lower()
        if mode not in ("lsgan", "hinge"):
            raise ValueError(f"Unknown adversarial mode '{mode}' (lsgan|hinge)")
        self.mode = mode

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        if self.mode == "lsgan":
            real = F.mse_loss(real_logits, torch.ones_like(real_logits))
            fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
            return 0.5 * (real + fake)
        # hinge
        real = F.relu(1.0 - real_logits).mean()
        fake = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (real + fake)

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "lsgan":
            return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
        # hinge
        return -fake_logits.mean()


def frequency_loss(fake: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between 3D FFT magnitude spectra.

    Encourages recovery of high-frequency detail removed by the PSF.
    """
    fake_mag = torch.fft.fftn(fake, dim=(-3, -2, -1)).abs()
    target_mag = torch.fft.fftn(target, dim=(-3, -2, -1)).abs()
    return F.l1_loss(fake_mag, target_mag)


def gradient_loss(fake: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of finite-difference gradients along Z, Y, X -- preserves edges."""
    loss = 0.0
    for dim in (-3, -2, -1):
        gf = torch.diff(fake, dim=dim)
        gt = torch.diff(target, dim=dim)
        loss = loss + F.l1_loss(gf, gt)
    return loss / 3.0


class GeneratorLoss(nn.Module):
    """Weighted sum of adversarial + reconstruction terms."""

    def __init__(
        self,
        adv: AdversarialLoss,
        lambda_adv: float = 1.0,
        lambda_l1: float = 100.0,
        lambda_freq: float = 1.0,
        lambda_grad: float = 0.0,
    ):
        super().__init__()
        self.adv = adv
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_grad = lambda_grad

    def forward(
        self,
        fake_logits: torch.Tensor,
        fake: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        adv = self.adv.generator_loss(fake_logits)
        l1 = F.l1_loss(fake, target)
        total = self.lambda_adv * adv + self.lambda_l1 * l1
        logs = {"g_adv": float(adv.detach()), "g_l1": float(l1.detach())}

        if self.lambda_freq > 0:
            freq = frequency_loss(fake, target)
            total = total + self.lambda_freq * freq
            logs["g_freq"] = float(freq.detach())
        if self.lambda_grad > 0:
            grad = gradient_loss(fake, target)
            total = total + self.lambda_grad * grad
            logs["g_grad"] = float(grad.detach())

        logs["g_total"] = float(total.detach())
        return total, logs
