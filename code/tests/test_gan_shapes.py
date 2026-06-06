"""Shape / wiring tests for the 3D restoration GAN networks."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "gan"))

from models import PatchGAN3D, ResUNet3D  # noqa: E402
from losses import AdversarialLoss, GeneratorLoss  # noqa: E402


def test_generator_preserves_shape():
    gen = ResUNet3D(base_channels=16, depth=3)
    x = torch.randn(2, 1, 32, 128, 128)
    y = gen(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.min() >= -1.0 and y.max() <= 1.0  # tanh range


def test_generator_anisotropic_z():
    gen = ResUNet3D(base_channels=16, depth=3, anisotropic_z=True)
    x = torch.randn(1, 1, 16, 128, 128)
    assert gen(x).shape == x.shape


def test_discriminator_patch_map():
    disc = PatchGAN3D(in_channels=2, base_channels=16, n_layers=3)
    raw = torch.randn(2, 1, 32, 128, 128)
    tgt = torch.randn(2, 1, 32, 128, 128)
    out = disc(raw, tgt)
    assert out.ndim == 5  # (B, 1, d, h, w) patch-logit map
    assert out.shape[0] == 2 and out.shape[1] == 1
    assert out.shape[2:] < raw.shape[2:]  # spatially downsampled


def test_discriminator_spectral_norm():
    disc = PatchGAN3D(base_channels=8, spectral_norm=True)
    out = disc(torch.randn(1, 1, 32, 64, 64), torch.randn(1, 1, 32, 64, 64))
    assert torch.isfinite(out).all()


def test_param_counts_sane():
    gen = ResUNet3D(base_channels=32, depth=3)
    disc = PatchGAN3D(base_channels=64)
    n_g = sum(p.numel() for p in gen.parameters())
    n_d = sum(p.numel() for p in disc.parameters())
    assert 1e6 < n_g < 200e6
    assert 1e5 < n_d < 100e6


def test_losses_run_and_backprop():
    gen = ResUNet3D(base_channels=8, depth=3)
    disc = PatchGAN3D(base_channels=8)
    adv = AdversarialLoss("lsgan")
    g_loss = GeneratorLoss(adv, lambda_l1=100.0, lambda_freq=1.0, lambda_grad=0.5)

    raw = torch.randn(1, 1, 32, 64, 64)
    target = torch.randn(1, 1, 32, 64, 64).clamp(-1, 1)

    fake = gen(raw)
    real_logits = disc(raw, target)
    fake_logits = disc(raw, fake.detach())
    d_loss = adv.discriminator_loss(real_logits, fake_logits)
    d_loss.backward()
    assert torch.isfinite(d_loss)

    fake_logits = disc(raw, fake)
    total, logs = g_loss(fake_logits, fake, target)
    total.backward()
    assert torch.isfinite(total)
    assert {"g_adv", "g_l1", "g_freq", "g_grad", "g_total"} <= set(logs)


@pytest.mark.parametrize("mode", ["lsgan", "hinge"])
def test_adversarial_modes(mode):
    adv = AdversarialLoss(mode)
    real = torch.randn(1, 1, 4, 4, 4)
    fake = torch.randn(1, 1, 4, 4, 4)
    assert torch.isfinite(adv.discriminator_loss(real, fake))
    assert torch.isfinite(adv.generator_loss(fake))
