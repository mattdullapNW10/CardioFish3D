"""Network architectures for the 3D restoration GAN."""

from .generator import ResUNet3D
from .discriminator import PatchGAN3D

__all__ = ["ResUNet3D", "PatchGAN3D"]
