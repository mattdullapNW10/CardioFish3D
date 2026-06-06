"""3D GAN for blind deconvolution / restoration of nuclei volumes.

Trains a conditional GAN (pix2pix-style) that maps a raw, PSF-blurred nuclei
z-stack to its Richardson-Lucy-deconvolved counterpart. At inference the
generator restores volumes feed-forward, needing no PSF and no metadata.
"""

__all__ = [
    "models",
    "losses",
    "dataset",
    "degradation",
]
