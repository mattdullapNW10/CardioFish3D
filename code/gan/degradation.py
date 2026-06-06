"""Optional synthetic re-blur augmentation (OFF by default).

Real (raw -> deconvolved) pairs are the primary training signal. This module is
only used when ``reblur_augment`` is enabled in the config, to manufacture extra
(blur, sharp) pairs by forward-blurring a sharp/deconvolved volume with a random
library PSF plus shot + read noise (domain randomization).

All volumes are float32 in ``[0, 1]`` with ZYX ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_library_psfs(psf_library_root: str | Path, methods: list[str] | None = None) -> list[np.ndarray]:
    """Load cached ``.npy`` PSFs from ``outputs/psf_library`` (per_sample/averages)."""
    root = Path(psf_library_root)
    psfs: list[np.ndarray] = []
    for npy in sorted(root.rglob("*.npy")):
        if methods and not any(m in npy.stem for m in methods):
            continue
        try:
            psf = np.load(npy).astype(np.float32)
        except Exception:
            continue
        if psf.ndim == 3:
            s = psf.sum()
            if s > 0:
                psfs.append(psf / s)
    return psfs


def theory_psf(dxy_um: float, dz_um: float, nz: int = 31, nx: int = 63) -> np.ndarray:
    """Generate a single theoretical PSF, reusing psf_library_build if available."""
    try:
        from psf_library_build import OpticalParams, theory_psf_vol  # type: ignore

        psf = theory_psf_vol(dxy_um, dz_um, OpticalParams(), nz=nz, nx=nx, model="gaussian")
    except Exception:
        import psfmodels as psfm  # type: ignore

        psf = psfm.make_psf(nz, nx, dxy=dxy_um, dz=dz_um, NA=0.75, wvl=0.6,
                            model="gaussian", ni=1.0, ni0=1.0, ns=1.33,
                            oversample_factor=3, normalize=True)
    psf = np.asarray(psf, dtype=np.float32)
    s = psf.sum()
    return psf / s if s > 0 else psf


def reblur(
    sharp: np.ndarray,
    psf: np.ndarray,
    gain: float = 1000.0,
    read_sigma: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Forward model: ``blur = conv(sharp, psf)`` + Poisson(shot) + Gaussian(read).

    Args:
        sharp: clean volume in [0, 1], ZYX.
        psf: normalized PSF (sums to 1), ZYX.
        gain: photons-per-unit-intensity for the Poisson noise (higher = less noise).
        read_sigma: Gaussian read-noise std in [0, 1] units.
    """
    rng = rng or np.random.default_rng()
    blurred = fftconvolve(sharp, psf, mode="same")
    blurred = np.clip(blurred, 0.0, None)
    noisy = rng.poisson(blurred * gain).astype(np.float32) / gain
    noisy = noisy + rng.normal(0.0, read_sigma, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)
