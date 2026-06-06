"""Minimal, isolated reblur / deconvolution utilities for PSF *evaluation only*.

This is intentionally small and is NOT a deconvolution pipeline. It exists so the
comparison framework can run an optional data-driven consistency test:

    deconvolve(raw, psf) -> reblur(estimate, psf) -> compare reblur vs raw

A PSF that genuinely matches the image blur should let a few Richardson-Lucy
iterations sharpen the volume and then reblur back close to the original, with a
low residual and a measurable sharpness gain. Everything here works on a cropped
sub-volume to stay cheap.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _pad_psf_to(psf: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Center-pad/crop a PSF to ``shape`` and normalise to unit sum."""
    psf = np.asarray(psf, dtype=np.float64)
    psf = psf / max(psf.sum(), 1e-12)
    out = np.zeros(shape, dtype=np.float64)
    slices_src, slices_dst = [], []
    for s_psf, s_out in zip(psf.shape, shape):
        n = min(s_psf, s_out)
        src_start = (s_psf - n) // 2
        dst_start = (s_out - n) // 2
        slices_src.append(slice(src_start, src_start + n))
        slices_dst.append(slice(dst_start, dst_start + n))
    out[tuple(slices_dst)] = psf[tuple(slices_src)]
    return out


def reblur(volume: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve ``volume`` with ``psf`` (unit-sum) — forward blur model."""
    psf = np.asarray(psf, dtype=np.float64)
    psf = psf / max(psf.sum(), 1e-12)
    return fftconvolve(volume.astype(np.float64), psf, mode="same")


def richardson_lucy(volume: np.ndarray, psf: np.ndarray, iters: int = 8,
                    clip: bool = True) -> np.ndarray:
    """A compact Richardson-Lucy deconvolution (few iterations).

    Kept deliberately simple: no acceleration, no regularisation. Returns an
    estimate the same shape as ``volume``.
    """
    image = volume.astype(np.float64)
    image = np.clip(image, 1e-9, None)
    psf = np.asarray(psf, dtype=np.float64)
    psf = psf / max(psf.sum(), 1e-12)
    psf_mirror = psf[::-1, ::-1, ::-1]
    est = np.full(image.shape, image.mean(), dtype=np.float64)
    for _ in range(max(1, iters)):
        conv = fftconvolve(est, psf, mode="same")
        conv = np.clip(conv, 1e-9, None)
        relative = image / conv
        est *= fftconvolve(relative, psf_mirror, mode="same")
        if clip:
            est = np.clip(est, 0, None)
    return est


def _sharpness(vol: np.ndarray) -> float:
    """Gradient-energy sharpness proxy (higher = sharper)."""
    g = np.gradient(vol.astype(np.float64))
    return float(np.mean(sum(gi ** 2 for gi in g)))


def reblur_consistency(raw: np.ndarray, psf: np.ndarray, *, iters: int = 8,
                       crop: int = 96) -> dict[str, float]:
    """Run deconvolve→reblur and score how consistent ``psf`` is with ``raw``.

    Returns a dict with:
      * ``residual``      — normalised RMSE between reblurred estimate and raw
                            (lower is better),
      * ``sharpness_gain``— sharpness(estimate)/sharpness(raw) (higher is better),
      * ``correlation``   — Pearson r between reblur and raw (higher is better).
    Operates on a centred crop of side ``crop`` for speed.
    """
    raw = np.asarray(raw, dtype=np.float64)
    sub = _center_crop(raw, crop)
    sub_n = sub / max(sub.max(), 1e-12)

    psf_s = _pad_psf_to(psf, sub_n.shape) if psf.shape != sub_n.shape else psf
    est = richardson_lucy(sub_n, psf_s, iters=iters)
    re = reblur(est, psf_s)
    re_n = re / max(re.max(), 1e-12)

    diff = re_n - sub_n
    residual = float(np.sqrt(np.mean(diff ** 2)) / (sub_n.std() + 1e-12))
    gain = _sharpness(est / max(est.max(), 1e-12)) / max(_sharpness(sub_n), 1e-12)
    a, b = re_n.ravel(), sub_n.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"residual": residual, "sharpness_gain": float(gain), "correlation": corr}


def _center_crop(vol: np.ndarray, side: int) -> np.ndarray:
    out = vol
    for ax, n in enumerate(vol.shape):
        s = min(side, n) if ax > 0 else min(side, n)  # crop all axes
        start = (n - s) // 2
        out = np.take(out, range(start, start + s), axis=ax)
    return out
