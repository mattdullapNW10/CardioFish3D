"""Interactive 3-window napari view: raw volume, deconvolved volume, and PSF.

Multi-channel aware: all channels (e.g. lifeact + nuclear) are deconvolved and
shown as separate, additively-blended napari layers so distinct anatomical
structures stay visible. This is an *evaluation* convenience (it reuses the
minimal Richardson-Lucy utility in :mod:`psfselect.reblur`), not a deconvolution
pipeline.

Three independent napari windows are opened:
  1. the before / raw volume (one layer per channel),
  2. the deconvolved volume (one layer per channel),
  3. a 3D visualisation of the PSF(s) used.

Each window uses the physical voxel size as its ``scale`` so axial anisotropy is
rendered faithfully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import render_psf
from .io_utils import LOG, read_voxel_um_from_tiff  # noqa: F401 (re-exported for cli)
from .parameters import PSFParams
from .reblur import richardson_lucy

# Default per-channel colormaps (additive blending). Green first because the
# lifeact-EGFP / membrane channel is usually channel 0 in these stacks.
DEFAULT_COLORMAPS = ["green", "magenta", "cyan", "yellow", "red", "bop blue"]


def load_channels(path: str | Path, channels: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    """Load a volume as ``(C, Z, Y, X)`` float32, selecting ``channels``.

    ``channels=None`` loads every channel. A single-channel / 3D file yields
    ``C=1``. Returns ``(array, channel_indices_used)``.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        axes = None
    else:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            axes = tif.series[0].axes if tif.series else None
    cxyz = _to_czyx(np.asarray(arr), axes)
    n = cxyz.shape[0]
    idx = list(range(n)) if channels is None else [c for c in channels if 0 <= c < n]
    if not idx:
        raise ValueError(f"No valid channels in {channels} for a {n}-channel image")
    return cxyz[idx].astype(np.float32), idx


def _to_czyx(image: np.ndarray, axes: str | None) -> np.ndarray:
    """Normalise any layout to ``(C, Z, Y, X)``; 3D input becomes ``(1, Z, Y, X)``."""
    if image.ndim == 3:
        return image[None]
    if image.ndim == 4:
        if axes and set("CZYX") <= set(axes):
            order = [axes.index(a) for a in "CZYX"]
            return np.transpose(image, order)
        # No/odd axis metadata: assume the smallest leading axis is channels.
        spatial = set(image.shape[-2:])
        cands = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
        cax = min(cands, key=lambda t: t[1])[0] if cands else 0
        moved = np.moveaxis(image, cax, 0)
        return moved
    if image.ndim == 5:  # e.g. TZCYX -> take first time point
        out = image[0]
        return _to_czyx(out, axes[1:] if axes else None)
    raise ValueError(f"Unsupported image ndim={image.ndim}, shape {image.shape}")


def _fit_psf_grid(params: PSFParams, vol_shape: tuple[int, int, int]) -> PSFParams:
    """Cap PSF grid so the kernel is not larger than the image on any axis."""
    nz, ny, nx = vol_shape
    new_nz = int(min(params.nz, nz if nz % 2 else nz - 1))
    new_nx = int(min(params.nx, ny, nx))
    return params.copy_with(nz=max(new_nz, 9), nx=max(new_nx, 15))


def _center_crop_czyx(vol: np.ndarray, side_xy: int, side_z: int) -> np.ndarray:
    _, nz, ny, nx = vol.shape
    sz = min(side_z, nz) if side_z > 0 else nz
    sy = min(side_xy, ny) if side_xy > 0 else ny
    sx = min(side_xy, nx) if side_xy > 0 else nx
    z0, y0, x0 = (nz - sz) // 2, (ny - sy) // 2, (nx - sx) // 2
    return vol[:, z0:z0 + sz, y0:y0 + sy, x0:x0 + sx]


def deconvolve_multichannel(
    raw_czyx: np.ndarray,
    model: str,
    params: PSFParams,
    *,
    wavelengths_nm: list[float] | None = None,
    backend: str = "auto",
    jar_path=None,
    iters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a PSF per channel and Richardson-Lucy deconvolve each channel.

    ``wavelengths_nm`` (one per channel) lets each channel use its own emission
    wavelength for the PSF; otherwise ``params.wavelength_nm`` is used for all.
    Returns ``(dec_czyx, psf_czyx)`` both as ``(C, Z, Y, X)`` float32.
    """
    n_ch = raw_czyx.shape[0]
    vol_shape = raw_czyx.shape[1:]
    fitted = _fit_psf_grid(params, vol_shape)

    # Render one PSF per distinct wavelength (cached) to avoid redundant work.
    wl = wavelengths_nm or [params.wavelength_nm] * n_ch
    if len(wl) < n_ch:
        wl = wl + [params.wavelength_nm] * (n_ch - len(wl))
    psf_cache: dict[float, np.ndarray] = {}
    dec_list, psf_list = [], []
    for c in range(n_ch):
        w = float(wl[c])
        if w not in psf_cache:
            psf, backend_used, _ = render_psf(model, fitted.copy_with(wavelength_nm=w),
                                              backend=backend, jar_path=jar_path)
            psf_cache[w] = psf
            LOG.info("Channel %d: PSF (%.0f nm) via '%s' backend, shape %s",
                     c, w, backend_used, psf.shape)
        psf = psf_cache[w]
        chan = raw_czyx[c]
        chan_n = chan / max(float(chan.max()), 1e-12)
        LOG.info("Channel %d: %d Richardson-Lucy iterations on %s ...", c, iters, vol_shape)
        dec = richardson_lucy(chan_n, psf, iters=iters)
        dec_list.append((dec / max(float(dec.max()), 1e-12)).astype(np.float32))
        psf_list.append(psf.astype(np.float32))
    return np.stack(dec_list, 0), np.stack(psf_list, 0)


def show_three_windows(
    raw_czyx: np.ndarray,
    dec_czyx: np.ndarray,
    psf_czyx: np.ndarray,
    *,
    voxel_um: tuple[float, float, float],
    psf_voxel_um: tuple[float, float, float] | None = None,
    channel_indices: list[int] | None = None,
    channel_names: list[str] | None = None,
    title_prefix: str = "",
    block: bool = True,
) -> None:
    """Open three independent napari windows; channels become separate layers."""
    import napari

    psf_voxel_um = psf_voxel_um or voxel_um
    pre = (title_prefix + " ") if title_prefix else ""
    n_ch = raw_czyx.shape[0]
    idx = channel_indices or list(range(n_ch))
    cmaps = [DEFAULT_COLORMAPS[i % len(DEFAULT_COLORMAPS)] for i in range(n_ch)]
    names = channel_names or [f"ch{idx[i]}" for i in range(n_ch)]

    def add_multichannel(viewer, data, suffix):
        clims = [_clim(data[c]) for c in range(n_ch)]
        viewer.add_image(
            data,
            channel_axis=0,
            name=[f"{names[c]} {suffix}" for c in range(n_ch)],
            colormap=cmaps,
            scale=voxel_um,
            blending="additive",
            rendering="mip",
            contrast_limits=clims,
        )

    v_raw = napari.Viewer(title=f"{pre}RAW (before)", ndisplay=3)
    add_multichannel(v_raw, raw_czyx, "raw")

    v_dec = napari.Viewer(title=f"{pre}DECONVOLVED", ndisplay=3)
    add_multichannel(v_dec, dec_czyx, "deconv")

    v_psf = napari.Viewer(title=f"{pre}PSF (3D)", ndisplay=3)
    if n_ch > 1 and psf_czyx.shape[0] == n_ch:
        v_psf.add_image(psf_czyx, channel_axis=0,
                        name=[f"{names[c]} PSF" for c in range(n_ch)],
                        colormap=cmaps, scale=psf_voxel_um, blending="additive",
                        rendering="attenuated_mip")
    else:
        v_psf.add_image(psf_czyx[0], name="psf", colormap="inferno",
                        scale=psf_voxel_um, rendering="attenuated_mip")

    if block:
        napari.run()


def _clim(vol: np.ndarray) -> tuple[float, float]:
    lo = float(np.percentile(vol, 1))
    hi = float(np.percentile(vol, 99.5))
    if hi <= lo:
        hi = float(vol.max()) or 1.0
    return (lo, hi)
