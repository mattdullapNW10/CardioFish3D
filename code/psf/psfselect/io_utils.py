"""Lightweight IO helpers: TIFF / OME-TIFF / NumPy volumes, JSON, logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str = "psfselect") -> logging.Logger:
    """Return a module logger configured once with a console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOG = get_logger()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Volume IO
# --------------------------------------------------------------------------- #
def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D volume (ZYX) from TIFF / OME-TIFF / .npy.

    Multi-channel or time data are reduced to a single 3D ZYX volume by taking
    the first non-spatial slice; a warning is logged when that happens.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        vol = np.load(path)
    elif path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            vol = tif.asarray()
            axes = tif.series[0].axes if tif.series else None
        vol = _reduce_to_zyx(vol, axes)
    else:
        raise ValueError(f"Unsupported volume format: {path.suffix}")
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D ZYX volume, got shape {vol.shape}")
    return vol.astype(np.float32, copy=False)


def _reduce_to_zyx(image: np.ndarray, axes: str | None) -> np.ndarray:
    """Collapse a >3D hyperstack to a single ZYX volume."""
    if image.ndim == 3:
        return image
    if image.ndim < 3:
        raise ValueError(f"Volume has too few dimensions: {image.shape}")
    if axes and len(axes) == image.ndim:
        # Drop everything that is not Z/Y/X, taking index 0.
        keep = {"Z", "Y", "X"}
        # Build slices: 0 for non-spatial axes, full slice for spatial.
        sl: list[Any] = []
        for ax in axes:
            sl.append(slice(None) if ax in keep else 0)
        reduced = image[tuple(sl)]
        LOG.warning("Reduced hyperstack axes %s %s -> ZYX %s", axes, image.shape, reduced.shape)
        return reduced
    # No axis metadata: assume leading axes are non-spatial; take index 0 of each.
    reduced = image
    while reduced.ndim > 3:
        reduced = reduced[0]
    LOG.warning("Reduced hyperstack %s -> %s (no axis metadata)", image.shape, reduced.shape)
    return reduced


def save_volume(path: str | Path, vol: np.ndarray, *, voxel_um: tuple[float, float, float] | None = None) -> Path:
    """Save a ZYX (3D) or CZYX (4D, multi-channel) volume as ImageJ TIFF (or .npy)."""
    path = Path(path)
    ensure_dir(path.parent)
    vol = np.asarray(vol, dtype=np.float32)
    if path.suffix.lower() == ".npy":
        np.save(path, vol)
        return path
    import tifffile

    if vol.ndim == 4:
        # ImageJ expects multi-channel hyperstacks in TZCYXS order; emit ZCYX.
        vol = np.transpose(vol, (1, 0, 2, 3))  # CZYX -> ZCYX
        axes = "ZCYX"
    else:
        axes = "ZYX"
    kwargs: dict[str, Any] = {"imagej": True}
    if voxel_um is not None:
        dz, dy, dx = voxel_um
        kwargs["resolution"] = (1.0 / dx, 1.0 / dy)
        kwargs["metadata"] = {"spacing": dz, "unit": "um", "axes": axes}
    tifffile.imwrite(path, vol, **kwargs)
    return path


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_NpEncoder)
    return path


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def read_voxel_um_from_tiff(path: str | Path) -> tuple[float, float, float] | None:
    """Best-effort (dz, dy, dx) in microns from TIFF/ImageJ metadata, else None."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            dx = dy = None
            xr = tags.get("XResolution")
            if xr is not None:
                num, den = xr.value
                if num:
                    dx = float(den) / float(num)
            yr = tags.get("YResolution")
            if yr is not None:
                num, den = yr.value
                if num:
                    dy = float(den) / float(num)
            ij = tif.imagej_metadata or {}
            dz = float(ij["spacing"]) if ij.get("spacing") is not None else None
        if dx is None:
            return None
        dy = dy or dx
        dz = dz or dx
        return (dz, dy, dx)
    except Exception:  # pragma: no cover - metadata best effort
        return None
