"""Shared helpers for reading TIFF hyperstacks and voxel sizes."""

from pathlib import Path

import numpy as np
import tifffile


def detect_channel_axis(image: np.ndarray, axes: str | None) -> int | None:
    if image.ndim != 4:
        return None
    if axes and "C" in axes:
        return axes.index("C")
    spatial = set(image.shape[-2:])
    candidates = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
    return min(candidates, key=lambda x: x[1])[0]


def extract_channel(image: np.ndarray, axes: str | None, target_channel_index: int = 1) -> np.ndarray:
    if image.ndim == 4:
        ca = detect_channel_axis(image, axes)
        if ca is None:
            raise ValueError(f"Cannot find channel axis for shape {image.shape}")
        return image.take(target_channel_index, axis=ca)
    if image.ndim == 3:
        return image
    raise ValueError(f"Unsupported dimensions: {image.shape}")


def read_voxel_sizes_um(tif_path: Path) -> tuple[float, float]:
    """Pixel size dxy (µm) and z-step dz (µm) from TIFF / ImageJ metadata."""
    dx = 0.5675
    dz = 0.6849
    with tifffile.TiffFile(tif_path) as tif:
        tags = tif.pages[0].tags
        xr = tags.get("XResolution")
        if xr is not None:
            num, den = xr.value
            if den and num != 0:
                dx = float(den) / float(num)
        ij = tif.imagej_metadata or {}
        if ij.get("spacing") is not None:
            dz = float(ij["spacing"])
    return dx, dz


def load_hyperstack_volume(
    tif_path: Path, channel_index: int = 1
) -> tuple[np.ndarray, str | None, tuple[float, float]]:
    dxy, dz = read_voxel_sizes_um(tif_path)
    with tifffile.TiffFile(tif_path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None
    vol = extract_channel(image, axes, target_channel_index=channel_index)
    if vol.ndim != 3:
        raise ValueError(f"Expected single-channel ZYX volume, got shape {vol.shape}")
    return vol, axes, (dxy, dz)


def iter_raw_tifs(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    return sorted({*raw_root.rglob("*.tif"), *raw_root.rglob("*.tiff"), *raw_root.rglob("*.TIFF")})


def safe_sample_name(stem: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in stem)[:180]
