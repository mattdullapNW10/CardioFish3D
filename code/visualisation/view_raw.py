#!/usr/bin/env python3
"""Visualise the raw 3D image of a given file in napari.

Standalone, single purpose: load one microscopy file and open it in a 3D napari
window. Multi-channel stacks are shown as separate additively-blended layers,
scaled by the physical voxel size read from the TIFF metadata.

Usage
-----
    python visualisation/view_raw.py "path/to/stack.tif"
    python visualisation/view_raw.py stack.tif --channel 1          # one channel
    python visualisation/view_raw.py stack.tif --voxel 0.685,0.568,0.568
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

COLORMAPS = ["green", "magenta", "cyan", "yellow", "red", "bop blue"]


def to_czyx(image: np.ndarray, axes: str | None) -> np.ndarray:
    """Normalise any layout to (C, Z, Y, X); 3D input becomes (1, Z, Y, X)."""
    if image.ndim == 3:
        return image[None]
    if image.ndim == 4:
        if axes and set("CZYX") <= set(axes):
            return np.transpose(image, [axes.index(a) for a in "CZYX"])
        spatial = set(image.shape[-2:])
        cands = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
        cax = min(cands, key=lambda t: t[1])[0] if cands else 0
        return np.moveaxis(image, cax, 0)
    if image.ndim == 5:  # e.g. TZCYX -> first time point
        return to_czyx(image[0], axes[1:] if axes else None)
    raise ValueError(f"Unsupported image ndim={image.ndim}, shape {image.shape}")


def read_voxel_um(path: Path) -> tuple[float, float, float]:
    """Best-effort (dz, dy, dx) microns from TIFF/ImageJ metadata, else 1,1,1."""
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            dx = dy = 1.0
            xr = tags.get("XResolution")
            if xr is not None and xr.value[0]:
                dx = float(xr.value[1]) / float(xr.value[0])
            yr = tags.get("YResolution")
            if yr is not None and yr.value[0]:
                dy = float(yr.value[1]) / float(yr.value[0])
            ij = tif.imagej_metadata or {}
            dz = float(ij["spacing"]) if ij.get("spacing") is not None else dx
        return (dz, dy or dx, dx)
    except Exception:
        return (1.0, 1.0, 1.0)


def main() -> int:
    DEFAULT_TIF = (
        Path(__file__).parent.parent
        / "data" / "raw" / "cmlc2_lifeactXnuclear" / "48hpf"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
    )

    ap = argparse.ArgumentParser(description="Visualise the raw image of a given file in napari.")
    ap.add_argument("image", nargs="?", default=str(DEFAULT_TIF), help="path to a TIFF / OME-TIFF / .npy 3D(+C) image")
    ap.add_argument("--channel", type=int, default=None, help="show only this channel index")
    ap.add_argument("--voxel", help="override voxel size as 'dz,dy,dx' in microns")
    args = ap.parse_args()

    path = Path(args.image)
    if path.suffix.lower() == ".npy":
        arr, axes = np.load(path), None
    else:
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            axes = tif.series[0].axes if tif.series else None
    czyx = to_czyx(np.asarray(arr), axes).astype(np.float32)

    if args.channel is not None:
        czyx = czyx[args.channel:args.channel + 1]

    if args.voxel:
        dz, dy, dx = (float(v) for v in args.voxel.split(","))
        voxel = (dz, dy, dx)
    else:
        voxel = read_voxel_um(path)

    n_ch = czyx.shape[0]
    print(f"{path.name}: {n_ch} channel(s), volume(C,Z,Y,X)={czyx.shape}, "
          f"voxel(dz,dy,dx)={voxel}")

    import napari

    viewer = napari.Viewer(title=f"RAW — {path.name}", ndisplay=3)
    if n_ch > 1:
        viewer.add_image(
            czyx, channel_axis=0,
            name=[f"ch{c}" for c in range(n_ch)],
            colormap=[COLORMAPS[c % len(COLORMAPS)] for c in range(n_ch)],
            scale=voxel, blending="additive", rendering="mip",
        )
    else:
        viewer.add_image(czyx[0], name="raw", colormap="gray",
                         scale=voxel, rendering="mip")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
