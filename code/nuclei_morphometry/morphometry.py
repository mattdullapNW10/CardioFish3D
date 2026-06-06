#!/usr/bin/env python3
"""3D nuclei segmentation + geometric morphometry.

Takes a 3D microscopy image, segments the nuclei, counts them, and measures key
geometric properties per nucleus — with everything in physical units (microns)
using the voxel size read from the TIFF (so the anisotropic z axis is handled
correctly).

Pipeline
--------
    load -> pick nuclei channel -> 3D Otsu/Li threshold -> distance-transform
    watershed split -> regionprops -> per-nucleus table + summary (+ optional
    napari label view).

Per-nucleus measurements
------------------------
  volume_um3                 physical volume
  equivalent_diameter_um     diameter of a sphere of equal volume
  centroid_{z,y,x}_um        centre of mass
  extent_{x,y,z}_um          axis-aligned bounding-box size along each axis
  axis_{major,inter,minor}_um  ellipsoid principal-axis lengths (PCA, orientation
                               invariant)
  elongation                 axis_major / axis_minor   (1 = sphere, >1 = cigar/disc)
  flatness                   axis_inter / axis_minor   (>1 = flattened)
  sphericity_pca             axis_minor / axis_major   (1 = isotropic)
  solidity, extent           voxel-based convexity / bbox fill (skimage)

Usage
-----
    python nuclei_morphometry/morphometry.py "<image.tif>" --channel 1
    python nuclei_morphometry/morphometry.py img.tif --voxel 0.685,0.567,0.567 --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import feature, filters, measure, segmentation


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_channel(path: Path, channel: int) -> tuple[np.ndarray, str | None]:
    """Load a 3D ZYX volume for ``channel`` from a TIFF/OME-TIFF/.npy file."""
    if path.suffix.lower() == ".npy":
        arr, axes = np.load(path), None
    else:
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            axes = tif.series[0].axes if tif.series else None
    arr = np.asarray(arr)
    vol = _select_channel(arr, axes, channel)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D ZYX after channel select, got {vol.shape}")
    return vol.astype(np.float32), axes


def _select_channel(image: np.ndarray, axes: str | None, channel: int) -> np.ndarray:
    if image.ndim == 3:
        return image
    if image.ndim == 4:
        if axes and "C" in axes:
            cax = axes.index("C")
        else:
            spatial = set(image.shape[-2:])
            cands = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
            cax = min(cands, key=lambda t: t[1])[0] if cands else 0
        n = image.shape[cax]
        ch = min(channel, n - 1)
        if ch != channel:
            print(f"  [warn] requested channel {channel} but only {n}; using {ch}")
        return np.take(image, ch, axis=cax)
    out = image
    while out.ndim > 3:
        out = out[0]
    return out


def read_voxel_um(path: Path) -> tuple[float, float, float]:
    """(dz, dy, dx) microns from TIFF/ImageJ metadata; (1,1,1) if unknown."""
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


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment_nuclei(
    volume: np.ndarray,
    *,
    voxel_um: tuple[float, float, float],
    sigma_um: float = 0.4,
    threshold: str = "otsu",
    min_volume_um3: float = 5.0,
    min_distance_um: float = 3.0,
    clear_border: bool = False,
) -> np.ndarray:
    """Segment nuclei in 3D and return a labelled volume.

    Smoothing / size / peak-distance parameters are given in **microns** and
    converted to voxels using ``voxel_um`` so behaviour is resolution-aware.
    """
    dz, dy, dx = voxel_um
    sigma = (sigma_um / dz, sigma_um / dy, sigma_um / dx)
    blurred = filters.gaussian(volume, sigma=sigma, preserve_range=True)

    if threshold == "li":
        thr = filters.threshold_li(blurred)
    else:
        thr = filters.threshold_otsu(blurred)
    binary = blurred > thr

    voxel_vol = dz * dy * dx
    min_size = max(1, int(round(min_volume_um3 / voxel_vol)))
    binary = _remove_small(binary, min_size)
    binary = ndi.binary_fill_holes(binary)

    # Distance transform in physical units so the split is isotropic.
    distance = ndi.distance_transform_edt(binary, sampling=voxel_um)
    footprint = _peak_footprint(min_distance_um, voxel_um)
    peaks = feature.peak_local_max(distance, footprint=footprint, labels=binary,
                                   exclude_border=False)
    marker_mask = np.zeros(distance.shape, dtype=bool)
    if peaks.size:
        marker_mask[tuple(peaks.T)] = True
    markers = measure.label(marker_mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    if clear_border:
        labels = segmentation.clear_border(labels)
    return measure.label(labels)  # relabel 1..N contiguously


def _remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components with fewer than ``min_size`` voxels."""
    lbl = measure.label(binary)
    counts = np.bincount(lbl.ravel())
    keep = counts >= min_size
    keep[0] = False  # background
    return keep[lbl]


def _peak_footprint(min_distance_um: float, voxel_um) -> np.ndarray:
    rz = max(1, int(round(min_distance_um / voxel_um[0])))
    ry = max(1, int(round(min_distance_um / voxel_um[1])))
    rx = max(1, int(round(min_distance_um / voxel_um[2])))
    return np.ones((2 * rz + 1, 2 * ry + 1, 2 * rx + 1), dtype=bool)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure_nuclei(labels: np.ndarray, intensity: np.ndarray,
                   voxel_um: tuple[float, float, float]) -> pd.DataFrame:
    """Per-nucleus geometric properties in physical (micron) units."""
    dz, dy, dx = voxel_um
    voxel_vol = dz * dy * dx
    rows = []
    for rp in measure.regionprops(labels, intensity_image=intensity):
        coords = rp.coords.astype(np.float64) * np.array([dz, dy, dx])  # -> um
        major, inter, minor, axis_vec = _principal_axes(coords)
        vol_um3 = rp.area * voxel_vol
        zc, yc, xc = rp.centroid
        minz, miny, minx, maxz, maxy, maxx = rp.bbox
        rows.append({
            "label": rp.label,
            "volume_voxels": int(rp.area),
            "volume_um3": vol_um3,
            "equivalent_diameter_um": (6.0 * vol_um3 / np.pi) ** (1.0 / 3.0),
            "centroid_z_um": zc * dz,
            "centroid_y_um": yc * dy,
            "centroid_x_um": xc * dx,
            "extent_x_um": (maxx - minx) * dx,
            "extent_y_um": (maxy - miny) * dy,
            "extent_z_um": (maxz - minz) * dz,
            "axis_major_um": major,
            "axis_inter_um": inter,
            "axis_minor_um": minor,
            "elongation": (major / minor) if minor > 0 else np.nan,
            "flatness": (inter / minor) if minor > 0 else np.nan,
            "sphericity_pca": (minor / major) if major > 0 else np.nan,
            "major_axis_dz": axis_vec[0],
            "major_axis_dy": axis_vec[1],
            "major_axis_dx": axis_vec[2],
            "solidity": _safe(rp, "solidity"),
            "extent": _safe(rp, "extent"),
            "mean_intensity": _mean_intensity(rp),
        })
    return pd.DataFrame(rows)


def _principal_axes(coords_um: np.ndarray):
    """Ellipsoid principal-axis lengths (um) + major-axis unit vector via PCA.

    Axis length uses the image-moment convention ``4*sqrt(eigenvalue)`` so values
    are comparable to skimage's ``major_axis_length`` but in microns and 3D.
    """
    if coords_um.shape[0] < 2:
        return (0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    centered = coords_um - coords_um.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    order = np.argsort(eigvals)[::-1]              # major -> minor
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    lengths = 4.0 * np.sqrt(eigvals)
    major_vec = eigvecs[:, 0]
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]),
            tuple(float(v) for v in major_vec))


def _mean_intensity(rp):
    val = getattr(rp, "intensity_mean", None)
    if val is None:
        val = getattr(rp, "mean_intensity", None)
    return float(np.mean(val)) if val is not None else np.nan


def _safe(rp, attr):
    try:
        return float(getattr(rp, attr))
    except Exception:
        return np.nan


def summarize(df: pd.DataFrame, voxel_um) -> dict:
    """Aggregate summary statistics over all nuclei."""
    if df.empty:
        return {"n_nuclei": 0, "voxel_um_dz_dy_dx": list(voxel_um)}
    def stat(col):
        s = df[col].dropna()
        return {"mean": float(s.mean()), "median": float(s.median()),
                "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}
    return {
        "n_nuclei": int(len(df)),
        "voxel_um_dz_dy_dx": list(voxel_um),
        "volume_um3": stat("volume_um3"),
        "equivalent_diameter_um": stat("equivalent_diameter_um"),
        "elongation": stat("elongation"),
        "flatness": stat("flatness"),
        "extent_x_um": stat("extent_x_um"),
        "extent_y_um": stat("extent_y_um"),
        "extent_z_um": stat("extent_z_um"),
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("volume_um3", "Volume (um^3)"),
        ("equivalent_diameter_um", "Equivalent diameter (um)"),
        ("elongation", "Elongation (major/minor)"),
        ("flatness", "Flatness (inter/minor)"),
        ("extent_z_um", "Axial extent z (um)"),
        ("extent_x_um", "Lateral extent x (um)"),
    ]
    for axis, (col, title) in zip(ax.ravel(), panels):
        vals = df[col].dropna()
        if len(vals):
            axis.hist(vals, bins=min(30, max(5, len(vals) // 2)), color="#4477aa",
                      edgecolor="white")
            axis.axvline(vals.median(), color="#ee6677", ls="--",
                         label=f"median {vals.median():.2f}")
            axis.legend(fontsize=8)
        axis.set_title(title)
    fig.suptitle(f"Nuclei morphometry (n = {len(df)})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="3D image (TIFF/OME-TIFF/.npy)")
    ap.add_argument("--channel", type=int, default=1,
                    help="nuclei channel index (default 1)")
    ap.add_argument("--voxel", help="override voxel size 'dz,dy,dx' in microns")
    ap.add_argument("--threshold", choices=["otsu", "li"], default="otsu")
    ap.add_argument("--sigma-um", type=float, default=0.4, help="smoothing sigma (um)")
    ap.add_argument("--min-volume-um3", type=float, default=5.0,
                    help="discard objects smaller than this (um^3)")
    ap.add_argument("--min-distance-um", type=float, default=3.0,
                    help="minimum separation between nuclei centres (um)")
    ap.add_argument("--clear-border", action="store_true",
                    help="drop nuclei touching the XY border")
    ap.add_argument("-o", "--outdir", default="morphometry_outputs")
    ap.add_argument("--show", action="store_true", help="open result in napari")
    args = ap.parse_args()

    path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vol, axes = load_channel(path, args.channel)
    if args.voxel:
        voxel = tuple(float(v) for v in args.voxel.split(","))
    else:
        voxel = read_voxel_um(path)
    print(f"Image {path.name}: channel {args.channel}, volume {vol.shape} (Z,Y,X), "
          f"voxel(dz,dy,dx)={tuple(round(v,4) for v in voxel)} um")

    print("Segmenting nuclei ...")
    labels = segment_nuclei(
        vol, voxel_um=voxel, sigma_um=args.sigma_um, threshold=args.threshold,
        min_volume_um3=args.min_volume_um3, min_distance_um=args.min_distance_um,
        clear_border=args.clear_border,
    )
    n = int(labels.max())
    print(f"  -> {n} nuclei segmented")

    df = measure_nuclei(labels, vol, voxel)
    csv_path = outdir / f"{path.stem}_nuclei.csv"
    df.to_csv(csv_path, index=False)

    summary = summarize(df, voxel)
    (outdir / f"{path.stem}_summary.json").write_text(json.dumps(summary, indent=2))

    label_tif = outdir / f"{path.stem}_labels.tif"
    tifffile.imwrite(label_tif, labels.astype(np.uint16), imagej=True,
                     metadata={"spacing": voxel[0], "unit": "um", "axes": "ZYX"})

    if not df.empty:
        plot_distributions(df, outdir / f"{path.stem}_distributions.png")

    # Console summary
    print("\n=== Summary ===")
    print(f"  Count            : {summary['n_nuclei']} nuclei")
    if summary["n_nuclei"]:
        print(f"  Volume (um^3)    : median {summary['volume_um3']['median']:.1f} "
              f"(mean {summary['volume_um3']['mean']:.1f})")
        print(f"  Eq. diameter     : median {summary['equivalent_diameter_um']['median']:.2f} um")
        print(f"  Elongation       : median {summary['elongation']['median']:.2f} "
              f"(major/minor; 1 = round)")
        print(f"  Flatness         : median {summary['flatness']['median']:.2f}")
        print(f"  Extent x/y/z (um): "
              f"{summary['extent_x_um']['median']:.1f} / "
              f"{summary['extent_y_um']['median']:.1f} / "
              f"{summary['extent_z_um']['median']:.1f}  (median)")
    print(f"\nOutputs in {outdir}/:")
    print(f"  {csv_path.name}  |  {path.stem}_summary.json  |  "
          f"{label_tif.name}  |  {path.stem}_distributions.png")

    if args.show:
        import napari
        v = napari.Viewer(title=f"Nuclei — {path.name}", ndisplay=3)
        v.add_image(vol, name="raw", scale=voxel, colormap="gray", blending="additive")
        v.add_labels(labels, name=f"nuclei ({n})", scale=voxel)
        napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
