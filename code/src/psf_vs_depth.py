"""
Estimate *effective* PSF breadth vs imaging depth from 3D TIFF stacks under data/raw.

This does not replace bead-based calibration. It detects bright puncta (nuclear/channel
spots), fits Gaussian models in-plane (XY) and along Z at each landmark, and reports
median lateral / axial extent (FWHM) as a function of Z position (µm from stack start).

Run:
  python src/psf_vs_depth.py
  python src/psf_vs_depth.py --raw-root path/to/stacks --channel 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.filters import gaussian
import tifffile

from microscopy_io import extract_channel, read_voxel_sizes_um


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "data/raw"


def gaussian_2d(
    xy: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    bg: float,
    yo: float,
    xo: float,
    sy: float,
    sx: float,
) -> np.ndarray:
    yy, xx = xy
    return bg + amplitude * np.exp(-0.5 * (((yy - yo) / sy) ** 2 + ((xx - xo) / sx) ** 2))


def fit_gaussian_xy_profile(patch: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    patch = patch.astype(np.float64, copy=False)
    pg = gaussian(patch, sigma=0.4, preserve_range=True)
    bg = float(np.percentile(pg, 5))
    pp = pg - bg
    amp_est = float(np.max(pp))
    if amp_est <= 1e-9:
        return None, None
    h, w = pp.shape
    yy, xx = np.indices((h, w))
    mass = pp.sum()
    if mass <= 1e-9:
        return None, None
    yo_est = float((yy * pp).sum() / mass)
    xo_est = float((xx * pp).sum() / mass)
    var_y = float(((yy - yo_est) ** 2 * pp).sum() / mass)
    var_x = float(((xx - xo_est) ** 2 * pp).sum() / mass)
    sy_est = max(np.sqrt(max(var_y, 1e-6)), 0.35)
    sx_est = max(np.sqrt(max(var_x, 1e-6)), 0.35)

    p0 = np.array([amp_est, bg, yo_est, xo_est, sy_est, sx_est], dtype=np.float64)

    bounds_lo = [-np.inf, -np.inf, -0.5, -0.5, 0.2, 0.2]
    bounds_hi = [np.inf, np.inf, h - 0.501, w - 0.501, h / 2, w / 2]
    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (yy.ravel(), xx.ravel()),
            pg.ravel(),
            p0=p0,
            bounds=(bounds_lo, bounds_hi),
            maxfev=5000,
        )
        sigma_xy = float(0.5 * (abs(popt[4]) + abs(popt[5])))
    except (RuntimeError, ValueError):
        return None, None

    lateral_fwhm = 2.0 * np.sqrt(2 * np.log(2)) * sigma_xy
    return sigma_xy, float(lateral_fwhm)


def gaussian_z_line(z_indices: np.ndarray, amp: float, bg: float, z0: float, sz: float) -> np.ndarray:
    return bg + amp * np.exp(-0.5 * ((z_indices - z0) / sz) ** 2)


def fit_gaussian_axial_profile(z_vals: np.ndarray, profile: np.ndarray) -> tuple[float | None, float | None]:
    prof = profile.astype(np.float64, copy=False)
    bg = float(np.percentile(prof, 10))
    p = prof - bg
    if float(np.max(p)) <= 1e-12:
        return None, None
    z0_est = float(z_vals[np.argmax(p)])
    mass = float(p.sum())
    if mass <= 0:
        return None, None
    z_cent = float((z_vals * p).sum() / mass)
    vz = float(((z_vals - z_cent) ** 2 * p).sum() / mass)
    sz_est = max(np.sqrt(max(vz, 1e-6)), 0.35)
    amp_est = float(np.max(p))

    bounds_lo = [0.0, -np.inf, z_vals.min() - 1.0, 0.25]
    bounds_hi = [np.inf, np.inf, z_vals.max() + 1.0, len(z_vals)]
    try:
        popt, _ = curve_fit(
            gaussian_z_line,
            z_vals.astype(np.float64),
            prof.astype(np.float64),
            p0=[amp_est, bg, z0_est, sz_est],
            bounds=(bounds_lo, bounds_hi),
            maxfev=5000,
        )
        sz_fit = abs(float(popt[3]))
    except (RuntimeError, ValueError):
        return None, None

    ax_fwhm = 2.0 * np.sqrt(2 * np.log(2)) * sz_fit
    return sz_fit, float(ax_fwhm)


def find_peak_coords(
    volume: np.ndarray,
    percentile: float = 99.2,
    min_distance: int = 6,
) -> np.ndarray:
    """3D intensity peaks → Nx3 array (z, y, x)."""
    v = volume.astype(np.float64)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    scaled = ((v - vmin) / max(vmax - vmin, 1e-12)) if vmax > vmin else v
    sm = gaussian(scaled, sigma=(0.5, 1.0, 1.0), preserve_range=True)
    thresh = np.percentile(sm, percentile)
    coords = peak_local_max(
        sm,
        min_distance=min_distance,
        threshold_abs=float(thresh),
        exclude_border=True,
    )
    return coords.astype(int)


def analyze_stack(
    tif_path: Path,
    raw_root: Path,
    channel: int,
    patch_half: int,
    axial_half_slices: int,
    peak_percentile: float,
    min_peak_distance: int,
) -> pd.DataFrame:
    dxy, dz = read_voxel_sizes_um(tif_path)
    with tifffile.TiffFile(tif_path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None

    volume = extract_channel(image, axes, target_channel_index=channel)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX volume after channel extract, got {volume.shape}")

    Z, _, _ = volume.shape
    coords = find_peak_coords(
        volume,
        percentile=peak_percentile,
        min_distance=min_peak_distance,
    )

    rows: list[dict] = []

    hz = axial_half_slices
    for zi, yi, xi in coords:
        if (
            yi < patch_half + 2
            or xi < patch_half + 2
            or yi >= volume.shape[1] - patch_half - 2
            or xi >= volume.shape[2] - patch_half - 2
            or zi < hz + 1
            or zi >= Z - hz - 1
        ):
            continue

        slab = volume[zi]
        ys = slice(yi - patch_half, yi + patch_half + 1)
        xs = slice(xi - patch_half, xi + patch_half + 1)
        patch = slab[ys, xs]

        sigma_lat_px, fwhm_lat_px = fit_gaussian_xy_profile(patch)
        if sigma_lat_px is None:
            continue

        zs = slice(zi - hz, zi + hz + 1)
        col = volume[zs, yi, xi].astype(np.float64)
        zz = np.arange(zi - hz, zi + hz + 1, dtype=np.float64)
        sigma_ax_slices, fwhm_ax_slices = fit_gaussian_axial_profile(zz, col)
        if sigma_ax_slices is None:
            continue

        z_um = float(zi * dz)
        try:
            rel_stack = str(tif_path.resolve().relative_to(raw_root.resolve()))
        except ValueError:
            rel_stack = tif_path.name
        rows.append(
            {
                "stack": rel_stack,
                "path": str(tif_path.resolve()),
                "z_slice": zi,
                "z_um": z_um,
                "y": yi,
                "x": xi,
                "dxy_um": dxy,
                "dz_um": dz,
                "sigma_lateral_um": float(sigma_lat_px * dxy),
                "fwhm_lateral_um": float(fwhm_lat_px * dxy),
                "sigma_axial_um": float(sigma_ax_slices * dz),
                "fwhm_axial_um": float(fwhm_ax_slices * dz),
            }
        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.insert(0, "stack_name", tif_path.stem)
    return df


def iter_tifs(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    return sorted({*raw_root.rglob("*.tif"), *raw_root.rglob("*.tiff"), *raw_root.rglob("*.TIFF")})


def plot_results(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        print("No measurements to plot (empty dataframe).")
        return

    fig, axes_arr = plt.subplots(1, 2, figsize=(12, 4.5))

    stacks = sorted(df["stack_name"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(stacks), 1)))

    for ax_idx, metric in enumerate(["fwhm_lateral_um", "fwhm_axial_um"]):
        ax = axes_arr[ax_idx]
        for i, nm in enumerate(stacks):
            sub = df[df["stack_name"] == nm]
            ax.scatter(sub["z_um"], sub[metric], s=6, alpha=0.35, color=colors[i % 10], label=nm[:40])

        df_bin = df.copy()
        df_bin["depth_bin_um"] = (df_bin["z_um"] // 2.0) * 2.0
        g = df_bin.groupby("depth_bin_um", sort=True)[metric].median().reset_index()

        ax.plot(g["depth_bin_um"], g[metric], color="black", lw=2, label="Median (all, 2 µm bins)")

        title = (
            "Lateral FWHM (XY effective blur)" if metric.startswith("fwhm_lateral") else "Axial FWHM (Z effective blur)"
        )
        ax.set_title(title)
        ax.set_xlabel("Depth (µm along Z, slice index × dz)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)

    axes_arr[0].legend(fontsize=6, ncol=2, loc="upper left")
    axes_arr[1].legend([], [], frameon=False)

    fig.suptitle("Effective PSF vs depth — bright puncta fit (proxy, not bead PSF)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PSF-vs-depth proxy from puncta Gaussian fits.")
    ap.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW,
        help=f"Recursive search root for TIFF stacks (default: {DEFAULT_RAW})",
    )
    ap.add_argument("--channel", type=int, default=1, help="Channel index for 4D stacks (default: 1)")
    ap.add_argument("--patch-half", type=int, default=7, help="Half-size of XY patch around each peak")
    ap.add_argument("--axial-half", type=int, default=10, help="± slices for axial Gaussian fit around peak slice")
    ap.add_argument("--peak-percentile", type=float, default=99.2)
    ap.add_argument("--min-peak-distance", type=int, default=6)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "psf_vs_depth_outputs",
        help="Directory for CSV and PNG outputs",
    )
    args = ap.parse_args(argv)

    raw_abs = args.raw_root.resolve()
    tifs = iter_tifs(raw_abs)
    if not tifs:
        print(f"No .tif/.tiff files found under {args.raw_root}")
        print("Place hyperstacks under data/raw (any subfolder).")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []

    print(f"Found {len(tifs)} TIFF file(s). Fitting puncta...")
    for p in tifs:
        rp = (
            str(p.relative_to(raw_abs))
            if p.is_relative_to(raw_abs)
            else str(p.resolve())
        )
        print(f"  {rp}")
        try:
            dfi = analyze_stack(
                p,
                raw_abs,
                channel=args.channel,
                patch_half=args.patch_half,
                axial_half_slices=args.axial_half,
                peak_percentile=args.peak_percentile,
                min_peak_distance=args.min_peak_distance,
            )
        except Exception as e:
            print(f"    skip: {e}")
            continue
        if dfi.empty:
            print("    (no fits)")
        else:
            print(f"    {len(dfi)} landmark fits")
        all_dfs.append(dfi)

    out_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    csv_path = args.output_dir / "psf_landmark_fits.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote measurements: {csv_path}")

    plot_results(out_df, args.output_dir / "psf_vs_depth.png")

    if not out_df.empty:
        cor_lat = float(out_df[["z_um", "fwhm_lateral_um"]].corr().iloc[0, 1])
        cor_ax = float(out_df[["z_um", "fwhm_axial_um"]].corr().iloc[0, 1])
        print("\nPearson correlation (approximate linear trend vs depth; interpret with care):")
        print(f"  z_um vs lateral FWHM: {cor_lat:.4f}")
        print(f"  z_um vs axial FWHM:   {cor_ax:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
