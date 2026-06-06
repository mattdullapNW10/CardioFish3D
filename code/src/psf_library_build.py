#!/usr/bin/env python3
"""
Build a library of *estimated* / *model-based* 3D PSFs across all TIFF stacks in data/raw.

Theoretical PSFs mirror classes described by EPFL PSF Generator
https://bigwww.epfl.ch/algorithms/psfgenerator/ using the Python ``psfmodels`` package:

  • richards_wolf_vectorial → ``model='vectorial'`` (vectorial diffraction / Richards–Wolf type)
  • gibson_lanni_scalar → ``model='scalar'`` (scalar Gibson–Lanni)
  • gaussian_axial_linear → ``model='gaussian'`` (Gaussian lateral + axial defocus analogue)

Born & Wolf specifically is **not** reimplemented here (use ImageJ/Java PSF Generator for that).

Empirical stacks average normalized 3D crops around bright puncta in each sample (noise proxy —
not bead PSF calibration).

Outputs (default):

  outputs/psf_library/per_sample/*.npy
  outputs/psf_library/averages/*.npy + meta JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import psfmodels as psfm
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian

from microscopy_io import load_hyperstack_volume, iter_raw_tifs, safe_sample_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "data/raw"
DEFAULT_OUT = PROJECT_ROOT / "outputs/psf_library"

THEORY_MODELS_EPFL_NAMES: dict[str, tuple[str, str]] = {
    "richards_wolf_vectorial": ("vectorial", "Richards & Wolf (vectorial, psfmodels)"),
    "gibson_lanni_scalar": ("scalar", "Gibson & Lanni scalar (psfmodels)"),
    "gaussian_axial_linear": ("gaussian", "Gaussian + axial variation (EPFL-style approx.)"),
}
EMPIRICAL_METHOD = ("empirical_puncta_aligned", "Empirical puncta-aligned average (dataset)")


@dataclass
class OpticalParams:
    NA: float = 0.75
    wavelength_um: float = 0.6
    ni: float = 1.0
    ni0: float = 1.0
    ns: float = 1.33
    tg_um: float = 170.0
    tg0_um: float = 170.0
    ng: float = 1.515
    ng0: float = 1.515
    ti0_um: float = 150.0
    oversample: int = 3


def theory_psf_vol(
    dxy_um: float,
    dz_um: float,
    optics: OpticalParams,
    *,
    nz: int,
    nx: int,
    model: str,
) -> np.ndarray:
    """Return ZYX PSF normalized max=1 (psfmodels)."""
    return np.asarray(
        psfm.make_psf(
            nz,
            nx,
            dxy=dxy_um,
            dz=dz_um,
            NA=optics.NA,
            wvl=optics.wavelength_um,
            ns=optics.ns,
            ni=optics.ni,
            ni0=optics.ni0,
            tg=optics.tg_um,
            tg0=optics.tg0_um,
            ng=optics.ng,
            ng0=optics.ng0,
            ti0=optics.ti0_um,
            oversample_factor=optics.oversample,
            normalize=True,
            model=model,
        ),
        dtype=np.float64,
    )


def find_peak_coords_subsampled(
    volume: np.ndarray,
    *,
    z_stride: int,
    percentile: float,
    min_distance: int,
) -> np.ndarray:
    v = volume.astype(np.float64, copy=False)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    scaled = ((v - vmin) / max(vmax - vmin, 1e-12)) if vmax > vmin else v
    sm = gaussian(scaled, sigma=(0.6, 1.1, 1.1), preserve_range=True)
    vz = sm[::z_stride, :, :]
    thr = np.percentile(vz, percentile)
    md = max(5, min_distance // 2)
    coords = peak_local_max(vz, min_distance=md, threshold_abs=float(thr), exclude_border=True)
    if coords.size == 0:
        return coords.astype(int)
    out = coords.copy()
    out[:, 0] = np.clip((out[:, 0] * z_stride).astype(int), 0, volume.shape[0] - 1)
    out[:, 1] = coords[:, 1].astype(int)
    out[:, 2] = coords[:, 2].astype(int)
    return out.astype(int)


def preprocess_patch(crop: np.ndarray) -> np.ndarray:
    crop = crop.astype(np.float64, copy=False)
    bg = float(np.percentile(crop, 12.0))
    p = crop - bg
    p = np.clip(p, 0.0, None)
    m = float(np.max(p))
    if m <= 1e-12:
        return p
    p /= m
    return p


def empirical_psf_from_volume(
    volume: np.ndarray,
    *,
    half_z: int,
    half_xy: int,
    z_stride_peaks: int,
    percentile: float,
    min_peak_distance: int,
    max_patches_per_volume: int,
) -> np.ndarray | None:
    Z, Y, X = volume.shape
    if Z < 2 * half_z + 3 or min(Y, X) < 2 * half_xy + 3:
        return None

    coords = find_peak_coords_subsampled(
        volume,
        z_stride=z_stride_peaks,
        percentile=percentile,
        min_distance=min_peak_distance,
    )
    if coords.size == 0:
        return None

    hz, hxy = half_z, half_xy
    mats: list[np.ndarray] = []

    rng = np.random.default_rng(0)
    if len(coords) > max_patches_per_volume:
        pick = rng.choice(len(coords), size=max_patches_per_volume, replace=False)
        coords = coords[pick]

    for zi, yi, xi in coords:
        if (
            yi < hxy + 1
            or xi < hxy + 1
            or yi >= Y - hxy - 1
            or xi >= X - hxy - 1
            or zi < hz + 1
            or zi >= Z - hz - 1
        ):
            continue
        yz_sl = slice(zi - hz, zi + hz + 1)
        y_sl = slice(yi - hxy, yi + hxy + 1)
        x_sl = slice(xi - hxy, xi + hxy + 1)
        mats.append(preprocess_patch(volume[yz_sl, y_sl, x_sl]))

    if not mats:
        return None

    stack = np.stack(mats, axis=0)
    mean_psf = np.mean(stack, axis=0)
    m = float(np.max(mean_psf))
    if m > 1e-12:
        mean_psf = mean_psf / m
    return mean_psf


def resize_to_reference(vol: np.ndarray, ref_shape: tuple[int, int, int]) -> np.ndarray:
    """Trilinear-ish resampling to matching ZYX shapes for cross-sample averaging."""
    if vol.shape == ref_shape:
        return vol
    fz = ref_shape[0] / vol.shape[0]
    fy = ref_shape[1] / vol.shape[1]
    fx = ref_shape[2] / vol.shape[2]
    return ndi.zoom(vol, zoom=(fz, fy, fx), order=3)


def compute_sample_psfs(
    tif_path: Path,
    *,
    optics: OpticalParams,
    channel: int,
    nz_theory: int,
    nx_theory: int,
    half_z_patch: int,
    half_xy_patch: int,
    z_stride_peaks: int,
    peak_percentile: float,
    min_peak_distance: int,
    max_patches: int,
) -> tuple[dict[str, np.ndarray], float, float]:
    vol, _, (dxy, dz) = load_hyperstack_volume(tif_path, channel_index=channel)
    results: dict[str, np.ndarray] = {}

    for slug in THEORY_MODELS_EPFL_NAMES:
        pmodel = THEORY_MODELS_EPFL_NAMES[slug][0]
        results[slug] = theory_psf_vol(
            dxy,
            dz,
            optics,
            nz=nz_theory,
            nx=nx_theory,
            model=pmodel,
        )

    emp = empirical_psf_from_volume(
        vol,
        half_z=half_z_patch,
        half_xy=half_xy_patch,
        z_stride_peaks=z_stride_peaks,
        percentile=peak_percentile,
        min_peak_distance=min_peak_distance,
        max_patches_per_volume=max_patches,
    )
    if emp is not None:
        results[EMPIRICAL_METHOD[0]] = emp

    return results, dxy, dz


def build_library(
    raw_root: Path,
    output_dir: Path,
    optics: OpticalParams,
    channel: int,
    *,
    nz_theory: int = 31,
    nx_theory: int = 63,
    half_z_patch: int,
    half_xy_patch: int,
    z_stride_peaks: int,
    peak_percentile: float,
    min_peak_distance: int,
    max_patches: int,
    limit_samples: int | None,
) -> None:
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    per_sample_dir = output_dir / "per_sample"
    averages_dir = output_dir / "averages"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    averages_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    tifs = iter_raw_tifs(raw_root)
    if limit_samples is not None:
        tifs = tifs[: int(limit_samples)]
    ref_shape_theory = (nz_theory, nx_theory, nx_theory)
    ref_shape_empirical = (
        2 * half_z_patch + 1,
        2 * half_xy_patch + 1,
        2 * half_xy_patch + 1,
    )

    method_arrays: dict[str, list[np.ndarray]] = {
        **{slug: [] for slug in THEORY_MODELS_EPFL_NAMES},
        EMPIRICAL_METHOD[0]: [],
    }

    optics_json = {
        **asdict(optics),
        "epfl_reference": "https://bigwww.epfl.ch/algorithms/psfgenerator/",
        "python_implementation_note": (
            "Theoretical PSFs computed with psfmodels (vectorial, scalar Gibson–Lanni, Gaussian). "
            "Not bit-identical to the Java EPFL PSF Generator; cite both when publishing."
        ),
        "method_labels": {
            **{slug: THEORY_MODELS_EPFL_NAMES[slug][1] for slug in THEORY_MODELS_EPFL_NAMES},
            EMPIRICAL_METHOD[0]: EMPIRICAL_METHOD[1],
        },
    }
    (output_dir / "optics.json").write_text(json.dumps(optics_json, indent=2), encoding="utf-8")

    for i, tp in enumerate(tifs):
        name = safe_sample_name(tp.stem)
        print(f"[{i+1}/{len(tifs)}] {tp.relative_to(raw_root) if tp.is_relative_to(raw_root) else tp.name}")

        try:
            sample_psfs, dxy, dz = compute_sample_psfs(
                tp,
                optics=optics,
                channel=channel,
                nz_theory=nz_theory,
                nx_theory=nx_theory,
                half_z_patch=half_z_patch,
                half_xy_patch=half_xy_patch,
                z_stride_peaks=z_stride_peaks,
                peak_percentile=peak_percentile,
                min_peak_distance=min_peak_distance,
                max_patches=max_patches,
            )
        except Exception as exc:
            print(f"    skip: {exc}")
            continue
        manifest.append(
            {
                "tif": str(tp.resolve()),
                "sample_name": name,
                "dxy_um": dxy,
                "dz_um": dz,
                "methods": list(sample_psfs.keys()),
            }
        )

        meta_one = {"dxy_um": dxy, "dz_um": dz, **asdict(optics)}
        samp_dir = per_sample_dir / name
        samp_dir.mkdir(parents=True, exist_ok=True)
        for slug, arr in sample_psfs.items():
            np.save(samp_dir / f"{slug}.npy", arr.astype(np.float32))
            (samp_dir / f"{slug}_meta.json").write_text(
                json.dumps({**meta_one, "slug": slug, "shape_zyx": list(arr.shape)}),
                encoding="utf-8",
            )
            resized = resize_to_reference(
                arr,
                ref_shape_theory if slug in THEORY_MODELS_EPFL_NAMES else ref_shape_empirical,
            )
            if resized.max() <= 1e-9:
                continue
            if slug in THEORY_MODELS_EPFL_NAMES:
                resized = resized / float(np.max(resized))
                method_arrays[slug].append(resized)
            else:
                if float(np.sum(resized > 1e-6)) > resized.size * 0.01:
                    method_arrays[slug].append(resized / float(np.max(resized)))

        with open(per_sample_dir / name / "sample_manifest.json", "w", encoding="utf-8") as f:
            json.dump(meta_one, f, indent=2)

    median_dxy, median_dz = 0.5675, 0.6849
    dxs = [m["dxy_um"] for m in manifest]
    dzs = [m["dz_um"] for m in manifest]
    if dxs:
        median_dxy = float(np.median(np.asarray(dxs)))
    if dzs:
        median_dz = float(np.median(np.asarray(dzs)))

    averages_meta: dict[str, dict] = {}
    for slug, arr_list in method_arrays.items():
        if not arr_list:
            print(f"No arrays collected for '{slug}'.")
            continue
        ave = np.mean(np.stack(arr_list, axis=0), axis=0)
        ave = ave / max(float(np.max(ave)), 1e-12)
        shape = ave.shape

        dz_ref = median_dz
        dxy_ref = median_dxy
        if slug in THEORY_MODELS_EPFL_NAMES:
            ref_sz = ref_shape_theory
        else:
            ref_sz = ref_shape_empirical

        averages_meta[slug] = {
            "label": optics_json["method_labels"].get(slug, slug),
            "shape_zyx": shape,
            "mean_dxy_um": median_dxy,
            "mean_dz_um": median_dz,
            "spatial_extent_um_approx": [(shape[0] - 1) * dz_ref, (shape[1] - 1) * dxy_ref, (shape[2] - 1) * dxy_ref],
            "samples_averaged": len(arr_list),
        }
        np.save(averages_dir / f"{slug}_average.npy", ave.astype(np.float32))

    full_manifest = {"created_from": str(raw_root), "n_success": len(manifest), "samples": manifest}
    (output_dir / "manifest.json").write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")
    (averages_dir / "averages_summary.json").write_text(
        json.dumps(averages_meta, indent=2), encoding="utf-8"
    )
    print(f"\nSaved library under {output_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute multi-method PSF library from raw TIFFs.")
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--na", type=float, default=0.75)
    ap.add_argument("--wavelength-um", type=float, default=0.6)
    ap.add_argument("--ni", type=float, default=1.0)
    ap.add_argument("--ni0", type=float, default=1.0)
    ap.add_argument("--ns", type=float, default=1.33)
    ap.add_argument("--nz-theory", type=int, default=31)
    ap.add_argument("--nx-theory", type=int, default=63)
    ap.add_argument("--half-z-patch", type=int, default=15)
    ap.add_argument("--half-xy-patch", type=int, default=31)
    ap.add_argument("--z-stride-peaks", type=int, default=5)
    ap.add_argument("--peak-percentile", type=float, default=98.5)
    ap.add_argument("--min-peak-distance", type=int, default=12)
    ap.add_argument("--max-patches", type=int, default=120)
    ap.add_argument("--limit-samples", type=int, default=None)
    args = ap.parse_args(argv)

    optics = OpticalParams(
        NA=args.na,
        wavelength_um=args.wavelength_um,
        ni=args.ni,
        ni0=args.ni0,
        ns=args.ns,
    )

    if not iter_raw_tifs(args.raw_root.resolve()):
        print(f"No TIFFs under {args.raw_root}")
        return 1

    build_library(
        args.raw_root.resolve(),
        args.output_dir.resolve(),
        optics,
        args.channel,
        nz_theory=args.nz_theory,
        nx_theory=args.nx_theory,
        half_z_patch=args.half_z_patch,
        half_xy_patch=args.half_xy_patch,
        z_stride_peaks=args.z_stride_peaks,
        peak_percentile=args.peak_percentile,
        min_peak_distance=args.min_peak_distance,
        max_patches=args.max_patches,
        limit_samples=args.limit_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
