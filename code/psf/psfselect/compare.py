"""Ground-truth-free comparison and scoring of candidate PSFs.

Because there is no measured bead PSF, candidates are judged by *indirect*
criteria, each mapped to a sub-score in [0, 1] where higher is better:

  plausibility  — does the PSF agree with diffraction theory implied by the
                  metadata (lateral & axial FWHM), is it well sampled (Nyquist),
                  centred, and fully contained in the grid?
  stability     — how little do the shape metrics move under small (±5%)
                  perturbations of NA / RI / depth? Robust models score higher.
  reblur        — (optional, needs raw data) does a few-iteration
                  deconvolve→reblur cycle reproduce the raw volume with low
                  residual and high correlation?

A weighted composite combines whichever sub-scores are available. Depth
sensitivity is measured separately and fed to the recommendation logic (it is a
*diagnostic*, not a quality axis: high depth sensitivity argues for VRI-GL, it
does not make a candidate "worse").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_WEIGHTS = {"plausibility": 0.5, "stability": 0.25, "reblur": 0.25}


def _theory_fwhm(na: float, lam_um: float, ni: float) -> tuple[float, float]:
    """Diffraction-limited lateral & axial FWHM (microns) from NA/lambda/ni."""
    fwhm_lat = 0.51 * lam_um / na
    fwhm_ax = 1.77 * ni * lam_um / (na ** 2)
    return fwhm_lat, fwhm_ax


def _closeness(measured: float, expected: float, tol: float = 0.5) -> float:
    """1.0 when measured==expected, decaying to 0 as the relative error grows.

    ``tol`` is the relative error at which the score reaches ~0.37 (1/e).
    """
    if not np.isfinite(measured) or not np.isfinite(expected) or expected <= 0:
        return 0.0
    rel = abs(measured - expected) / expected
    return float(math.exp(-(rel / tol) ** 2))


def score_plausibility(metrics: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
    """Plausibility sub-score with named components (each in [0,1])."""
    na = params["na"]
    lam_um = params["wavelength_nm"] / 1000.0
    ni = params["ni"]
    exp_lat, exp_ax = _theory_fwhm(na, lam_um, ni)

    lat_ok = _closeness(metrics["fwhm_lateral_um"], exp_lat, tol=0.6)
    ax_ok = _closeness(metrics["fwhm_z_um"], exp_ax, tol=0.8)

    # Nyquist sampling: voxel should be <= FWHM / 2.3 on each axis.
    dz, dxy = params["voxel_z_um"], params["voxel_xy_um"]
    lat_nyq = _sampling_score(dxy, metrics["fwhm_lateral_um"])
    ax_nyq = _sampling_score(dz, metrics["fwhm_z_um"])

    # Containment & centering.
    contained = 1.0 if metrics.get("contained") else 0.3
    off = metrics.get("peak_offset_vox", [0, 0, 0])
    nz = params["nz"]
    nx = params["nx"]
    center_err = abs(off[0]) / max(nz, 1) + abs(off[1]) / max(nx, 1) + abs(off[2]) / max(nx, 1)
    centered = float(math.exp(-(center_err / 0.15) ** 2))

    # Lateral symmetry (NaN-safe).
    sym = metrics.get("lateral_symmetry")
    sym = float(sym) if (sym is not None and np.isfinite(sym)) else 0.8

    components = {
        "lateral_fwhm_match": lat_ok,
        "axial_fwhm_match": ax_ok,
        "lateral_sampling": lat_nyq,
        "axial_sampling": ax_nyq,
        "containment": contained,
        "centering": centered,
        "symmetry": sym,
    }
    weights = {
        "lateral_fwhm_match": 0.25,
        "axial_fwhm_match": 0.2,
        "lateral_sampling": 0.15,
        "axial_sampling": 0.15,
        "containment": 0.1,
        "centering": 0.1,
        "symmetry": 0.05,
    }
    score = sum(components[k] * weights[k] for k in components)
    components["score"] = float(score)
    return components


def _sampling_score(voxel: float, fwhm: float) -> float:
    """1.0 when voxel <= FWHM/2.3 (Nyquist), decaying for coarser sampling."""
    if not np.isfinite(fwhm) or fwhm <= 0 or voxel <= 0:
        return 0.0
    samples_per_fwhm = fwhm / voxel
    # Need >= 2.3 samples per FWHM. Full marks at >=2.3, ~0 by ~1.0.
    return float(np.clip((samples_per_fwhm - 1.0) / (2.3 - 1.0), 0.0, 1.0))


def score_stability(base_metrics: dict[str, Any],
                    perturbation_fwhm: dict[str, dict[str, float]]) -> dict[str, float]:
    """Stability sub-score from relative metric drift under perturbations."""
    if not perturbation_fwhm:
        return {"score": float("nan"), "max_rel_drift": float("nan"), "n": 0}
    base_lat = base_metrics["fwhm_lateral_um"]
    base_ax = base_metrics["fwhm_z_um"]
    drifts: list[float] = []
    for vals in perturbation_fwhm.values():
        if base_lat and np.isfinite(base_lat):
            drifts.append(abs(vals["fwhm_lateral_um"] - base_lat) / base_lat)
        if base_ax and np.isfinite(base_ax):
            drifts.append(abs(vals["fwhm_z_um"] - base_ax) / base_ax)
    if not drifts:
        return {"score": float("nan"), "max_rel_drift": float("nan"), "n": 0}
    mean_drift = float(np.mean(drifts))
    max_drift = float(np.max(drifts))
    # 0% drift -> 1.0; ~20% mean drift -> ~0.37.
    score = float(math.exp(-(mean_drift / 0.2) ** 2))
    return {"score": score, "mean_rel_drift": mean_drift, "max_rel_drift": max_drift, "n": len(drifts)}


def depth_sensitivity(perturbation_fwhm: dict[str, dict[str, float]],
                      base_metrics: dict[str, Any]) -> float:
    """Relative axial-FWHM change attributable to depth perturbation (diagnostic)."""
    base_ax = base_metrics["fwhm_z_um"]
    if not base_ax or not np.isfinite(base_ax):
        return float("nan")
    depth_keys = [k for k in perturbation_fwhm if k.startswith("particle_depth_um")]
    if not depth_keys:
        return float("nan")
    drifts = [abs(perturbation_fwhm[k]["fwhm_z_um"] - base_ax) / base_ax for k in depth_keys]
    return float(np.mean(drifts))


def score_reblur(reblur: dict[str, float] | None) -> dict[str, float]:
    """Reblur consistency sub-score from residual / correlation / sharpness gain."""
    if not reblur:
        return {"score": float("nan")}
    residual = reblur.get("residual", float("nan"))
    corr = reblur.get("correlation", float("nan"))
    gain = reblur.get("sharpness_gain", float("nan"))
    res_score = float(math.exp(-(max(residual, 0.0) / 0.5) ** 2)) if np.isfinite(residual) else 0.0
    corr_score = float(np.clip((corr + 1) / 2, 0, 1)) if np.isfinite(corr) else 0.0
    gain_score = float(np.clip((gain - 1.0) / 1.0, 0, 1)) if np.isfinite(gain) else 0.0
    score = 0.5 * res_score + 0.35 * corr_score + 0.15 * gain_score
    return {"score": float(score), "residual_score": res_score,
            "correlation_score": corr_score, "gain_score": gain_score}


def composite_score(subscores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Weighted mean over available (non-NaN) sub-scores; re-normalises weights."""
    weights = weights or DEFAULT_WEIGHTS
    num = den = 0.0
    for key, w in weights.items():
        val = subscores.get(key)
        if val is not None and np.isfinite(val):
            num += w * val
            den += w
    return float(num / den) if den > 0 else float("nan")


def score_candidate(candidate, weights: dict[str, float] | None = None) -> None:
    """Compute and attach all sub-scores + composite to a Candidate in place."""
    plaus = score_plausibility(candidate.metrics, candidate.params)
    stab = score_stability(candidate.metrics, candidate.perturbation_fwhm)
    reb = score_reblur(candidate.reblur)
    depth_sens = depth_sensitivity(candidate.perturbation_fwhm, candidate.metrics)

    subs = {
        "plausibility": plaus["score"],
        "stability": stab["score"],
        "reblur": reb["score"],
    }
    comp = composite_score(subs, weights)
    candidate.scores = {
        "plausibility": plaus["score"],
        "plausibility_components": plaus,
        "stability": stab["score"],
        "stability_detail": stab,
        "reblur": reb["score"],
        "reblur_detail": reb,
        "depth_sensitivity": depth_sens,
        "composite": comp,
    }


def score_all(candidates: list, weights: dict[str, float] | None = None) -> list:
    for c in candidates:
        score_candidate(c, weights)
    return candidates
