"""Candidate PSF generation: render models × parameter sweeps and persist them.

A *candidate* is one (model, parameter-set) pairing rendered to a 3D volume,
together with its shape metrics, optional stability (perturbation) metrics, and
optional data-driven reblur consistency. Candidates are the unit that the
comparison and ranking stages operate on.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import MODELS, MODEL_LABELS
from .backends import render_psf
from .io_utils import LOG, ensure_dir, save_volume, write_json
from .metrics import PSFMetrics, compute_metrics
from .parameters import PSFParams, perturbations


@dataclass
class Candidate:
    candidate_id: str
    sample_id: str | None
    model: str
    backend_used: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    voxel_um: tuple[float, float, float]
    perturbation_fwhm: dict[str, dict[str, float]] = field(default_factory=dict)
    reblur: dict[str, float] | None = None
    psf_path: str | None = None
    config_path: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    recommendation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["voxel_um"] = list(self.voxel_um)
        d["model_label"] = MODEL_LABELS.get(self.model, self.model)
        return d


def _candidate_id(model: str, params: PSFParams) -> str:
    key = f"{params.sample_id}|{model}|{params.na}|{params.wavelength_nm}|{params.ni}|{params.ns}|{params.voxel_xy_um}|{params.voxel_z_um}|{params.nx}|{params.nz}|{params.particle_depth_um}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    sid = params.sample_id or "sample"
    return f"{sid}__{model}__{h}"


def generate_candidate(
    model: str,
    params: PSFParams,
    *,
    backend: str = "auto",
    outdir: str | Path | None = None,
    jar_path=None,
    with_stability: bool = True,
    stability_rel: float = 0.05,
    raw_volume: np.ndarray | None = None,
    reblur_iters: int = 8,
) -> Candidate:
    """Render one candidate PSF and compute its metrics (+ optional stability/reblur)."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'")
    voxel = (params.voxel_z_um, params.voxel_xy_um, params.voxel_xy_um)

    vol, backend_used, cfg_text = render_psf(model, params, backend=backend, jar_path=jar_path)
    m = compute_metrics(vol, voxel)

    cid = _candidate_id(model, params)
    psf_path = cfg_path = None
    if outdir is not None:
        outdir = ensure_dir(outdir)
        psf_path = str(save_volume(outdir / f"{cid}.tif", vol, voxel_um=voxel))
        if cfg_text is not None:
            cfg_path = str((outdir / f"{cid}.config.txt"))
            Path(cfg_path).write_text(cfg_text)

    cand = Candidate(
        candidate_id=cid,
        sample_id=params.sample_id,
        model=model,
        backend_used=backend_used,
        params=params.to_dict(),
        metrics=m.to_dict(),
        voxel_um=voxel,
        psf_path=psf_path,
        config_path=cfg_path,
    )

    # Stability: render small perturbations and record FWHM summaries.
    if with_stability:
        for label, pert in perturbations(params, rel=stability_rel):
            try:
                pvol, _, _ = render_psf(model, pert, backend=backend, jar_path=jar_path)
                pm = compute_metrics(pvol, (pert.voxel_z_um, pert.voxel_xy_um, pert.voxel_xy_um))
                cand.perturbation_fwhm[label] = {
                    "fwhm_lateral_um": pm.fwhm_lateral_um,
                    "fwhm_z_um": pm.fwhm_z_um,
                    "anisotropy": pm.anisotropy,
                }
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Perturbation %s failed for %s: %s", label, cid, exc)

    # Optional data-driven reblur consistency.
    if raw_volume is not None:
        try:
            from .reblur import reblur_consistency

            cand.reblur = reblur_consistency(raw_volume, vol, iters=reblur_iters)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Reblur consistency failed for %s: %s", cid, exc)

    return cand


def generate_candidates(
    base_params: PSFParams,
    *,
    models: list[str] | None = None,
    sweep_params: list[PSFParams] | None = None,
    backend: str = "auto",
    outdir: str | Path | None = None,
    jar_path=None,
    with_stability: bool = True,
    raw_volume: np.ndarray | None = None,
) -> list[Candidate]:
    """Generate candidates over the cross-product of models and parameter sets."""
    models = models or list(MODELS)
    param_sets = sweep_params or [base_params]
    candidates: list[Candidate] = []
    total = len(models) * len(param_sets)
    LOG.info("Generating %d candidate PSF(s): %d model(s) x %d param set(s)",
             total, len(models), len(param_sets))
    for p in param_sets:
        for model in models:
            cand = generate_candidate(
                model, p, backend=backend, outdir=outdir, jar_path=jar_path,
                with_stability=with_stability, raw_volume=raw_volume,
            )
            LOG.info("  rendered %s (backend=%s, FWHM_lat=%.3f um, aniso=%.2f)",
                     cand.candidate_id, cand.backend_used,
                     cand.metrics["fwhm_lateral_um"], cand.metrics["anisotropy"])
            candidates.append(cand)
    return candidates


def save_manifest(candidates: list[Candidate], path: str | Path) -> Path:
    """Write a JSON manifest of all candidates (params, metrics, scores, paths)."""
    payload = {
        "n_candidates": len(candidates),
        "models": sorted({c.model for c in candidates}),
        "samples": sorted({c.sample_id for c in candidates if c.sample_id}),
        "candidates": [c.to_dict() for c in candidates],
    }
    return write_json(path, payload)
