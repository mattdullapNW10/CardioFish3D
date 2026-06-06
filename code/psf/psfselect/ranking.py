"""Ranking of scored candidates and the PSF-model recommendation strategy.

Ranking sorts candidates by composite score (per sample, and overall). The
recommendation encodes the project's default decision strategy for zebrafish 3D
cardiac microscopy:

  1. Start with Gibson & Lanni (robust scalar baseline with stratified media).
  2. Compare against Born & Wolf (does stratified-media aberration matter at all?).
  3. Test Richards & Wolf when vectorial effects are likely relevant
     (high NA, or a clear axial side-lobe difference vs GL).
  4. Escalate to Variable-RI Gibson & Lanni when depth trends or RI mismatch
     suggest the simpler models are insufficient.

The output is a structured recommendation per sample with explicit reasons that
quote the numbers behind each decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import pandas as pd

# Thresholds for the decision rules (documented so they can be tuned).
HIGH_NA = 1.0                 # above this, vectorial (Richards & Wolf) is worth testing
RI_MISMATCH = 0.04            # |ni - ns| above this hints at index-mismatch aberration
DEEP_UM = 30.0                # imaging depth beyond which depth aberration matters
DEPTH_SENS_HI = 0.05          # >5% axial-FWHM drift with depth => depth-dependent
SIDELOBE_VECTORIAL = 0.08     # axial side-lobe gap that flags vectorial relevance


def rank_candidates(candidates: list) -> pd.DataFrame:
    """Return a tidy DataFrame of candidates sorted by composite score (desc)."""
    rows = []
    for c in candidates:
        s = c.scores or {}
        m = c.metrics
        rows.append({
            "candidate_id": c.candidate_id,
            "sample_id": c.sample_id,
            "model": c.model,
            "backend": c.backend_used,
            "na": c.params.get("na"),
            "wavelength_nm": c.params.get("wavelength_nm"),
            "depth_um": c.params.get("particle_depth_um"),
            "voxel_xy_um": c.params.get("voxel_xy_um"),
            "voxel_z_um": c.params.get("voxel_z_um"),
            "fwhm_lateral_um": m.get("fwhm_lateral_um"),
            "fwhm_z_um": m.get("fwhm_z_um"),
            "anisotropy": m.get("anisotropy"),
            "axial_sidelobe": m.get("axial_sidelobe_ratio"),
            "plausibility": s.get("plausibility"),
            "stability": s.get("stability"),
            "reblur": s.get("reblur"),
            "depth_sensitivity": s.get("depth_sensitivity"),
            "composite": s.get("composite"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sample_id", "composite"], ascending=[True, False], na_position="last")
        df["rank_in_sample"] = df.groupby("sample_id")["composite"].rank(ascending=False, method="min")
    return df.reset_index(drop=True)


@dataclass
class Recommendation:
    sample_id: str | None
    recommended_model: str
    confidence: str                     # "low" | "medium" | "high"
    reasons: list[str] = field(default_factory=list)
    escalation_flags: list[str] = field(default_factory=list)
    best_by_score: str | None = None
    ranking: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "recommended_model": self.recommended_model,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "escalation_flags": self.escalation_flags,
            "best_by_score": self.best_by_score,
            "ranking": self.ranking,
        }


def _by_model(cands: list) -> dict[str, Any]:
    return {c.model: c for c in cands}


def recommend_for_sample(sample_id: str | None, cands: list) -> Recommendation:
    """Apply the decision strategy to one sample's candidates.

    Candidates are expected to be already scored. If a sweep produced several
    candidates per model, the best-scoring one per model is used for the rules.
    """
    # Keep the best-scoring candidate per model.
    best_per_model: dict[str, Any] = {}
    for c in cands:
        cur = best_per_model.get(c.model)
        cs = (c.scores or {}).get("composite", float("nan"))
        if cur is None or (np.isfinite(cs) and cs > (cur.scores or {}).get("composite", -1)):
            best_per_model[c.model] = c

    by_model = best_per_model
    reasons: list[str] = []
    flags: list[str] = []

    # Overall best by composite score (for transparency).
    scored = [(m, (c.scores or {}).get("composite", float("nan"))) for m, c in by_model.items()]
    scored = [(m, s) for m, s in scored if np.isfinite(s)]
    best_by_score = max(scored, key=lambda t: t[1])[0] if scored else None

    # --- Rule 1: default baseline is Gibson & Lanni -----------------------
    recommended = "gibson_lanni" if "gibson_lanni" in by_model else (best_by_score or next(iter(by_model)))
    reasons.append("Default strategy starts from Gibson & Lanni: a robust scalar "
                   "baseline that models stratified-media (coverslip/immersion/specimen) aberration.")

    gl = by_model.get("gibson_lanni")
    bw = by_model.get("born_wolf")
    rw = by_model.get("richards_wolf")
    vrigl = by_model.get("vri_gibson_lanni")

    na = (gl or next(iter(by_model.values()))).params.get("na", 0.0)
    ni = (gl or next(iter(by_model.values()))).params.get("ni", 1.33)
    ns = (gl or next(iter(by_model.values()))).params.get("ns", 1.33)
    depth = (gl or next(iter(by_model.values()))).params.get("particle_depth_um", 0.0)
    ri_gap = abs(ni - ns)

    # --- Rule 2: compare with Born & Wolf ---------------------------------
    if gl is not None and bw is not None:
        gl_ax = gl.metrics.get("fwhm_z_um", float("nan"))
        bw_ax = bw.metrics.get("fwhm_z_um", float("nan"))
        if np.isfinite(gl_ax) and np.isfinite(bw_ax) and bw_ax > 0:
            rel = abs(gl_ax - bw_ax) / bw_ax
            if rel < 0.05:
                reasons.append(
                    f"Gibson & Lanni and Born & Wolf give near-identical axial FWHM "
                    f"({gl_ax:.2f} vs {bw_ax:.2f} um, {rel:.0%} diff): stratified-media "
                    f"aberration is mild here, so Born & Wolf is an acceptable simpler baseline."
                )
            else:
                reasons.append(
                    f"Gibson & Lanni differs from Born & Wolf in axial FWHM "
                    f"({gl_ax:.2f} vs {bw_ax:.2f} um, {rel:.0%}): stratified-media aberration "
                    f"is non-trivial, favouring Gibson & Lanni over Born & Wolf."
                )

    # --- Rule 3: vectorial relevance -> test Richards & Wolf --------------
    vectorial_relevant = na >= HIGH_NA
    if rw is not None and gl is not None:
        rw_sl = rw.metrics.get("axial_sidelobe_ratio", 0.0) or 0.0
        gl_sl = gl.metrics.get("axial_sidelobe_ratio", 0.0) or 0.0
        if abs(rw_sl - gl_sl) >= SIDELOBE_VECTORIAL:
            vectorial_relevant = True
            reasons.append(
                f"Richards & Wolf shows a different axial side-lobe structure than "
                f"Gibson & Lanni ({rw_sl:.2f} vs {gl_sl:.2f}): vectorial effects are "
                f"measurable; test Richards & Wolf."
            )
    if vectorial_relevant:
        flags.append(f"vectorial: NA={na:.2f} (>= {HIGH_NA}) or side-lobe gap — test Richards & Wolf")
        rw_score = (rw.scores or {}).get("composite", float("nan")) if rw else float("nan")
        gl_score = (gl.scores or {}).get("composite", float("nan")) if gl else float("nan")
        if rw is not None and np.isfinite(rw_score) and np.isfinite(gl_score) and rw_score > gl_score + 0.02:
            recommended = "richards_wolf"
            reasons.append(
                f"Richards & Wolf out-scores Gibson & Lanni ({rw_score:.2f} vs {gl_score:.2f}) "
                f"and vectorial effects are relevant: recommend Richards & Wolf."
            )

    # --- Rule 4: escalate to Variable-RI Gibson & Lanni -------------------
    depth_sens_vals = [
        (c.scores or {}).get("depth_sensitivity", float("nan")) for c in by_model.values()
    ]
    depth_sens_vals = [v for v in depth_sens_vals if np.isfinite(v)]
    max_depth_sens = max(depth_sens_vals) if depth_sens_vals else float("nan")

    escalate = False
    if ri_gap >= RI_MISMATCH:
        escalate = True
        flags.append(f"RI mismatch |ni-ns|={ri_gap:.3f} (>= {RI_MISMATCH})")
    if depth >= DEEP_UM:
        escalate = True
        flags.append(f"deep imaging: depth={depth:.0f} um (>= {DEEP_UM})")
    if np.isfinite(max_depth_sens) and max_depth_sens >= DEPTH_SENS_HI:
        escalate = True
        flags.append(f"depth-dependent axial FWHM: sensitivity={max_depth_sens:.0%} (>= {DEPTH_SENS_HI:.0%})")

    if escalate:
        reasons.append(
            "Depth/RI-mismatch indicators are triggered (see escalation flags): "
            "explore Variable-RI Gibson & Lanni, which models depth-dependent "
            "refractive-index aberration that the simpler models cannot capture."
        )
        if vrigl is not None:
            vg_score = (vrigl.scores or {}).get("composite", float("nan"))
            cur_score = (by_model.get(recommended).scores or {}).get("composite", float("nan"))
            if np.isfinite(vg_score) and np.isfinite(cur_score) and vg_score >= cur_score:
                recommended = "vri_gibson_lanni"
                reasons.append(
                    f"Variable-RI Gibson & Lanni matches or beats the current pick "
                    f"({vg_score:.2f} vs {cur_score:.2f}) under triggered depth/RI flags: "
                    f"recommend Variable-RI Gibson & Lanni."
                )
        else:
            reasons.append("Variable-RI Gibson & Lanni was not generated; add it to the model "
                           "set to act on this escalation.")

    # Confidence from how decisive the scores are and how complete the inputs are.
    confidence = _confidence(by_model, recommended, flags)

    ranking = sorted(
        ({"model": m, "composite": (c.scores or {}).get("composite")} for m, c in by_model.items()),
        key=lambda r: (r["composite"] is None, -(r["composite"] or 0)),
    )

    return Recommendation(
        sample_id=sample_id,
        recommended_model=recommended,
        confidence=confidence,
        reasons=reasons,
        escalation_flags=flags,
        best_by_score=best_by_score,
        ranking=ranking,
    )


def _confidence(by_model: dict, recommended: str, flags: list[str]) -> str:
    scores = [(c.scores or {}).get("composite", float("nan")) for c in by_model.values()]
    scores = sorted([s for s in scores if np.isfinite(s)], reverse=True)
    if len(scores) < 2:
        return "low"
    margin = scores[0] - scores[1]
    has_reblur = any((c.scores or {}).get("reblur") is not None and
                     np.isfinite((c.scores or {}).get("reblur", float("nan")))
                     for c in by_model.values())
    if margin > 0.1 and has_reblur:
        return "high"
    if margin > 0.05 or has_reblur:
        return "medium"
    return "low"


def recommend(candidates: list) -> list[Recommendation]:
    """Group candidates by sample and produce one recommendation per sample."""
    by_sample: dict[Any, list] = {}
    for c in candidates:
        by_sample.setdefault(c.sample_id, []).append(c)
    recs = [recommend_for_sample(sid, cs) for sid, cs in by_sample.items()]
    # Attach per-sample ranking detail.
    full = rank_candidates(candidates)
    for r in recs:
        sub = full[full["sample_id"] == r.sample_id] if "sample_id" in full else full
        r.ranking = sub.to_dict(orient="records") if not full.empty else r.ranking
    return recs
