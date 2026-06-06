"""Reporting: comparison tables, summary figures, and a ranked Markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import MODEL_LABELS
from .io_utils import LOG, ensure_dir, write_json
from .ranking import rank_candidates, recommend
from .viz import (
    plot_model_comparison,
    plot_orthoviews_from_candidate,
    plot_score_breakdown,
    plot_sweep_trends,
)

SWEEP_AXES = ["na", "wavelength_nm", "depth_um", "voxel_xy_um", "voxel_z_um"]


def build_report(candidates: list, outdir: str | Path, *,
                 make_orthoviews: bool = True) -> dict[str, Any]:
    """Produce tables, figures and a ranked Markdown report. Returns artefact paths."""
    outdir = ensure_dir(outdir)
    fig_dir = ensure_dir(outdir / "figures")

    df = rank_candidates(candidates)
    table_csv = outdir / "comparison_table.csv"
    df.to_csv(table_csv, index=False)

    recs = recommend(candidates)
    rec_json = write_json(outdir / "recommendations.json", [r.to_dict() for r in recs])

    # Figures (overall + per sample).
    figures: list[str] = []
    samples = sorted({c.sample_id for c in candidates}, key=lambda s: (s is None, s))
    for sid in samples:
        tag = sid or "all"
        figures.append(str(plot_model_comparison(df, fig_dir / f"compare_{tag}.png", sample_id=sid)))
        figures.append(str(plot_score_breakdown(df, fig_dir / f"scores_{tag}.png", sample_id=sid)))
        for axis in SWEEP_AXES:
            p = plot_sweep_trends(df, axis, fig_dir / f"sweep_{tag}_{axis}.png", sample_id=sid)
            if p is not None:
                figures.append(str(p))

    if make_orthoviews:
        ortho_dir = ensure_dir(fig_dir / "orthoviews")
        for c in candidates:
            p = plot_orthoviews_from_candidate(c, ortho_dir)
            if p is not None:
                figures.append(str(p))

    md_path = _write_markdown(outdir / "report.md", df, recs, figures, fig_dir, outdir)

    LOG.info("Report written: %s", md_path)
    return {
        "report_md": str(md_path),
        "comparison_table": str(table_csv),
        "recommendations": str(rec_json),
        "figures": figures,
    }


def _write_markdown(path: Path, df: pd.DataFrame, recs: list, figures: list[str],
                    fig_dir: Path, outdir: Path) -> Path:
    lines: list[str] = []
    lines.append("# PSF model selection report\n")
    lines.append(f"Candidates evaluated: **{len(df)}**  |  "
                 f"Models: {', '.join(sorted(df['model'].unique())) if not df.empty else 'none'}  |  "
                 f"Samples: {df['sample_id'].nunique() if not df.empty else 0}\n")
    lines.append("Scores are ground-truth-free indirect criteria in [0,1] "
                 "(higher = better): **plausibility** (diffraction-theory & sampling "
                 "consistency), **stability** (robustness to ±parameter perturbation), "
                 "**reblur** (optional data-driven deconvolve→reblur consistency). "
                 "**depth_sensitivity** is a diagnostic, not a quality score.\n")

    # Recommendations per sample.
    lines.append("## Recommendations\n")
    for r in recs:
        lines.append(f"### Sample: `{r.sample_id}`\n")
        lines.append(f"- **Recommended model:** `{r.recommended_model}` "
                     f"— {MODEL_LABELS.get(r.recommended_model, '')}")
        lines.append(f"- **Confidence:** {r.confidence}")
        if r.best_by_score:
            lines.append(f"- **Highest raw score:** `{r.best_by_score}`")
        if r.escalation_flags:
            lines.append(f"- **Escalation flags:** {'; '.join(r.escalation_flags)}")
        lines.append("\n**Reasoning:**")
        for reason in r.reasons:
            lines.append(f"  - {reason}")
        lines.append("")

    # Ranked comparison table.
    lines.append("## Ranked comparison table\n")
    if not df.empty:
        cols = ["rank_in_sample", "sample_id", "model", "backend", "fwhm_lateral_um",
                "fwhm_z_um", "anisotropy", "plausibility", "stability", "reblur",
                "depth_sensitivity", "composite"]
        cols = [c for c in cols if c in df.columns]
        show = df[cols].copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        lines.append(show.to_markdown(index=False))
    lines.append("")

    # Figures.
    lines.append("## Figures\n")
    for f in figures:
        rel = Path(f).relative_to(outdir) if Path(f).is_relative_to(outdir) else Path(f)
        lines.append(f"![{Path(f).stem}]({rel})")
    lines.append("")

    lines.append("## Assumptions & caveats\n")
    lines.append("- No measured (bead) PSF ground truth is used; rankings are *relative* "
                 "and based on physical plausibility, stability and optional data consistency.")
    lines.append("- Theoretical FWHM references use 0.51·λ/NA (lateral) and 1.77·n·λ/NA² (axial).")
    lines.append("- Any optical fields missing from metadata were filled with documented "
                 "defaults; check `metadata_resolved.json` for what was assumed per sample.")
    lines.append("- If the EPFL PSF Generator JAR was unavailable, the psfmodels fallback "
                 "backend was used (see the `backend` column). Born & Wolf is then "
                 "approximated by a matched-RI scalar model and Variable-RI GL by a "
                 "depth-adjusted specimen RI.")

    path.write_text("\n".join(lines))
    return path
