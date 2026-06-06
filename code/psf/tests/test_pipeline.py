"""End-to-end smoke test using the always-available gaussian backend (fast)."""

from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psfselect.candidates import generate_candidates, save_manifest
from psfselect.compare import score_all
from psfselect.metadata import validate_samples, SampleMetadata
from psfselect.parameters import params_from_metadata, expand_sweep, SweepSpec
from psfselect.ranking import rank_candidates, recommend
from psfselect.report import build_report


def _sample():
    return validate_samples([SampleMetadata(sample_id="t1", na=1.1, wavelength_nm=510,
                                            ni=1.33, ns=1.40, voxel_x_um=0.1,
                                            voxel_z_um=0.3, imaging_depth_um=40)])[0]


def test_generate_score_rank_report(tmp_path):
    s = _sample()
    base = params_from_metadata(s, nx=48, nz=32)
    cands = generate_candidates(base, backend="gaussian", outdir=tmp_path / "psfs",
                                with_stability=True)
    assert len(cands) == 4  # four models
    for c in cands:
        assert c.metrics["fwhm_lateral_um"] > 0
        assert c.perturbation_fwhm  # stability renders happened

    score_all(cands)
    for c in cands:
        assert 0.0 <= c.scores["composite"] <= 1.0

    df = rank_candidates(cands)
    assert "rank_in_sample" in df.columns
    assert df["composite"].notna().any()

    recs = recommend(cands)
    assert len(recs) == 1
    r = recs[0]
    # NA=1.1 (>1.0) + RI mismatch 0.07 + depth 40 -> escalation flags present
    assert r.escalation_flags
    assert r.recommended_model in {"gibson_lanni", "richards_wolf", "vri_gibson_lanni",
                                   "born_wolf"}

    art = build_report(cands, tmp_path / "report", make_orthoviews=True)
    assert Path(art["report_md"]).exists()
    assert Path(art["comparison_table"]).exists()
    assert len(art["figures"]) > 0


def test_manifest_roundtrip(tmp_path):
    s = _sample()
    base = params_from_metadata(s, nx=40, nz=24)
    cands = generate_candidates(base, models=["gibson_lanni", "born_wolf"],
                                backend="gaussian", outdir=tmp_path, with_stability=False)
    mpath = save_manifest(cands, tmp_path / "manifest.json")
    assert mpath.exists()


def test_sweep_expansion():
    s = _sample()
    base = params_from_metadata(s)
    sweep = SweepSpec.from_dict({"na": [0.8, 1.0], "particle_depth_um": [0, 20]})
    sets = expand_sweep(base, sweep)
    assert len(sets) == 4
    assert {p.na for p in sets} == {0.8, 1.0}
