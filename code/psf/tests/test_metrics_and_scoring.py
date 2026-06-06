import numpy as np

from psfselect.metrics import compute_metrics
from psfselect.compare import (
    score_plausibility, score_stability, score_reblur, composite_score, _theory_fwhm,
)


def _gaussian_psf(nz=64, nx=64, sz=4.0, sxy=2.0):
    z = np.arange(nz) - nz // 2
    y = np.arange(nx) - nx // 2
    gz = np.exp(-0.5 * (z / sz) ** 2)
    gy = np.exp(-0.5 * (y / sxy) ** 2)
    vol = gz[:, None, None] * gy[None, :, None] * gy[None, None, :]
    return vol.astype(np.float32)


def test_fwhm_and_anisotropy():
    vol = _gaussian_psf(sz=4.0, sxy=2.0)
    m = compute_metrics(vol, voxel_um=(1.0, 1.0, 1.0))
    # FWHM = 2.3548 * sigma
    assert abs(m.fwhm_z_um - 2.3548 * 4.0) < 0.6
    assert abs(m.fwhm_lateral_um - 2.3548 * 2.0) < 0.6
    # axial broader than lateral -> anisotropy ~ 2
    assert m.anisotropy > 1.5
    assert m.contained


def test_plausibility_in_range():
    vol = _gaussian_psf()
    m = compute_metrics(vol, (1.0, 1.0, 1.0)).to_dict()
    params = {"na": 0.8, "wavelength_nm": 510, "ni": 1.33,
              "voxel_z_um": 1.0, "voxel_xy_um": 1.0, "nz": 64, "nx": 64}
    s = score_plausibility(m, params)
    assert 0.0 <= s["score"] <= 1.0
    for k in ("lateral_fwhm_match", "containment", "centering"):
        assert 0.0 <= s[k] <= 1.0


def test_stability_perfect_when_no_drift():
    base = {"fwhm_lateral_um": 1.0, "fwhm_z_um": 2.0}
    pert = {
        "na+5%": {"fwhm_lateral_um": 1.0, "fwhm_z_um": 2.0, "anisotropy": 2.0},
        "na-5%": {"fwhm_lateral_um": 1.0, "fwhm_z_um": 2.0, "anisotropy": 2.0},
    }
    s = score_stability(base, pert)
    assert s["score"] > 0.99
    assert s["max_rel_drift"] == 0.0


def test_reblur_score_monotonic():
    good = score_reblur({"residual": 0.1, "correlation": 0.95, "sharpness_gain": 1.5})
    bad = score_reblur({"residual": 1.0, "correlation": 0.1, "sharpness_gain": 1.0})
    assert good["score"] > bad["score"]
    assert score_reblur(None)["score"] != score_reblur(None)["score"]  # NaN


def test_composite_skips_nan():
    subs = {"plausibility": 0.8, "stability": float("nan"), "reblur": 0.6}
    c = composite_score(subs, {"plausibility": 0.5, "stability": 0.25, "reblur": 0.25})
    # stability dropped, weights renormalised over plausibility+reblur
    expected = (0.5 * 0.8 + 0.25 * 0.6) / (0.5 + 0.25)
    assert abs(c - expected) < 1e-9


def test_theory_fwhm_scaling():
    lat1, ax1 = _theory_fwhm(0.8, 0.51, 1.33)
    lat2, ax2 = _theory_fwhm(1.2, 0.51, 1.33)
    # higher NA -> tighter PSF
    assert lat2 < lat1 and ax2 < ax1
