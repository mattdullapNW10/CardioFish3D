"""Pin the EPFL PSF Generator config format to the documented key names."""

from psfselect.parameters import PSFParams
from psfselect.psfgenerator import build_config


def _cfg(model="gibson_lanni", **kw):
    base = dict(na=1.4, wavelength_nm=610, ni=1.5, ns=1.33,
                voxel_xy_um=0.1, voxel_z_um=0.25, nx=256, nz=65,
                particle_depth_um=2.0, working_distance_um=150.0)
    base.update(kw)
    return build_config(model, PSFParams(**base))


def test_common_block_keys():
    cfg = _cfg()
    for key in ("PSF-shortname=GL", "ResLateral=100.0", "ResAxial=250.0",
                "NX=256", "NY=256", "NZ=65", "Type=32-bits", "NA=1.4",
                "Lambda=610.0", "Scale=Linear"):
        assert any(line.startswith(key) for line in cfg.splitlines()), key


def test_documented_model_keys_present():
    cfg = _cfg()
    # Exact key names from the EPFL reference config (no legacy ni0/tg/Lambda-per-model).
    expected = [
        "psf-BW-NI=", "psf-BW-accuracy=",
        "psf-RW-NI=", "psf-RW-accuracy=",
        "psf-GL-NI=", "psf-GL-NS=", "psf-GL-accuracy=", "psf-GL-ZPos=", "psf-GL-TI=",
        "psf-VRIGL-NI=", "psf-VRIGL-NS1=", "psf-VRIGL-NS2=", "psf-VRIGL-NG=",
        "psf-VRIGL-TG=", "psf-VRIGL-TI=", "psf-VRIGL-RIvary=Linear", "psf-VRIGL-ZPos=",
    ]
    for key in expected:
        assert key in cfg, f"missing documented key {key}"
    # Legacy/incorrect keys must NOT appear.
    for bad in ("psf-GL-ni0", "psf-GL-tg", "psf-GL-Lambda", "psf-BW-NA", "psf-RW-ti0"):
        assert bad not in cfg, f"stale key leaked: {bad}"


def test_shortname_per_model():
    assert "PSF-shortname=BW" in _cfg("born_wolf")
    assert "PSF-shortname=RW" in _cfg("richards_wolf")
    assert "PSF-shortname=VRIGL" in _cfg("vri_gibson_lanni")


def test_zpos_in_nm_and_vrigl_gradient():
    cfg = _cfg(particle_depth_um=2.0, ns=1.33, ns2=1.40)
    assert "psf-GL-ZPos=2000.0" in cfg            # 2 um -> 2000 nm
    assert "psf-VRIGL-NS1=1.33" in cfg
    assert "psf-VRIGL-NS2=1.40" in cfg            # explicit depth RI honoured
