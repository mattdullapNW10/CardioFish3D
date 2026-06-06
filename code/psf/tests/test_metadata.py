import json

from psfselect.metadata import load_metadata, validate_samples, DEFAULTS, CRITICAL_FIELDS


def test_load_yaml_and_defaults(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text(
        "samples:\n"
        "  - sample_id: s1\n"
        "    na: 0.8\n"
        "    wavelength_nm: 510\n"
    )
    samples = validate_samples(load_metadata(p))
    assert len(samples) == 1
    s = samples[0]
    # missing ns should be filled from defaults and flagged
    assert "ns" in s.missing_fields
    assert s.ns == DEFAULTS["ns"]
    assert "na" not in s.missing_fields


def test_csv_aliases(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("id,NA,emission_wavelength,dz\nfoo,0.9,488,0.4\n")
    samples = validate_samples(load_metadata(p))
    s = samples[0]
    assert s.sample_id == "foo"
    assert s.na == 0.9
    assert s.wavelength_nm == 488
    assert s.voxel_z_um == 0.4


def test_critical_missing_flag(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps([{"sample_id": "bare"}]))
    s = validate_samples(load_metadata(p))[0]
    crit = s.critical_missing()
    assert set(crit) == set(CRITICAL_FIELDS)
