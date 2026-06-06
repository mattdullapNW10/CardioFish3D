"""Microscopy metadata schema, ingestion (JSON/CSV/YAML), and validation.

The schema is deliberately tolerant: every optical field is optional. Missing
fields are recorded and surfaced (``missing_fields``) rather than raising, so a
partially-documented dataset can still flow through the pipeline with sensible
defaults and explicit provenance.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .io_utils import LOG, read_json, read_voxel_um_from_tiff

# Defaults used when a field is missing. These mirror a generic confocal water /
# air objective and are intentionally conservative; every applied default is
# recorded in ``SampleMetadata.applied_defaults`` so nothing is silently assumed.
DEFAULTS: dict[str, Any] = {
    "modality": "confocal",
    "wavelength_nm": 600.0,        # emission wavelength
    "na": 0.8,
    "ni": 1.33,                    # immersion medium RI (water default)
    "ni0": 1.33,                   # design immersion RI
    "ns": 1.35,                    # specimen RI (zebrafish tissue ~1.35)
    "ng": 1.515,                   # coverslip RI
    "ng0": 1.515,
    "coverslip_um": 170.0,
    "coverslip_design_um": 170.0,
    "working_distance_um": 2000.0,
    "voxel_z_um": 1.0,
    "voxel_y_um": 0.3,
    "voxel_x_um": 0.3,
    "imaging_depth_um": 0.0,
    "objective": "unknown",
}

# Fields we *require* the user to think about for trustworthy PSF modelling.
# Missing ones are flagged (not fatal) so the report can warn about them.
CRITICAL_FIELDS = ("wavelength_nm", "na", "ni", "ns", "voxel_z_um", "voxel_x_um")


@dataclass
class SampleMetadata:
    """One imaged sample / volume and its optical context."""

    sample_id: str
    modality: str | None = None
    wavelength_nm: float | None = None       # emission wavelength
    na: float | None = None                  # numerical aperture
    ni: float | None = None                  # immersion medium refractive index
    ni0: float | None = None                 # design immersion RI
    ns: float | None = None                  # specimen / sample refractive index
    ng: float | None = None                  # coverslip RI
    ng0: float | None = None                 # design coverslip RI
    coverslip_um: float | None = None        # actual coverslip thickness
    coverslip_design_um: float | None = None # design coverslip thickness
    working_distance_um: float | None = None
    voxel_x_um: float | None = None
    voxel_y_um: float | None = None
    voxel_z_um: float | None = None
    image_x: int | None = None
    image_y: int | None = None
    image_z: int | None = None
    imaging_depth_um: float | None = None    # depth of acquisition / z-position
    objective: str | None = None
    image_path: str | None = None            # optional path to the raw volume
    notes: str | None = None

    # Provenance (filled by validation):
    missing_fields: list[str] = field(default_factory=list)
    applied_defaults: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def with_defaults(self) -> "SampleMetadata":
        """Return a copy with missing optical fields filled from DEFAULTS.

        Records which fields were missing and which defaults were applied. Voxel
        sizes are additionally pulled from the TIFF metadata when an image path
        is available and the value is missing.
        """
        data = dataclasses.asdict(self)
        missing: list[str] = []
        applied: dict[str, Any] = {}

        # Try to enrich voxel sizes from the image file itself.
        if self.image_path and Path(self.image_path).exists():
            vox = read_voxel_um_from_tiff(self.image_path)
            if vox is not None:
                dz, dy, dx = vox
                for key, val in (("voxel_z_um", dz), ("voxel_y_um", dy), ("voxel_x_um", dx)):
                    if data.get(key) is None:
                        data[key] = val
                        applied[key] = f"{val:.4f} (from TIFF metadata)"

        for f in fields(self):
            if f.name in ("sample_id", "missing_fields", "applied_defaults",
                          "image_path", "notes", "image_x", "image_y", "image_z"):
                continue
            if data.get(f.name) is None:
                missing.append(f.name)
                if f.name in DEFAULTS:
                    data[f.name] = DEFAULTS[f.name]
                    applied.setdefault(f.name, DEFAULTS[f.name])

        data["missing_fields"] = missing
        data["applied_defaults"] = applied
        out = SampleMetadata(**data)
        return out

    def critical_missing(self) -> list[str]:
        """Critical fields that were missing in the *original* metadata."""
        return [f for f in CRITICAL_FIELDS if f in self.missing_fields]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _coerce_record(rec: dict[str, Any]) -> SampleMetadata:
    """Map an arbitrary dict (CSV row / JSON / YAML entry) to SampleMetadata."""
    valid = {f.name for f in fields(SampleMetadata)}
    # Common alias normalisation so users can be a little loose.
    aliases = {
        "id": "sample_id",
        "name": "sample_id",
        "emission_wavelength": "wavelength_nm",
        "wavelength": "wavelength_nm",
        "emission_wavelength_nm": "wavelength_nm",
        "numerical_aperture": "na",
        "NA": "na",
        "immersion_ri": "ni",
        "n_immersion": "ni",
        "sample_ri": "ns",
        "n_sample": "ns",
        "specimen_ri": "ns",
        "dz": "voxel_z_um",
        "dx": "voxel_x_um",
        "dy": "voxel_y_um",
        "depth": "imaging_depth_um",
        "depth_um": "imaging_depth_um",
        "z_position_um": "imaging_depth_um",
        "path": "image_path",
        "file": "image_path",
    }
    clean: dict[str, Any] = {}
    for key, val in rec.items():
        if val in ("", None):
            continue
        norm = aliases.get(key, key)
        if norm not in valid:
            continue
        clean[norm] = _maybe_number(norm, val)
    if "sample_id" not in clean:
        raise ValueError(f"Metadata record missing 'sample_id': {rec}")
    return SampleMetadata(**clean)


_FLOAT_FIELDS = {
    "wavelength_nm", "na", "ni", "ni0", "ns", "ng", "ng0", "coverslip_um",
    "coverslip_design_um", "working_distance_um", "voxel_x_um", "voxel_y_um",
    "voxel_z_um", "imaging_depth_um",
}
_INT_FIELDS = {"image_x", "image_y", "image_z"}


def _maybe_number(field_name: str, val: Any) -> Any:
    if field_name in _FLOAT_FIELDS:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    if field_name in _INT_FIELDS:
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return None
    return str(val)


def load_metadata(path: str | Path) -> list[SampleMetadata]:
    """Load metadata from JSON / CSV / YAML into a list of SampleMetadata.

    Accepts a single record or a list of records for JSON/YAML; CSV is one
    record per row.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        records = rows
    elif suffix in (".json",):
        obj = read_json(path)
        records = obj if isinstance(obj, list) else [obj]
    elif suffix in (".yaml", ".yml"):
        import yaml

        with open(path) as fh:
            obj = yaml.safe_load(fh)
        if isinstance(obj, dict) and "samples" in obj:
            obj = obj["samples"]
        records = obj if isinstance(obj, list) else [obj]
    else:
        raise ValueError(f"Unsupported metadata format: {suffix}")

    samples = [_coerce_record(r) for r in records]
    LOG.info("Loaded %d sample(s) from %s", len(samples), path.name)
    return samples


def validate_samples(samples: list[SampleMetadata]) -> list[SampleMetadata]:
    """Fill defaults and log missing/critical fields for each sample."""
    out = []
    for s in samples:
        filled = s.with_defaults()
        if filled.missing_fields:
            crit = filled.critical_missing()
            level = LOG.warning if crit else LOG.info
            level(
                "Sample %s missing %d field(s)%s; defaults applied: %s",
                filled.sample_id,
                len(filled.missing_fields),
                f" (CRITICAL: {crit})" if crit else "",
                ", ".join(filled.applied_defaults) or "none",
            )
        out.append(filled)
    return out
