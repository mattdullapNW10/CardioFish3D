"""Extract PSF-relevant optical metadata from microscopy files.

Targets Leica ``.lif`` TIFF exports (the ImageJ ``Info`` block), with fallbacks
to OME-XML and plain TIFF tags. Produces a :class:`~psfselect.metadata.SampleMetadata`
with the fields that matter for PSF modelling, filled from the file wherever
possible:

  * NA                         <- NumericalAperture
  * immersion RI (ni)          <- RefractionIndex / Immersion (DRY=1.0, WATER=1.33, OIL=1.515)
  * emission wavelength        <- fluorophore (DyeName) emission, NOT the pinhole
                                  reference wavelength
  * voxel sizes                <- TIFF resolution / ImageJ spacing
  * image dimensions, objective, modality

Anything absent is left to the documented defaults in :mod:`psfselect.metadata`
(e.g. specimen RI ns -> 1.35 for zebrafish tissue), and flagged there.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io_utils import LOG, read_voxel_um_from_tiff
from .metadata import SampleMetadata

# Immersion medium -> refractive index (used if RefractionIndex is absent).
IMMERSION_RI = {
    "DRY": 1.0, "AIR": 1.0,
    "WATER": 1.333, "WI": 1.333, "W": 1.333,
    "OIL": 1.515, "OEL": 1.515,
    "GLYCEROL": 1.45, "GLYC": 1.45, "GLY": 1.45,
    "SILICONE": 1.405, "SIL": 1.405,
}

# Fluorophore -> emission peak (nm). Used so we don't mistake Leica's
# "EmissionWavelengthForPinholeAiryCalculation" for the real emission.
DYE_EMISSION_NM = {
    "EGFP": 509, "GFP": 509, "EYFP": 527, "YFP": 527, "CFP": 477, "ECFP": 475,
    "DSRED": 583, "DSRED2": 583, "RFP": 600, "MRFP": 607, "MCHERRY": 610,
    "MSCARLET": 600, "TDTOMATO": 581, "TOMATO": 581, "MKATE": 633, "MKATE2": 633,
    "DAPI": 461, "HOECHST": 461, "FITC": 519, "TRITC": 576,
    "ALEXA488": 519, "ALEXA546": 573, "ALEXA555": 565, "ALEXA568": 603,
    "ALEXA594": 617, "ALEXA633": 647, "ALEXA647": 668, "CY3": 570, "CY5": 670,
    "TEXASRED": 615, "MNEONGREEN": 517,
}


def _read_info_text(path: Path) -> tuple[str, str | None]:
    """Return (metadata_text, series_name) from a TIFF's ImageJ/OME metadata."""
    import tifffile

    text = ""
    with tifffile.TiffFile(path) as tif:
        if tif.is_imagej and tif.imagej_metadata and "Info" in tif.imagej_metadata:
            text += str(tif.imagej_metadata["Info"]) + "\n"
        if getattr(tif, "is_ome", False) and tif.ome_metadata:
            text += str(tif.ome_metadata) + "\n"
        if not text and tif.pages:
            desc = tif.pages[0].tags.get("ImageDescription")
            if desc:
                text += str(desc.value)
    m = re.search(r"(Series\d+)", path.name)
    return text, (m.group(1) if m else None)


def _series_lines(text: str, series: str | None) -> list[str]:
    lines = text.splitlines()
    if series:
        sel = [ln for ln in lines if ln.strip().startswith(series)]
        if sel:
            return sel
    return lines


def _find(lines: list[str], needle: str) -> str | None:
    """First value whose key ends with ``needle`` (matched before ' = ')."""
    for ln in lines:
        if " = " not in ln:
            continue
        key, val = ln.split(" = ", 1)
        if key.strip().endswith(needle):
            return val.strip()
    return None


def _find_float(lines: list[str], needle: str) -> float | None:
    v = _find(lines, needle)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def _find_int(lines: list[str], needle: str) -> int | None:
    f = _find_float(lines, needle)
    return int(f) if f is not None else None


def _collect_dyes(lines: list[str]) -> list[str]:
    dyes = []
    for ln in lines:
        if "DyeName = " in ln:
            raw = ln.split("DyeName = ", 1)[1].strip()
            name = raw.split("/")[-1].strip()  # "Leica/EGFP" -> "EGFP"
            if name:
                dyes.append(name)
    return dyes


def _emission_from_dyes(dyes: list[str]) -> tuple[float | None, str | None]:
    for d in dyes:
        key = re.sub(r"[^A-Z0-9]", "", d.upper())
        if key in DYE_EMISSION_NM:
            return float(DYE_EMISSION_NM[key]), d
    return (None, dyes[0] if dyes else None)


def extract_metadata(path: str | Path, sample_id: str | None = None) -> tuple[SampleMetadata, dict[str, Any]]:
    """Extract a SampleMetadata + an ``info`` dict of extra context.

    The ``info`` dict carries non-schema details useful to the pipeline, e.g.
    ``immersion`` (string), ``dyes`` (list), ``size_c`` (channel count),
    ``emission_source`` (how the wavelength was determined).
    """
    path = Path(path)
    sid = sample_id or path.stem
    info: dict[str, Any] = {}

    try:
        text, series = _read_info_text(path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not read embedded metadata from %s: %s", path.name, exc)
        text, series = "", None
    lines = _series_lines(text, series)
    info["series"] = series

    na = _find_float(lines, "NumericalAperture")
    objective = _find(lines, "ObjectiveName")
    immersion = _find(lines, "Immersion")
    refr = _find_float(lines, "RefractionIndex")

    # Immersion RI: prefer the explicit RefractionIndex; else map the medium name.
    ni = None
    if refr is not None and refr > 0:
        ni = refr
        info["ni_source"] = "RefractionIndex"
    elif immersion:
        key = immersion.strip().upper()
        ni = IMMERSION_RI.get(key)
        info["ni_source"] = f"Immersion={immersion}" if ni else None

    # Emission wavelength: fluorophore emission, not the pinhole reference.
    dyes = _collect_dyes(lines)
    wl, dye = _emission_from_dyes(dyes)
    if wl is not None:
        info["emission_source"] = f"dye:{dye}"
    else:
        wl = _find_float(lines, "EmissionWavelengthForPinholeAiryCalculation")
        if wl is not None:
            info["emission_source"] = "pinhole-airy-reference (approx)"

    # Voxel sizes (robust TIFF reader first).
    vox = read_voxel_um_from_tiff(path)
    dz = dy = dx = None
    if vox is not None:
        dz, dy, dx = vox

    size_x = _find_int(lines, "SizeX")
    size_y = _find_int(lines, "SizeY")
    size_z = _find_int(lines, "SizeZ")
    size_c = _find_int(lines, "SizeC")

    info.update({
        "immersion": immersion, "objective": objective, "dyes": dyes,
        "size_c": size_c, "na": na, "ni": ni, "emission_nm": wl,
    })

    meta = SampleMetadata(
        sample_id=sid,
        modality="confocal",
        wavelength_nm=wl,
        na=na,
        ni=ni,
        ni0=ni,
        ns=None,                 # not in metadata -> defaults to tissue ~1.35
        voxel_x_um=dx,
        voxel_y_um=dy,
        voxel_z_um=dz,
        image_x=size_x,
        image_y=size_y,
        image_z=size_z,
        objective=objective,
        image_path=str(path),
        notes=(f"immersion={immersion}; dyes={','.join(dyes) if dyes else 'n/a'}; "
               f"channels={size_c}"),
    )
    return meta, info
