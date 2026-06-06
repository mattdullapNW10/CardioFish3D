"""EPFL PSF Generator wrapper: write config files and call the standalone JAR.

PSF Generator (https://bigwww.epfl.ch/algorithms/psfgenerator/) runs as a Java
application that reads a human-readable config file and writes a 3D PSF as a
TIFF stack:

    java -cp PSFGenerator.jar PSFGenerator config.txt

This module turns a :class:`~psfselect.parameters.PSFParams` bundle into a valid
config file, invokes the JAR, and loads the resulting volume. The JAR location
is resolved from (in order): the ``jar_path`` argument, the ``PSF_GENERATOR_JAR``
environment variable. If neither is available, :func:`run_psfgenerator` raises
:class:`PSFGeneratorNotFound` — callers can then fall back to the psfmodels
backend (see :mod:`psfselect.backends`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import EPFL_SHORTNAMES
from .io_utils import LOG, load_volume
from .parameters import PSFParams


class PSFGeneratorNotFound(RuntimeError):
    """Raised when the PSF Generator JAR cannot be located or run."""


def resolve_jar(jar_path: str | Path | None = None) -> Path:
    candidate = jar_path or os.environ.get("PSF_GENERATOR_JAR")
    if not candidate:
        raise PSFGeneratorNotFound(
            "PSF Generator JAR not found. Set PSF_GENERATOR_JAR or pass jar_path. "
            "Download from https://bigwww.epfl.ch/algorithms/psfgenerator/"
        )
    p = Path(candidate).expanduser()
    if not p.exists():
        raise PSFGeneratorNotFound(f"PSF Generator JAR does not exist: {p}")
    return p


def build_config(model: str, params: PSFParams, *, accuracy: str = "Good",
                 lut: str = "Grays") -> str:
    """Return the text of a PSF Generator config file for ``model``/``params``.

    Key names follow the official EPFL PSF Generator config format (the
    ``PSF-shortname`` selects which model the JAR computes; all four model blocks
    are written so the file is self-describing). ``accuracy`` is the numerical
    integration setting (``Good`` | ``Better`` | ``Best``).
    """
    if model not in EPFL_SHORTNAMES:
        raise ValueError(f"Unknown model '{model}'. Known: {sorted(EPFL_SHORTNAMES)}")
    short = EPFL_SHORTNAMES[model]

    # Units expected by PSF Generator (per the EPFL reference config):
    #   ResLateral / ResAxial / Lambda / ZPos : nanometres
    #   TI (working distance) / TG (coverslip) : micrometres
    #   NI / NS / NG : refractive indices (dimensionless)
    #   NA, accuracy(Good|Better|Best) : global / per-model
    res_lat_nm = params.voxel_xy_um * 1000.0
    res_ax_nm = params.voxel_z_um * 1000.0
    lam_nm = params.wavelength_nm
    zpos_nm = params.particle_depth_um * 1000.0
    ti_um = params.working_distance_um
    ns1 = params.ns                                  # specimen RI at coverslip
    ns2 = params.ns2 if params.ns2 is not None else params.ns  # specimen RI at depth

    lines: list[str] = [
        "#PSFGenerator",
        f"PSF-shortname={short}",
        f"ResLateral={res_lat_nm:.4f}",
        f"ResAxial={res_ax_nm:.4f}",
        f"NX={int(params.nx)}",
        f"NY={int(params.nx)}",
        f"NZ={int(params.nz)}",
        "Type=32-bits",
        f"NA={params.na:.4f}",
        f"LUT={lut}",
        f"Lambda={lam_nm:.4f}",
        "Scale=Linear",
        # Born & Wolf (in-focus scalar): needs only immersion RI + accuracy.
        f"psf-BW-NI={params.ni:.4f}",
        f"psf-BW-accuracy={accuracy}",
        # Richards & Wolf (vectorial).
        f"psf-RW-NI={params.ni:.4f}",
        f"psf-RW-accuracy={accuracy}",
        # Gibson & Lanni (3-layer scalar).
        f"psf-GL-NI={params.ni:.4f}",
        f"psf-GL-NS={params.ns:.4f}",
        f"psf-GL-accuracy={accuracy}",
        f"psf-GL-ZPos={zpos_nm:.4f}",
        f"psf-GL-TI={ti_um:.4f}",
        # Variable-RI Gibson & Lanni (axially-varying specimen RI).
        f"psf-VRIGL-NI={params.ni:.4f}",
        f"psf-VRIGL-accuracy={accuracy}",
        f"psf-VRIGL-NS1={ns1:.4f}",
        f"psf-VRIGL-NS2={ns2:.4f}",
        f"psf-VRIGL-NG={params.ng:.4f}",
        f"psf-VRIGL-TG={params.coverslip_um:.4f}",
        f"psf-VRIGL-TI={ti_um:.4f}",
        "psf-VRIGL-RIvary=Linear",
        f"psf-VRIGL-ZPos={zpos_nm:.4f}",
    ]
    return "\n".join(lines) + "\n"


def run_psfgenerator(
    model: str,
    params: PSFParams,
    *,
    jar_path: str | Path | None = None,
    workdir: str | Path | None = None,
    java_bin: str = "java",
    timeout_s: int = 180,
    keep_config: bool = True,
    headless: bool = True,
) -> tuple[np.ndarray, Path]:
    """Generate one PSF with the EPFL JAR. Returns ``(volume_zyx, config_path)``.

    Invokes ``java -cp PSFGenerator.jar PSFGenerator <config>`` (the documented
    standalone/batch entry point). The JAR writes a TIFF into the working
    directory; we locate the newest ``.tif`` produced and load it as a ZYX volume
    normalised to max 1. ``headless`` adds ``-Djava.awt.headless=true`` so no GUI
    window appears during batch generation.
    """
    jar = resolve_jar(jar_path)
    if shutil.which(java_bin) is None:
        raise PSFGeneratorNotFound(f"Java executable '{java_bin}' not found on PATH")

    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="psfgen_"))
    wd.mkdir(parents=True, exist_ok=True)
    cfg_path = wd / f"config_{model}.txt"
    cfg_path.write_text(build_config(model, params))

    before = {p.name for p in wd.glob("*.tif")} | {p.name for p in wd.glob("*.tiff")}
    cmd = [java_bin]
    if headless:
        cmd.append("-Djava.awt.headless=true")
    cmd += ["-cp", str(jar), "PSFGenerator", str(cfg_path)]
    LOG.info("Running PSF Generator: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise PSFGeneratorNotFound(
            f"PSF Generator failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    produced = sorted(
        [p for p in (*wd.glob("*.tif"), *wd.glob("*.tiff")) if p.name not in before],
        key=lambda p: p.stat().st_mtime,
    )
    if not produced:
        raise PSFGeneratorNotFound(
            f"PSF Generator produced no TIFF in {wd}. STDOUT:\n{proc.stdout}"
        )
    out_tif = produced[-1]
    vol = load_volume(out_tif)
    vol = vol / max(float(vol.max()), 1e-12)
    if not keep_config:
        cfg_path.unlink(missing_ok=True)
    return vol.astype(np.float32), cfg_path
