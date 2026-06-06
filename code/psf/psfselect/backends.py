"""PSF rendering backends and a unified :func:`render_psf` entry point.

Two backends:

* ``epfl``      — the EPFL PSF Generator JAR (see :mod:`psfselect.psfgenerator`).
                  Physically authoritative; requires Java + the JAR.
* ``psfmodels`` — pure-Python theoretical PSFs via the ``psfmodels`` package.
                  Always available, so the whole pipeline (and the bundled
                  end-to-end example) runs without Java.

Model name -> psfmodels mapping used by the fallback:

* ``born_wolf``        -> scalar Gibson-Lanni with matched RIs (no stratified
                          aberration) — a faithful stand-in for Born & Wolf.
* ``gibson_lanni``     -> scalar Gibson-Lanni (stratified media).
* ``richards_wolf``    -> vectorial high-NA diffraction.
* ``vri_gibson_lanni`` -> scalar Gibson-Lanni evaluated with the specimen RI set
                          to a depth-adjusted value (a lightweight approximation
                          of variable-RI behaviour).

A final ``gaussian`` safety net renders an analytic Gaussian PSF if psfmodels is
unavailable, so the code never hard-fails for lack of an optional dependency.
"""

from __future__ import annotations

import numpy as np

from . import MODELS
from .io_utils import LOG
from .parameters import PSFParams


_EPFL_AVAILABLE: bool | None = None  # cached so auto-mode doesn't probe per render


def available_backends() -> dict[str, bool]:
    global _EPFL_AVAILABLE
    info = {"epfl": False, "psfmodels": False}
    try:
        from .psfgenerator import resolve_jar

        resolve_jar()
        info["epfl"] = True
    except Exception:
        info["epfl"] = False
    _EPFL_AVAILABLE = info["epfl"]
    try:
        import psfmodels  # noqa: F401

        info["psfmodels"] = True
    except Exception:
        info["psfmodels"] = False
    return info


# Models the EPFL standalone JAR does not render reliably. PSF Generator's
# Variable-RI Gibson & Lanni frequently fails to terminate (it prints
# "Computing VRIGL" and then runs effectively forever, regardless of sampling),
# so in 'auto' mode we skip the JAR for it and use the instant psfmodels
# approximation instead. Force it with backend='epfl' (and a long timeout) only
# if your JAR build is known to handle VRIGL.
EPFL_UNRELIABLE_MODELS = {"vri_gibson_lanni"}


def render_psf(
    model: str,
    params: PSFParams,
    *,
    backend: str = "auto",
    jar_path=None,
    workdir=None,
    timeout_s: int = 180,
) -> tuple[np.ndarray, str, str | None]:
    """Render a single PSF.

    Returns ``(volume_zyx, backend_used, config_text_or_None)``. The volume is
    float32, normalised to peak 1. ``backend='auto'`` prefers the EPFL JAR when
    available and silently falls back to psfmodels, then gaussian. In 'auto'
    mode the JAR is skipped for models in :data:`EPFL_UNRELIABLE_MODELS`.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Known: {MODELS}")

    if backend == "auto":
        global _EPFL_AVAILABLE
        if _EPFL_AVAILABLE is None:
            available_backends()  # populates the cache once
            if not _EPFL_AVAILABLE:
                LOG.info("EPFL PSF Generator JAR not found; using psfmodels backend "
                         "(set PSF_GENERATOR_JAR to use the EPFL Generator).")
        use_epfl = _EPFL_AVAILABLE and model not in EPFL_UNRELIABLE_MODELS
        if _EPFL_AVAILABLE and model in EPFL_UNRELIABLE_MODELS:
            LOG.info("Model '%s' is skipped on the EPFL JAR (it does not reliably "
                     "terminate); using psfmodels for it.", model)
        order = (["epfl"] if use_epfl else []) + ["psfmodels", "gaussian"]
    else:
        order = [backend]
    last_err: Exception | None = None
    for be in order:
        try:
            if be == "epfl":
                from .psfgenerator import build_config, run_psfgenerator

                vol, _cfg = run_psfgenerator(model, params, jar_path=jar_path,
                                            workdir=workdir, timeout_s=timeout_s)
                return vol, "epfl", build_config(model, params)
            if be == "psfmodels":
                return _render_psfmodels(model, params), "psfmodels", None
            if be == "gaussian":
                return _render_gaussian(model, params), "gaussian", None
            raise ValueError(f"Unknown backend '{be}'")
        except Exception as exc:  # noqa: BLE001 - fall through to next backend
            last_err = exc
            if backend == "auto":
                LOG.info("Backend '%s' unavailable (%s); trying next.", be, type(exc).__name__)
                continue
            raise
    raise RuntimeError(f"All backends failed for model {model}: {last_err}")


# --------------------------------------------------------------------------- #
def _render_psfmodels(model: str, p: PSFParams) -> np.ndarray:
    import psfmodels as psfm

    if model == "richards_wolf":
        pm_model = "vectorial"
    else:
        pm_model = "scalar"

    ns = p.ns
    ni = p.ni
    ni0 = p.ni0
    if model == "born_wolf":
        # No stratified-media aberration: match design to in-use RIs.
        ns = p.ni
        ni0 = p.ni
    elif model == "vri_gibson_lanni":
        # Depth-adjusted specimen RI: nudge ns toward ni with depth to emulate a
        # smoothly varying refractive index along z (bounded, monotone).
        depth = max(p.particle_depth_um, 0.0)
        frac = depth / (depth + 50.0)  # 0 at surface -> ~0.5 at 50 um
        ns = p.ns + (p.ni - p.ns) * frac

    nz = int(p.nz)
    nx = int(p.nx)
    vol = psfm.make_psf(
        nz,
        nx,
        dxy=p.voxel_xy_um,
        dz=p.voxel_z_um,
        NA=p.na,
        wvl=p.lambda_um(),
        ns=ns,
        ni=ni,
        ni0=ni0,
        tg=p.coverslip_um,
        tg0=p.coverslip_design_um,
        ng=p.ng,
        ng0=p.ng0,
        ti0=p.working_distance_um,
        pz=max(p.particle_depth_um, 0.0),
        normalize=True,
        model=pm_model,
    )
    vol = np.asarray(vol, dtype=np.float32)
    return vol / max(float(vol.max()), 1e-12)


def _render_gaussian(model: str, p: PSFParams) -> np.ndarray:
    """Analytic separable Gaussian PSF (last-resort backend).

    Lateral sigma from the diffraction limit (0.21 lambda / NA), axial sigma
    from 0.66 n lambda / NA^2; both converted to voxels. Richards & Wolf is
    given a slightly tighter lateral lobe to keep models distinguishable.
    """
    lam = p.lambda_um()
    fwhm_lat = 0.51 * lam / p.na
    fwhm_ax = 1.77 * p.ni * lam / (p.na ** 2)
    if model == "richards_wolf":
        fwhm_lat *= 0.95
    if model in ("gibson_lanni", "vri_gibson_lanni"):
        # depth broadens the axial lobe a touch
        fwhm_ax *= 1.0 + 0.002 * max(p.particle_depth_um, 0.0)
    to_sigma = 1.0 / 2.3548
    sx = (fwhm_lat / p.voxel_xy_um) * to_sigma
    sz = (fwhm_ax / p.voxel_z_um) * to_sigma

    nz, nx = int(p.nz), int(p.nx)
    z = np.arange(nz) - nz // 2
    y = np.arange(nx) - nx // 2
    x = np.arange(nx) - nx // 2
    gz = np.exp(-0.5 * (z / max(sz, 1e-6)) ** 2)
    gy = np.exp(-0.5 * (y / max(sx, 1e-6)) ** 2)
    gx = np.exp(-0.5 * (x / max(sx, 1e-6)) ** 2)
    vol = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    return (vol / vol.max()).astype(np.float32)
