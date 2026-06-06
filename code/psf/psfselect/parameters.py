"""PSF parameter management: optical parameter sets and parameter sweeps.

A :class:`PSFParams` bundle holds everything needed to render one PSF volume. It
can be built from a :class:`~psfselect.metadata.SampleMetadata` instance, edited,
and expanded into a grid of candidates via :func:`expand_sweep`.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .metadata import SampleMetadata


@dataclass
class PSFParams:
    """Optical + sampling parameters for a single PSF render.

    Distances are in microns, wavelength in nm, angles via NA. These map onto
    both the EPFL PSF Generator config keys and the psfmodels API.
    """

    # Optics
    na: float = 0.8
    wavelength_nm: float = 600.0
    ni: float = 1.33                 # immersion RI (in use)
    ni0: float = 1.33                # design immersion RI
    ns: float = 1.35                 # specimen RI (at coverslip; VRIGL NS1)
    ns2: float | None = None         # specimen RI at depth (VRIGL NS2); None -> = ns
    ng: float = 1.515
    ng0: float = 1.515
    coverslip_um: float = 170.0
    coverslip_design_um: float = 170.0
    working_distance_um: float = 2000.0
    particle_depth_um: float = 0.0   # depth of point source in specimen (z-position)

    # Sampling / grid
    voxel_xy_um: float = 0.1
    voxel_z_um: float = 0.25
    nx: int = 128                    # lateral size (square)
    nz: int = 64                     # axial size (odd recommended)

    # Bookkeeping
    sample_id: str | None = None

    def lambda_um(self) -> float:
        return self.wavelength_nm / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def copy_with(self, **overrides: Any) -> "PSFParams":
        data = self.to_dict()
        data.update(overrides)
        return PSFParams(**data)


def params_from_metadata(meta: SampleMetadata, *, nx: int = 128, nz: int = 64) -> PSFParams:
    """Build a PSFParams bundle from (defaulted) sample metadata.

    Lateral voxel uses the mean of x/y. Axial size is grown if the metadata
    image is large, but capped to keep renders cheap.
    """
    m = meta if not meta.missing_fields and meta.applied_defaults == {} else meta.with_defaults()
    vox_xy = float(((m.voxel_x_um or 0.1) + (m.voxel_y_um or m.voxel_x_um or 0.1)) / 2.0)
    return PSFParams(
        na=float(m.na),
        wavelength_nm=float(m.wavelength_nm),
        ni=float(m.ni),
        ni0=float(m.ni0),
        ns=float(m.ns),
        ng=float(m.ng),
        ng0=float(m.ng0),
        coverslip_um=float(m.coverslip_um),
        coverslip_design_um=float(m.coverslip_design_um),
        working_distance_um=float(m.working_distance_um),
        particle_depth_um=float(m.imaging_depth_um or 0.0),
        voxel_xy_um=vox_xy,
        voxel_z_um=float(m.voxel_z_um),
        nx=nx,
        nz=nz,
        sample_id=m.sample_id,
    )


# Names allowed in a sweep specification, mapped to the PSFParams attribute.
SWEEPABLE = {
    "na": "na",
    "wavelength_nm": "wavelength_nm",
    "voxel_xy_um": "voxel_xy_um",
    "voxel_z_um": "voxel_z_um",
    "nx": "nx",
    "nz": "nz",
    "ni": "ni",
    "ns": "ns",
    "ni0": "ni0",
    "particle_depth_um": "particle_depth_um",
}


@dataclass
class SweepSpec:
    """A parameter sweep: each named axis maps to a list of values.

    ``expand`` produces the Cartesian product of all axes applied on top of a
    base :class:`PSFParams`.
    """

    axes: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SweepSpec":
        axes: dict[str, list[Any]] = {}
        for key, vals in (d or {}).items():
            if key not in SWEEPABLE:
                raise ValueError(f"Unknown sweep axis '{key}'. Allowed: {sorted(SWEEPABLE)}")
            axes[key] = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        return cls(axes=axes)


def expand_sweep(base: PSFParams, sweep: SweepSpec | None) -> list[PSFParams]:
    """Expand a base parameter set over a sweep into a list of PSFParams.

    Returns ``[base]`` when there is no sweep. Each resulting params carries the
    same ``sample_id`` as the base.
    """
    if sweep is None or not sweep.axes:
        return [base]
    keys = list(sweep.axes)
    grids = [sweep.axes[k] for k in keys]
    out: list[PSFParams] = []
    for combo in itertools.product(*grids):
        overrides = dict(zip(keys, combo))
        out.append(base.copy_with(**overrides))
    return out


def perturbations(base: PSFParams, *, rel: float = 0.05,
                  knobs: Iterable[str] = ("na", "ni", "ns", "particle_depth_um")) -> list[tuple[str, PSFParams]]:
    """Return small ± perturbations of ``base`` for stability analysis.

    Each tuple is ``(label, params)``. Depth uses an absolute step (relative
    perturbation is meaningless at depth 0).
    """
    out: list[tuple[str, PSFParams]] = []
    for knob in knobs:
        val = getattr(base, knob)
        if knob == "particle_depth_um":
            step = 5.0  # microns
            out.append((f"{knob}+{step}", base.copy_with(**{knob: val + step})))
            out.append((f"{knob}-{step}", base.copy_with(**{knob: max(0.0, val - step)})))
        else:
            out.append((f"{knob}+{rel:.0%}", base.copy_with(**{knob: val * (1 + rel)})))
            out.append((f"{knob}-{rel:.0%}", base.copy_with(**{knob: val * (1 - rel)})))
    return out


def load_sweep_file(path: str | Path) -> SweepSpec:
    """Load a sweep spec from YAML/JSON (a mapping of axis -> list of values)."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
    else:
        from .io_utils import read_json

        d = read_json(path)
    if isinstance(d, dict) and "sweep" in d:
        d = d["sweep"]
    return SweepSpec.from_dict(d)
