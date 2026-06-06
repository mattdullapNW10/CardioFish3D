"""PSF shape metrics computed directly from a rendered volume.

All metrics operate on a ZYX float volume (peak need not be normalised). FWHMs
are returned both in voxels and in microns when voxel sizes are supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PSFMetrics:
    # Lateral / axial full-width at half maximum
    fwhm_x_um: float
    fwhm_y_um: float
    fwhm_z_um: float
    fwhm_lateral_um: float          # mean of x,y
    # Anisotropy = axial / lateral FWHM (1.0 = isotropic; >1 = axially elongated)
    anisotropy: float
    # Fraction of total energy within one lateral FWHM radius of the peak
    energy_concentration: float
    # Axial side-lobe ratio: secondary axial peak / main peak (vectorial/high-NA
    # and aberrated PSFs show stronger lobes). 0 = none detected.
    axial_sidelobe_ratio: float
    # Lateral radial symmetry (1.0 = perfectly circular in XY at focal plane)
    lateral_symmetry: float
    # Peak offset from geometric centre, in voxels (z, y, x); large = mis-centred
    peak_offset_vox: tuple[float, float, float]
    # Whether the half-max contour is fully contained in the volume (not clipped)
    contained: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["peak_offset_vox"] = list(self.peak_offset_vox)
        return d


def _fwhm_1d(profile: np.ndarray, spacing: float) -> tuple[float, bool]:
    """FWHM of a 1D profile (peak-normalised internally) by linear interpolation.

    Returns ``(fwhm_in_units, contained)`` where ``contained`` is False if the
    half-max crossing runs off either end of the profile.
    """
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 3 or p.max() <= 0:
        return float("nan"), False
    p = p / p.max()
    peak = int(np.argmax(p))
    half = 0.5

    def cross(idxs):
        prev = peak
        for i in idxs:
            if p[i] < half:
                # interpolate between prev and i
                x0, x1 = prev, i
                y0, y1 = p[prev], p[i]
                if y1 == y0:
                    return float(i)
                return x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            prev = i
        return None

    left = cross(range(peak - 1, -1, -1))
    right = cross(range(peak + 1, p.size))
    contained = left is not None and right is not None
    if left is None:
        left = 0.0
    if right is None:
        right = float(p.size - 1)
    return float(abs(right - left) * spacing), contained


def compute_metrics(vol: np.ndarray, voxel_um: tuple[float, float, float]) -> PSFMetrics:
    """Compute the full metric set. ``voxel_um`` is (dz, dy, dx)."""
    vol = np.asarray(vol, dtype=np.float64)
    dz, dy, dx = voxel_um
    nz, ny, nx = vol.shape

    # Peak (use the brightest voxel as the PSF centre).
    pk = np.unravel_index(int(np.argmax(vol)), vol.shape)
    zc, yc, xc = pk

    # 1D profiles through the peak.
    fz, cz = _fwhm_1d(vol[:, yc, xc], dz)
    fy, cy = _fwhm_1d(vol[zc, :, xc], dy)
    fx, cx = _fwhm_1d(vol[zc, yc, :], dx)
    fwhm_lat = float(np.nanmean([fx, fy]))
    aniso = fz / fwhm_lat if fwhm_lat and not np.isnan(fwhm_lat) else float("nan")

    # Energy concentration within one lateral-FWHM radius (in-plane) at focus.
    plane = vol[zc]
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_um = np.sqrt(((yy - yc) * dy) ** 2 + ((xx - xc) * dx) ** 2)
    radius = fwhm_lat if (fwhm_lat and not np.isnan(fwhm_lat)) else (3 * dx)
    total = float(plane.sum())
    inside = float(plane[r_um <= radius].sum())
    energy_conc = inside / total if total > 0 else float("nan")

    # Axial side-lobe ratio: largest local maximum away from the main peak on the
    # axial profile.
    axial = vol[:, yc, xc] / max(vol[:, yc, xc].max(), 1e-12)
    sidelobe = _sidelobe_ratio(axial, main_idx=zc)

    # Lateral symmetry: compare radial profile variance across angles. Cheap
    # proxy = ratio of min/max of the four cardinal half-profiles' FWHM.
    lat_sym = _lateral_symmetry(plane, yc, xc)

    offset = (float(zc - nz / 2), float(yc - ny / 2), float(xc - nx / 2))
    contained = bool(cz and cy and cx)

    return PSFMetrics(
        fwhm_x_um=fx,
        fwhm_y_um=fy,
        fwhm_z_um=fz,
        fwhm_lateral_um=fwhm_lat,
        anisotropy=aniso,
        energy_concentration=energy_conc,
        axial_sidelobe_ratio=sidelobe,
        lateral_symmetry=lat_sym,
        peak_offset_vox=offset,
        contained=contained,
    )


def _sidelobe_ratio(profile: np.ndarray, main_idx: int) -> float:
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 5:
        return 0.0
    # Find the first minima on each side of the main peak; beyond them, look for
    # the tallest secondary maximum.
    def first_min(idxs):
        prev = p[main_idx]
        for i in idxs:
            if p[i] > prev:
                return i
            prev = p[i]
        return idxs[-1] if len(idxs) else main_idx

    left_min = first_min(list(range(main_idx - 1, -1, -1)))
    right_min = first_min(list(range(main_idx + 1, p.size)))
    secondary = 0.0
    if left_min > 0:
        secondary = max(secondary, float(p[:left_min].max()))
    if right_min < p.size - 1:
        secondary = max(secondary, float(p[right_min + 1:].max()))
    return secondary / max(float(p[main_idx]), 1e-12)


def _lateral_symmetry(plane: np.ndarray, yc: int, xc: int) -> float:
    ny, nx = plane.shape
    up = plane[:yc + 1, xc][::-1]
    down = plane[yc:, xc]
    left = plane[yc, :xc + 1][::-1]
    right = plane[yc, xc:]
    fl_up, _ = _fwhm_1d(up, 1.0)
    fl_down, _ = _fwhm_1d(down, 1.0)
    fl_left, _ = _fwhm_1d(left, 1.0)
    fl_right, _ = _fwhm_1d(right, 1.0)
    vals = [v for v in (fl_up, fl_down, fl_left, fl_right) if v and not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    return float(min(vals) / max(vals))
