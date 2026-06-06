"""Visualisation: orthogonal PSF views, FWHM/score summaries, sweep trends.

Uses a non-interactive matplotlib backend so figures render headlessly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .io_utils import ensure_dir, load_volume  # noqa: E402


def _ortho_slices(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pk = np.unravel_index(int(np.argmax(vol)), vol.shape)
    zc, yc, xc = pk
    xy = vol[zc]            # XY plane
    xz = vol[:, yc, :]      # XZ plane (z rows, x cols)
    yz = vol[:, :, xc]      # YZ plane (z rows, y cols)
    return xy, xz, yz


def plot_psf_orthoviews(vol: np.ndarray, title: str, out_path: str | Path,
                        voxel_um: tuple[float, float, float] | None = None,
                        log_scale: bool = True) -> Path:
    """XY / XZ / YZ orthogonal views of a single PSF."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    xy, xz, yz = _ortho_slices(vol)

    def prep(a):
        a = a / max(a.max(), 1e-12)
        if log_scale:
            a = np.log10(np.clip(a, 1e-4, 1.0))
        return a

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, name in zip(axes, (prep(xy), prep(xz), prep(yz)), ("XY", "XZ", "YZ")):
        im = ax.imshow(img, cmap="inferno", aspect="auto", origin="lower")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title + ("  (log10 intensity)" if log_scale else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_orthoviews_from_candidate(candidate, out_dir: str | Path) -> Path | None:
    if not candidate.psf_path or not Path(candidate.psf_path).exists():
        return None
    vol = load_volume(candidate.psf_path)
    title = f"{candidate.model}  |  {candidate.candidate_id}"
    return plot_psf_orthoviews(
        vol, title, Path(out_dir) / f"{candidate.candidate_id}_ortho.png",
        voxel_um=tuple(candidate.voxel_um),
    )


def plot_model_comparison(df: pd.DataFrame, out_path: str | Path,
                          sample_id: str | None = None) -> Path:
    """Bar charts comparing models on FWHM, anisotropy and composite score."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    sub = df if sample_id is None else df[df["sample_id"] == sample_id]
    # One row per model: take the best composite per model.
    agg = (sub.sort_values("composite", ascending=False)
              .groupby("model", as_index=False).first())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(agg["model"], agg["fwhm_lateral_um"], color="#4477aa", label="lateral")
    axes[0].bar(agg["model"], agg["fwhm_z_um"], color="#ee6677", alpha=0.6, label="axial")
    axes[0].set_title("FWHM (um)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(agg["model"], agg["anisotropy"], color="#228833")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_title("Anisotropy (axial / lateral)")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(agg["model"], agg["composite"], color="#aa3377")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Composite score")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Model comparison{' — ' + sample_id if sample_id else ''}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_score_breakdown(df: pd.DataFrame, out_path: str | Path,
                         sample_id: str | None = None) -> Path:
    """Grouped bars of plausibility / stability / reblur per model."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    sub = df if sample_id is None else df[df["sample_id"] == sample_id]
    agg = (sub.sort_values("composite", ascending=False)
              .groupby("model", as_index=False).first())
    models = agg["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(("plausibility", "stability", "reblur")):
        vals = agg[key].fillna(0).to_numpy()
        ax.bar(x + (i - 1) * width, vals, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30)
    ax.set_ylim(0, 1)
    ax.set_ylabel("sub-score")
    ax.set_title(f"Score breakdown{' — ' + sample_id if sample_id else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_sweep_trends(df: pd.DataFrame, axis: str, out_path: str | Path,
                      sample_id: str | None = None) -> Path | None:
    """Trend of FWHM / composite vs a swept parameter, one line per model."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    sub = df if sample_id is None else df[df["sample_id"] == sample_id]
    if axis not in sub.columns or sub[axis].nunique() < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values(axis)
        axes[0].plot(g[axis], g["fwhm_z_um"], marker="o", label=f"{model} axial")
        axes[0].plot(g[axis], g["fwhm_lateral_um"], marker="s", ls="--", label=f"{model} lat")
        axes[1].plot(g[axis], g["composite"], marker="o", label=model)
    axes[0].set_xlabel(axis)
    axes[0].set_ylabel("FWHM (um)")
    axes[0].set_title("FWHM vs " + axis)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(axis)
    axes[1].set_ylabel("composite score")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Score vs " + axis)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
