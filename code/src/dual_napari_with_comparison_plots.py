"""
Open two napari windows (raw vs deconvolved stacks) and one matplotlib figure
with all comparison subplots (histogram, box, violin, ECDF, KDE, median shift).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import napari

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from compare_segmentation_original_vs_deconvolved import (  # noqa: E402
    populations_long_and_summary,
)
from view_tif import setup_viewer_layers  # noqa: E402


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path, Path]:
    raw = (
        repo_root
        / "data/raw/cmlc2_lifeactXnuclear/48hpf"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
    )
    dec = (
        repo_root
        / "deconvolution_results"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005_deconvolved_merged.tif"
    )
    orig_csv = (
        repo_root
        / "segmentation_comparison"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005_nuclei_features.csv"
    )
    dec_csv = (
        repo_root
        / "segmentation_comparison"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005_deconvolved_merged_nuclei_features.csv"
    )
    return raw, dec, orig_csv, dec_csv


def _extended_six(long: pd.DataFrame) -> list[str]:
    extended_6 = [
        "volume_voxels",
        "solidity",
        "equivalent_diameter",
        "aspect_ratio",
        "extent",
        "shape_major_axis",
    ]
    extended_6 = [c for c in extended_6 if c in long.columns]
    if len(extended_6) < 6:
        for fallback in ("bbox_y_length", "bbox_z_length", "equivalent_diameter"):
            if fallback in long.columns and fallback not in extended_6:
                extended_6.append(fallback)
        seen: set[str] = set()
        extended_6 = [x for x in extended_6 if not (x in seen or seen.add(x))][:6]
    return extended_6


def show_master_comparison_figure(long: pd.DataFrame, summary: pd.DataFrame | None) -> None:
    """Single figure: 4×4 core panels + 3×2 KDE + horizontal bar (median % change)."""
    sns.set_theme(style="whitegrid")
    palette = {"original": "#4a6fa5", "deconvolved": "#c87941"}

    primary_4 = ["volume_voxels", "solidity", "equivalent_diameter", "aspect_ratio"]
    primary_4 = [c for c in primary_4 if c in long.columns]
    extended_6 = _extended_six(long)

    fig = plt.figure(figsize=(18, 28))
    gs = fig.add_gridspec(8, 4, hspace=0.45, wspace=0.32, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1.1])

    fig.suptitle(
        "Segmentation comparison: original vs deconvolved (same pipeline)",
        fontsize=15,
        y=0.998,
    )

    titles_row = ["Density (step histogram)", "Box plot", "Violin", "ECDF"]
    for row, title in enumerate(titles_row):
        for i, col in enumerate(primary_4):
            ax = fig.add_subplot(gs[row, i])
            if row == 0:
                sns.histplot(
                    data=long,
                    x=col,
                    hue="condition",
                    hue_order=["original", "deconvolved"],
                    element="step",
                    stat="density",
                    common_norm=False,
                    palette=palette,
                    ax=ax,
                    legend=(i == 0),
                )
            elif row == 1:
                sns.boxplot(
                    data=long,
                    x="condition",
                    y=col,
                    hue="condition",
                    order=["original", "deconvolved"],
                    hue_order=["original", "deconvolved"],
                    dodge=False,
                    palette=palette,
                    legend=False,
                    ax=ax,
                )
            elif row == 2:
                sns.violinplot(
                    data=long,
                    x="condition",
                    y=col,
                    hue="condition",
                    order=["original", "deconvolved"],
                    hue_order=["original", "deconvolved"],
                    dodge=False,
                    palette=palette,
                    legend=False,
                    cut=0,
                    inner="quartile",
                    ax=ax,
                )
            else:
                _plot_ecdf(ax, long, col, palette)
            ax.set_title(f"{title}: {col.replace('_', ' ')}")
        for i in range(len(primary_4), 4):
            fig.add_subplot(gs[row, i]).set_visible(False)

    # Rows 4–6: KDE dashboard (2 cols × 3 rows = 6 panels), leave right side empty
    for k, col in enumerate(extended_6[:6]):
        r = 4 + k // 2
        c = k % 2
        ax = fig.add_subplot(gs[r, c])
        sns.kdeplot(
            data=long,
            x=col,
            hue="condition",
            hue_order=["original", "deconvolved"],
            palette=palette,
            common_norm=False,
            fill=True,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(f"KDE: {col.replace('_', ' ')}")

    # Row 7: median % change bar
    ax_bar = fig.add_subplot(gs[7, :])
    if summary is not None and not summary.empty:
        s = summary.dropna(subset=["median_original", "median_deconvolved"]).copy()
        s = s[s["median_original"] != 0]
        s["pct_change_median"] = (
            (s["median_deconvolved"] - s["median_original"])
            / s["median_original"].abs()
            * 100.0
        )
        s = s[np.isfinite(s["pct_change_median"])]
        if len(s):
            order = s.sort_values("pct_change_median").feature
            sns.barplot(data=s, y="feature", x="pct_change_median", order=order, ax=ax_bar, color="#567994")
            ax_bar.axvline(0, color="black", linewidth=0.8)
            ax_bar.set_xlabel("% change in median (deconvolved vs original)")
            ax_bar.set_title("Relative shift in medians per feature")
        else:
            ax_bar.text(0.5, 0.5, "No median-change data", ha="center", va="center")
    else:
        ax_bar.text(0.5, 0.5, "No summary table", ha="center", va="center")

    plt.show(block=False)


def _plot_ecdf(ax, long: pd.DataFrame, col: str, palette: dict) -> None:
    for cond, color in (("original", palette["original"]), ("deconvolved", palette["deconvolved"])):
        x = np.sort(long.loc[long["condition"] == cond, col].dropna().to_numpy())
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(
            np.concatenate([[x[0]], x]),
            np.concatenate([[0.0], y]),
            where="post",
            label=cond,
            color=color,
            linewidth=2,
        )
    ax.set_xlabel(col.replace("_", " "))
    ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_tif, dec_tif, def_orig_csv, def_dec_csv = _default_paths(repo_root)

    parser = argparse.ArgumentParser(
        description="Dual napari (raw + deconvolved) + combined matplotlib comparison figure."
    )
    parser.add_argument("--raw", type=str, default=str(raw_tif), help="Raw multi-channel .tif")
    parser.add_argument("--deconvolved", type=str, default=str(dec_tif), help="Deconvolved merged .tif")
    parser.add_argument(
        "--orig-csv",
        type=str,
        default=str(def_orig_csv),
        help="Nuclei features CSV from raw stack",
    )
    parser.add_argument(
        "--dec-csv",
        type=str,
        default=str(def_dec_csv),
        help="Nuclei features CSV from deconvolved stack",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Also run 3D segmentation overlay in each napari window (slow)",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    dec_path = Path(args.deconvolved)
    orig_csv = Path(args.orig_csv)
    dec_csv = Path(args.dec_csv)

    for p in (raw_path, dec_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing .tif: {p}")
    for p in (orig_csv, dec_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing feature CSV: {p}\n"
                "Run: ./venv/bin/python src/compare_segmentation_original_vs_deconvolved.py"
            )

    df_o = pd.read_csv(orig_csv)
    df_d = pd.read_csv(dec_csv)
    long, summary, n_o, n_d = populations_long_and_summary(df_o, df_d)
    print(f"Loaded features: original n={n_o}, deconvolved n={n_d}")

    show_master_comparison_figure(long, summary)

    v1 = napari.Viewer(title=f"Raw — {raw_path.name}")
    setup_viewer_layers(v1, raw_path, segment=args.segment)

    v2 = napari.Viewer(title=f"Deconvolved — {dec_path.name}")
    setup_viewer_layers(v2, dec_path, segment=args.segment)

    print("Close napari windows to exit.")
    napari.run()


if __name__ == "__main__":
    main()
