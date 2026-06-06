"""
Run 3D nuclei segmentation on the raw and deconvolved stacks, then compare
geometric features (population-level: counts, summaries, tests, plots).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from scipy import stats

from segment_nuclei_3d import extract_channel, extract_features, segment_3d


def _default_paths(repo_root: Path) -> tuple[Path, Path]:
    original = (
        repo_root
        / "data/raw/cmlc2_lifeactXnuclear/48hpf"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
    )
    deconvolved = (
        repo_root
        / "deconvolution_results"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005_deconvolved_merged.tif"
    )
    return original, deconvolved


def segment_stack(tif_path: Path, channel: int, label: str) -> pd.DataFrame:
    print(f"\n=== Segmenting: {label} ({tif_path.name}) ===")
    with tifffile.TiffFile(tif_path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None
    print(f"Shape: {image.shape}, axes: {axes}")
    volume = extract_channel(image, axes, target_channel_index=channel)
    print(f"Nuclear volume shape: {volume.shape}")
    labels = segment_3d(volume)
    df = extract_features(labels, volume)
    df["source"] = label
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["aspect_ratio"] = out["shape_major_axis"] / out["shape_minor_axis"].replace(0, np.nan)
    return out


def populations_long_and_summary(
    df_orig: pd.DataFrame,
    df_dec: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Build long-form dataframe and summary statistics (no file I/O)."""
    a = _add_derived(df_orig.drop(columns=["source"], errors="ignore"))
    b = _add_derived(df_dec.drop(columns=["source"], errors="ignore"))

    numeric_cols = [
        "volume_voxels",
        "solidity",
        "extent",
        "equivalent_diameter",
        "shape_major_axis",
        "shape_intermediate_axis",
        "shape_minor_axis",
        "aspect_ratio",
        "bbox_z_length",
        "bbox_y_length",
        "bbox_x_length",
    ]
    cols = [c for c in numeric_cols if c in a.columns and c in b.columns]

    summary_rows = []
    for col in cols:
        x, y = a[col].dropna().to_numpy(), b[col].dropna().to_numpy()
        row = {
            "feature": col,
            "n_original": len(x),
            "n_deconvolved": len(y),
            "median_original": float(np.median(x)) if len(x) else np.nan,
            "median_deconvolved": float(np.median(y)) if len(y) else np.nan,
            "mean_original": float(np.mean(x)) if len(x) else np.nan,
            "mean_deconvolved": float(np.mean(y)) if len(y) else np.nan,
        }
        if len(x) and len(y):
            row["ks_statistic"], row["ks_pvalue"] = stats.ks_2samp(x, y, method="auto")
            try:
                row["mw_pvalue"] = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
            except ValueError:
                row["mw_pvalue"] = np.nan
        else:
            row["ks_statistic"] = row["ks_pvalue"] = row["mw_pvalue"] = np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    long = pd.concat(
        [
            a.assign(condition="original"),
            b.assign(condition="deconvolved"),
        ],
        ignore_index=True,
    )
    return long, summary, len(a), len(b)


def compare_populations(df_orig: pd.DataFrame, df_dec: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    long, summary, n_o, n_d = populations_long_and_summary(df_orig, df_dec)

    summary_path = out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote summary table: {summary_path}")

    _make_comparison_plots(long, out_dir, summary)

    print("\n--- Counts ---")
    print(f"Nuclei (original):   {n_o}")
    print(f"Nuclei (deconvolved): {n_d}")
    print(f"Delta (deconv - orig): {n_d - n_o}")


def _make_comparison_plots(
    long: pd.DataFrame,
    out_dir: Path,
    summary: pd.DataFrame | None = None,
) -> None:
    """Save matplotlib/seaborn figures comparing original vs deconvolved distributions."""
    sns.set_theme(style="whitegrid")
    palette = {"original": "#4a6fa5", "deconvolved": "#c87941"}

    primary_4 = ["volume_voxels", "solidity", "equivalent_diameter", "aspect_ratio"]
    primary_4 = [c for c in primary_4 if c in long.columns]

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

    # --- 1) Density histograms (step) — key metrics
    n_p = len(primary_4)
    if n_p:
        nrows = int(np.ceil(n_p / 2))
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            "Original vs deconvolved — step histograms (density)",
            fontsize=14,
            y=1.0,
        )
        for i, col in enumerate(primary_4):
            r, cidx = divmod(i, 2)
            ax = axes[r, cidx]
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
            )
            ax.set_title(col.replace("_", " "))
        for j in range(i + 1, nrows * 2):
            r, cidx = divmod(j, 2)
            axes[r, cidx].set_visible(False)
        plt.tight_layout()
        p = out_dir / "comparison_histograms.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {p}")

    # --- 2) Box plots
    if n_p:
        nrows = int(np.ceil(n_p / 2))
        fig2, axes2 = plt.subplots(nrows, 2, figsize=(13, 4.2 * nrows))
        axes2 = np.atleast_2d(axes2)
        fig2.suptitle("Original vs deconvolved — box plots", fontsize=14)
        for i, col in enumerate(primary_4):
            r, cidx = divmod(i, 2)
            ax = axes2[r, cidx]
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
            ax.set_title(col.replace("_", " "))
        for j in range(i + 1, nrows * 2):
            r, cidx = divmod(j, 2)
            axes2[r, cidx].set_visible(False)
        plt.tight_layout()
        p2 = out_dir / "comparison_boxplots.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote: {p2}")

    # --- 3) Violin plots (full distribution shape)
    if n_p:
        fig_v, axes_v = plt.subplots(nrows, 2, figsize=(13, 4.2 * nrows))
        axes_v = np.atleast_2d(axes_v)
        fig_v.suptitle("Original vs deconvolved — violin plots", fontsize=14)
        for i, col in enumerate(primary_4):
            r, cidx = divmod(i, 2)
            ax = axes_v[r, cidx]
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
            ax.set_title(col.replace("_", " "))
        for j in range(i + 1, nrows * 2):
            r, cidx = divmod(j, 2)
            axes_v[r, cidx].set_visible(False)
        plt.tight_layout()
        pv = out_dir / "comparison_violins.png"
        fig_v.savefig(pv, dpi=150, bbox_inches="tight")
        plt.close(fig_v)
        print(f"Wrote: {pv}")

    # --- 4) ECDF curves
    if n_p:
        fig_e, axes_e = plt.subplots(nrows, 2, figsize=(13, 4.2 * nrows))
        axes_e = np.atleast_2d(axes_e)
        fig_e.suptitle("Original vs deconvolved — empirical CDF", fontsize=14)
        for i, col in enumerate(primary_4):
            r, cidx = divmod(i, 2)
            ax = axes_e[r, cidx]
            for cond, color in (("original", palette["original"]), ("deconvolved", palette["deconvolved"])):
                x = np.sort(long.loc[long["condition"] == cond, col].dropna().to_numpy())
                if len(x) == 0:
                    continue
                y = np.arange(1, len(x) + 1) / len(x)
                ax.step(np.concatenate([[x[0]], x]), np.concatenate([[0.0], y]), where="post", label=cond, color=color, linewidth=2)
            ax.set_xlabel(col.replace("_", " "))
            ax.set_ylabel("Cumulative probability")
            ax.legend(loc="lower right")
            ax.set_title(col.replace("_", " "))
        for j in range(i + 1, nrows * 2):
            r, cidx = divmod(j, 2)
            axes_e[r, cidx].set_visible(False)
        plt.tight_layout()
        pe = out_dir / "comparison_ecdf.png"
        fig_e.savefig(pe, dpi=150, bbox_inches="tight")
        plt.close(fig_e)
        print(f"Wrote: {pe}")

    # --- 5) Combined dashboard (3×2)
    n_dash = min(6, len(extended_6))
    if n_dash:
        fig_d, axes_d = plt.subplots(3, 2, figsize=(14, 16))
        fig_d.suptitle("Original vs deconvolved — combined dashboard", fontsize=15, y=0.995)
        for ax, col in zip(axes_d.ravel(), extended_6[:6]):
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
            ax.set_title(col.replace("_", " "))
        for k in range(n_dash, 6):
            axes_d.ravel()[k].set_visible(False)
        plt.tight_layout()
        pd_path = out_dir / "comparison_dashboard_kde.png"
        fig_d.savefig(pd_path, dpi=150, bbox_inches="tight")
        plt.close(fig_d)
        print(f"Wrote: {pd_path}")

    # --- 6) Summary bar chart: median change (%)
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
            fig_b, ax_b = plt.subplots(figsize=(10, max(4, 0.45 * len(s))))
            order = s.sort_values("pct_change_median").feature
            sns.barplot(data=s, y="feature", x="pct_change_median", order=order, ax=ax_b, color="#567994")
            ax_b.axvline(0, color="black", linewidth=0.8)
            ax_b.set_xlabel("% change in median (deconvolved vs original)")
            ax_b.set_title("Relative shift in medians per feature")
            plt.tight_layout()
            pb = out_dir / "comparison_median_change_bar.png"
            fig_b.savefig(pb, dpi=150, bbox_inches="tight")
            plt.close(fig_b)
            print(f"Wrote: {pb}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_orig, default_dec = _default_paths(repo_root)

    parser = argparse.ArgumentParser(
        description="Compare 3D segmentation feature tables for raw vs deconvolved stacks."
    )
    parser.add_argument(
        "--from-csv",
        nargs=2,
        metavar=("ORIG_CSV", "DEC_CSV"),
        help="Load existing nuclei feature CSVs and only write summary + figures (no segmentation).",
    )
    parser.add_argument(
        "--original",
        type=str,
        default=str(default_orig),
        help="Path to original multi-channel .tif",
    )
    parser.add_argument(
        "--deconvolved",
        type=str,
        default=str(default_dec),
        help="Path to deconvolved merged .tif (or any stack to compare)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(repo_root / "segmentation_comparison"),
        help="Directory for CSVs and figures",
    )
    parser.add_argument("--channel", type=int, default=1, help="Nuclear channel index (default 1)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        orig_csv = Path(args.from_csv[0])
        dec_csv = Path(args.from_csv[1])
        for p in (orig_csv, dec_csv):
            if not p.exists():
                raise FileNotFoundError(f"Missing CSV: {p}")
        df_orig = pd.read_csv(orig_csv)
        df_dec = pd.read_csv(dec_csv)
        print(f"Loaded features from CSV (no segmentation): {orig_csv.name}, {dec_csv.name}")
        compare_populations(df_orig, df_dec, out_dir)
        print("\nDone.")
        return

    orig_path = Path(args.original)
    dec_path = Path(args.deconvolved)

    for p in (orig_path, dec_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    df_orig = segment_stack(orig_path, args.channel, "original")
    df_dec = segment_stack(dec_path, args.channel, "deconvolved")

    orig_csv = out_dir / f"{orig_path.stem}_nuclei_features.csv"
    dec_csv = out_dir / f"{dec_path.stem}_nuclei_features.csv"
    df_orig.drop(columns=["source"]).to_csv(orig_csv, index=False)
    df_dec.drop(columns=["source"]).to_csv(dec_csv, index=False)
    print(f"\nSaved: {orig_csv}")
    print(f"Saved: {dec_csv}")

    compare_populations(df_orig, df_dec, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
