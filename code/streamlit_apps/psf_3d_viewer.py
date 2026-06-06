"""Interactive exploration of computed PSFs (run psf_library_build.py first).

From project root:
  PYTHONPATH=src streamlit run streamlit_apps/psf_3d_viewer.py

Override output directory:
  PSF_LIBRARY_DIR=/path/to/psf_library streamlit run streamlit_apps/psf_3d_viewer.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def library_dir() -> Path:
    env = os.environ.get("PSF_LIBRARY_DIR")
    if env:
        return Path(env).resolve()
    return (PROJECT_ROOT / "outputs/psf_library").resolve()


def load_optics(ld: Path) -> dict | None:
    p = ld / "optics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_samples(per_sample: Path) -> list[str]:
    if not per_sample.is_dir():
        return []
    return sorted(d.name for d in per_sample.iterdir() if d.is_dir())


def list_average_slugs(averages_dir: Path) -> list[str]:
    if not averages_dir.is_dir():
        return []
    return sorted(
        p.name[: -len("_average.npy")]
        for p in averages_dir.glob("*_average.npy")
        if p.name.endswith("_average.npy")
    )


def coords_mesh(vol: np.ndarray, dz_um: float, dxy_um: float):
    nz, ny, nx = vol.shape
    zc = (np.arange(nz) - nz // 2).astype(np.float64) * dz_um
    yc = (np.arange(ny) - ny // 2).astype(np.float64) * dxy_um
    xc = (np.arange(nx) - nx // 2).astype(np.float64) * dxy_um
    zz, yy, xx = np.meshgrid(zc, yc, xc, indexing="ij")
    return xx, yy, zz


def fig_volume(vol: np.ndarray, dxy_um: float, dz_um: float, title: str) -> go.Figure:
    vol = np.asarray(vol, dtype=np.float64)
    vol = np.nan_to_num(vol)
    vmax = float(np.nanmax(vol))
    if vmax <= 1e-12:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f14",
            title=dict(text=f"{title}\n(empty volume)", font=dict(size=13)),
            height=400,
        )
        return fig

    vclip = np.clip(vol, 0.0, float(np.percentile(vol, 99.5)))

    xx, yy, zz = coords_mesh(vol, dz_um, dxy_um)

    bg = "#0f0f14"
    fig = go.Figure(
        data=[
            go.Volume(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=vclip.flatten(),
                opacity=0.12,
                surface_count=18,
                colorscale="Plasma",
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=bg,
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=0, r=0, t=42, b=10),
        height=520,
        scene=dict(
            xaxis=dict(title=dict(text="x (µm)"), backgroundcolor="#1a1520", gridcolor="#445"),
            yaxis=dict(title=dict(text="y (µm)"), backgroundcolor="#1a1520", gridcolor="#445"),
            zaxis=dict(title=dict(text="z (µm)"), backgroundcolor="#1a1520", gridcolor="#445"),
            bgcolor=bg,
            aspectmode="data",
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.55, y=1.55, z=1.25)),
        ),
    )
    return fig


def main() -> None:
    st.set_page_config(layout="wide", page_title="3D PSF explorer", initial_sidebar_state="expanded")
    st.title("Interactive 3D PSF explorer")

    ld = library_dir()
    optics = load_optics(ld)
    averages_dir = ld / "averages"
    per_sample = ld / "per_sample"

    st.sidebar.markdown("### Library folder")
    st.sidebar.code(str(ld), language="text")

    labels_map: dict[str, str] = {}
    if optics and isinstance(optics.get("method_labels"), dict):
        labels_map.update(optics["method_labels"])

    if not optics:
        st.error(
            "No `optics.json` — build the PSF library first:\n```\n"
            "PYTHONPATH=src python src/psf_library_build.py\n```"
        )
        st.stop()

    method_slugs = list_average_slugs(averages_dir)
    if not method_slugs:
        st.warning("No `*_average.npy` in averages/. Run the builder script.")
        st.stop()

    summary: dict[str, dict] = {}
    summ_path = averages_dir / "averages_summary.json"
    if summ_path.is_file():
        summary = json.loads(summ_path.read_text(encoding="utf-8"))

    tab_avg, tab_samp = st.tabs(["Average PSFs side by side", "Per-sample browser"])

    with tab_avg:
        st.subheader("Population‑mean PSFs (one volume per estimation method)")
        st.caption(
            "Analogous to comparative panels such as Richards & Wolf, Gibson–Lanni, Gaussian; "
            "plus an empirical puncta stack. Rotate with drag; zoom with scroll."
        )
        n = len(method_slugs)
        cols = st.columns(n)
        default_dxy, default_dz = 0.5675, 0.6849
        for i, slug in enumerate(method_slugs):
            with cols[i]:
                ave = np.load(averages_dir / f"{slug}_average.npy")
                md = summary.get(slug, {})
                dz_um = float(md.get("mean_dz_um", default_dz))
                dxy_um = float(md.get("mean_dxy_um", default_dxy))
                ttl = labels_map.get(slug, slug)
                subtitle = ""
                if "samples_averaged" in md:
                    subtitle = f" (n stacks = {md['samples_averaged']})"
                st.plotly_chart(
                    fig_volume(ave, dxy_um, dz_um, ttl + subtitle),
                    use_container_width=True,
                    key=f"avg-{slug}-{i}",
                )

    with tab_samp:
        st.subheader("Inspect one TIFF-derived sample")
        samples = list_samples(per_sample)
        if not samples:
            st.info("No `per_sample/*` folders. Build the library to populate.")
        else:
            pick = st.selectbox("Sample", samples, format_func=lambda s: s[:80])
            samp_dir = per_sample / pick
            npys = sorted(samp_dir.glob("*.npy"))
            slugs_here = sorted(p.stem for p in npys)
            if not slugs_here:
                st.info("No `.npy` files in this folder.")
            else:
                common_sorted = sorted(set(slugs_here) & set(method_slugs))
                default_sel = (
                    common_sorted[:4]
                    if common_sorted
                    else slugs_here[: min(4, len(slugs_here))]
                )
                chosen = st.multiselect("Methods", slugs_here, default=default_sel)
                cols2 = st.columns(max(1, len(chosen)))
                for ci, slug in enumerate(chosen):
                    with cols2[ci]:
                        meta_path = samp_dir / f"{slug}_meta.json"
                        dxy_um, dz_um = 0.5675, 0.6849
                        if meta_path.is_file():
                            mj = json.loads(meta_path.read_text(encoding="utf-8"))
                            dxy_um = float(mj.get("dxy_um", dxy_um))
                            dz_um = float(mj.get("dz_um", dz_um))
                        vol = np.load(samp_dir / f"{slug}.npy")
                        ttl = labels_map.get(slug, slug)
                        st.plotly_chart(
                            fig_volume(vol, dxy_um, dz_um, ttl),
                            use_container_width=True,
                            key=f"samp-{pick}-{slug}-{ci}",
                        )


if __name__ == "__main__":
    main()
