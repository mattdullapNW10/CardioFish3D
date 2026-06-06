# Source Code

This directory contains all source code for the project.

## Organization

- `preprocessing/` — Image preprocessing, normalization, augmentation
- `models/` — Model architectures and definitions
- `train.py` — Training pipeline
- `evaluate.py` — Model evaluation
- `utils.py` — Utility functions

## Microscopy utilities (this repo)

| Script | Purpose |
|--------|---------|
| `microscopy_io.py` | Read TIFF voxel sizes (`dxy`, `dz`) and extract one channel (`ZYX`) |
| `psf_vs_depth.py` | Puncta-based effective PSF breadth vs axial slice |
| `psf_library_build.py` | Multi-method 3D PSFs over all TIFFs under `data/raw` |
| `../streamlit_apps/psf_3d_viewer.py` | Interactive Plotly 3D viewer for library outputs |

Theoretical panels follow the taxonomy of the EPFL [PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/) and are implemented in Python via **psfmodels** (Gaussian, Gibson–Lanni scalar, Richards–Wolf-type vectorial). Born & Wolf as in the Java package is **not** reimplemented here.

```bash
PYTHONPATH=src python src/psf_library_build.py

# Prefer this launcher — uses `FYP/venv` (Homebrew Python 3.13 often has no plotly / blocks pip)
./streamlit_apps/run_psf_viewer.sh
```
