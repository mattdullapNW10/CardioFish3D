# nuclei_morphometry

3D nuclei segmentation and **geometric morphometry** for fluorescence microscopy
stacks. Give it an image; it segments the nuclei, counts them, and measures key
geometric properties per nucleus — all in physical units (microns), using the
voxel size read from the TIFF so the anisotropic z axis is treated correctly.

## What it measures

Per nucleus (one row in the CSV):

| Column | Meaning |
|---|---|
| `volume_um3` | physical volume |
| `equivalent_diameter_um` | diameter of a sphere of the same volume |
| `centroid_{z,y,x}_um` | centre of mass |
| `extent_{x,y,z}_um` | axis-aligned bounding-box size along each axis (**XYZ elongation**) |
| `axis_{major,inter,minor}_um` | ellipsoid principal-axis lengths via PCA (orientation-invariant) |
| `elongation` | `axis_major / axis_minor` — 1 = round, >1 = elongated |
| `flatness` | `axis_inter / axis_minor` — >1 = flattened/disc-like |
| `sphericity_pca` | `axis_minor / axis_major` — 1 = isotropic |
| `major_axis_d{z,y,x}` | unit vector of the long axis (orientation) |
| `solidity`, `extent` | voxel-based convexity / bbox fill (skimage) |
| `mean_intensity` | mean signal inside the nucleus |

Plus a **count** and aggregate statistics (mean/median/std/min/max) in the
summary JSON.

## Pipeline

```
load -> pick nuclei channel -> Gaussian smooth -> Otsu/Li threshold ->
fill holes + remove small objects -> physical-units distance transform ->
peak-seeded watershed (splits touching nuclei) -> regionprops measurement
```

Smoothing, minimum size, and minimum centre separation are specified in
**microns** and converted to voxels internally, so results are resolution-aware.

## Usage

```bash
# from the FYP root
./venv/bin/python nuclei_morphometry/morphometry.py "<image.tif>" --channel 1

# override voxel size, use Li threshold, drop border-touching nuclei, view in napari
./venv/bin/python nuclei_morphometry/morphometry.py img.tif \
    --channel 1 --voxel 0.685,0.567,0.567 --threshold li --clear-border --show
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--channel N` | `1` | nuclei channel index (for multi-channel ZCYX stacks) |
| `--voxel dz,dy,dx` | from TIFF | voxel size in microns |
| `--threshold otsu\|li` | `otsu` | global threshold method |
| `--sigma-um` | `0.4` | Gaussian smoothing sigma (microns) |
| `--min-volume-um3` | `5.0` | discard objects smaller than this |
| `--min-distance-um` | `3.0` | minimum separation between nuclei centres |
| `--clear-border` | off | drop nuclei touching the XY border |
| `-o, --outdir` | `morphometry_outputs` | output folder |
| `--show` | off | open raw + labels in a 3D napari window |

> **Channel note:** point `--channel` at the *nuclear* marker. In the
> `cmlc2_..xnuclear` stacks the nuclear channel is typically index 1; the
> `cmlc2_lifeact_EGFP` stacks have no nuclear label (EGFP + transmission only),
> so segment a different marker or a different dataset there.

## Outputs (in `--outdir`)

```
<stem>_nuclei.csv          # one row per nucleus, all measurements
<stem>_summary.json        # count + aggregate stats + voxel size used
<stem>_labels.tif          # 16-bit label volume (open over the raw image)
<stem>_distributions.png   # histograms: volume, diameter, elongation, flatness, extents
```

## Dependencies

`numpy`, `scipy`, `pandas`, `scikit-image`, `tifffile`, `matplotlib` (all already
in the project venv); `napari` only for `--show`.

## Tuning tips

- **Over-merged nuclei** (count too low): lower `--min-distance-um`, or lower
  `--sigma-um`.
- **Over-split nuclei** (count too high): raise `--min-distance-um` / `--sigma-um`.
- **Debris counted as nuclei**: raise `--min-volume-um3`.
- This is a classical threshold+watershed segmenter — robust and dependency-light.
  For densely packed or low-contrast nuclei, a learned segmenter (e.g. Cellpose/
  StarDist) would do better; this tool is intentionally simple and transparent.
```
