# psfselect — PSF estimation & model selection for 3D fluorescence microscopy

A focused toolkit to **estimate candidate 3D point-spread functions (PSFs) from
physics-based models and pick the best PSF model** for anisotropic, depth-degraded
3D fluorescence volumes — built around the use case of **zebrafish cardiac
tissue / myocardium**.

It is deliberately scoped to PSF work *before* deconvolution / GAN supervision.
There is **no** SRGAN/ESRGAN, no training loop, and no full deconvolution
pipeline here — only a minimal, isolated reblur/deconvolve utility used to
*evaluate* PSFs.

It uses **EPFL's PSF Generator** (https://bigwww.epfl.ch/algorithms/psfgenerator/)
as the authoritative PSF source (Born & Wolf, Gibson & Lanni, Richards & Wolf,
Variable-RI Gibson & Lanni) via its standalone Java application, and falls back
to the pure-Python [`psfmodels`](https://github.com/tlambert03/PSFmodels) package
so the whole pipeline runs even without Java.

---

## ⭐ One-command pipeline (start here)

Give it a filename. It **extracts the metadata, sets the parameters, computes a
PSF with each model, and deconvolves the fluorescence channel** — no manual
parameter entry:

```bash
cd psf
export PSF_GENERATOR_JAR="$PWD/psfgenerator.jar"     # optional; psfmodels fallback otherwise
../venv/bin/python -m psfselect.cli pipeline "<path/to/stack.tif>"
```

What it does, automatically:

1. **Extract metadata** from the file (Leica `.lif`/ImageJ aware): NA, immersion
   RI (`DRY`→1.0 / water→1.33 / oil→1.515, or the stored `RefractionIndex`),
   fluorophore **emission wavelength** (from the dye, not the pinhole reference),
   voxel sizes, dimensions, objective. Missing fields fall back to documented
   defaults (e.g. specimen RI `ns`→1.35) and are flagged.
2. **Build PSF parameters** from that metadata.
3. **Compute a PSF with each model** (Born & Wolf, Gibson & Lanni, Richards & Wolf
   via the EPFL JAR; Variable-RI GL via psfmodels) + score + a comparison report.
4. **Deconvolve** the fluorescence channel with the chosen model's PSF
   (`--deconv-model`, default Gibson & Lanni).

Verified on real data (`…EGFP…Series001`): auto-detected NA 0.75, **ni 1.0 (dry
objective)**, EGFP emission 509 nm, voxel 0.685/0.567 µm — then GL axial FWHM
2939 nm vs Born & Wolf 1912 nm (the air/tissue RI-mismatch aberration), and a
deconvolved volume saved.

Useful flags: `--channel N` (fluorescence channel, default 0), `--deconv-model`,
`--depth UM`, `--iters N`, `--crop-xy/--crop-z` (faster deconv on a sub-volume),
`--no-deconv` (just compare PSFs), `--show` (raw/deconvolved/PSF in napari),
`-o OUTDIR`.

Outputs under `pipeline_outputs/`: `metadata_resolved.json`, `report.md`,
`comparison_table.csv`, `psfs/` (one TIFF + EPFL config per model),
`deconvolved/` (raw + deconvolved + PSF TIFFs).

The lower-level commands below (`ingest`/`generate`/`compare`/`rank`/`report`/
`view`) remain available for finer control.

---

## What it does

1. **Metadata ingestion** — JSON / CSV / YAML, tolerant of missing fields (flags them).
2. **PSF parameter management** — per-sample optical parameters + parameter sweeps.
3. **PSF Generator wrapper** — writes valid config files, calls the JAR, loads outputs.
4. **Candidate PSF generation** — models × sweeps, with manifests and saved volumes.
5. **Ground-truth-free comparison** — plausibility, shape, anisotropy, stability,
   optional reblur consistency, explicit scoring.
6. **Ranking + recommendation** — a documented decision strategy tailored to
   zebrafish cardiac imaging.
7. **Visualisation / reporting** — ortho views, FWHM/score charts, sweep trends,
   a ranked Markdown report.
8. **CLI** — `ingest`, `generate`, `compare`, `rank`, `report`, `run`, `backends`.

---

## Install

```bash
cd psf
pip install -r requirements.txt        # or: pip install -e .
```

Python ≥ 3.9. Core scientific stack (numpy/scipy/pandas/tifffile/matplotlib),
`pyyaml`, `tabulate`, and `psfmodels`.

### Enabling the EPFL PSF Generator (optional but recommended)

The PSF Generator is a **Java application**, not a pip package. The EPFL page
offers two different JARs — use the **standalone** one:

| File | Use | This project |
|---|---|---|
| `PSFGenerator.jar` (2.5 Mb, "ImageJ bundled") | standalone / Matlab | ✅ **this is the one** |
| `PSF_Generator.jar` (508 Kb) | ImageJ/Fiji/Icy *plugin* | ❌ not used here |

1. Download **`PSFGenerator.jar`** (the 2.5 Mb standalone) from
   https://bigwww.epfl.ch/algorithms/psfgenerator/ — do **not** unzip it.
2. Have a Java runtime on your `PATH` (`java -version`; Java 8+ works, the JAR
   targets JRE 1.6+).
3. Point the tool at the JAR and confirm:

```bash
export PSF_GENERATOR_JAR=/path/to/PSFGenerator.jar
psfselect backends            # should now show: epfl AVAILABLE
```

Under the hood the wrapper runs the documented batch command in a temp dir:

```bash
java -Djava.awt.headless=true -cp PSFGenerator.jar PSFGenerator config.txt
```

and reads back the TIFF the JAR writes. The generated `config.txt` uses the
official key format (`PSF-shortname`, `ResLateral/ResAxial` in nm, `Lambda` in
nm, `psf-GL-NI/NS/TI/ZPos`, `psf-VRIGL-NS1/NS2/RIvary`, `…-accuracy`, etc.) — the
exact same file you would save from the GUI, so you can sanity-check it there.

If the JAR is absent, the `psfmodels` backend is used automatically (the
`backend` column in the outputs records which backend produced each PSF). Force
a backend explicitly with `--backend epfl`.

> **Note on Variable-RI Gibson & Lanni (VRIGL):** the EPFL standalone's VRIGL
> model frequently does not terminate (it prints `Computing VRIGL` and then runs
> effectively forever, regardless of voxel size / parameters). Verified here:
> Born & Wolf, Gibson & Lanni and Richards & Wolf render in seconds via the JAR,
> but VRIGL hangs. So in `--backend auto` the JAR is **skipped for VRIGL only**
> and the instant `psfmodels` approximation is used for it instead — the other
> three models still use the real EPFL JAR. (`--backend epfl` will attempt VRIGL
> on the JAR and time out after `run_psfgenerator(timeout_s=180)`.)

---

## Quick start (minimal end-to-end example)

No real data or JAR needed — uses bundled example metadata and the psfmodels
fallback:

```bash
bash examples/run_example.sh            # writes ./example_outputs/
```

or step by step:

```bash
# 1. Ingest + validate metadata (flags missing fields, applies documented defaults)
psfselect ingest examples/example_metadata.yaml -o out

# 2. Generate candidate PSFs for all four models
psfselect generate examples/example_metadata.yaml -o out --nx 96 --nz 48

# 3. Score candidates without ground truth
psfselect compare out/candidates_manifest.json -o out

# 4. Rank + recommend
psfselect rank out/candidates_scored.json -o out

# 5. Full report (tables + figures + ranked Markdown)
psfselect report out/candidates_scored.json -o out
```

Or all five at once:

```bash
psfselect run examples/example_metadata.yaml -o out
```

### Visualise: raw + deconvolved + PSF in three napari windows

`view` renders a PSF for a chosen model at the image's voxel sampling, runs a few
Richardson-Lucy iterations, and opens **three independent napari windows** (raw /
deconvolved / PSF-3D), each scaled by the physical voxel size. It is
**multi-channel aware**: by default *every* channel is deconvolved and shown as a
separate, additively-blended layer (green / magenta / cyan …), so structures like
the nuclear channel stay visible alongside lifeact.

```bash
# ALL channels by default; voxel size auto-read from the TIFF
psfselect view "data/raw/.../my_stack.tif" --model gibson_lanni

# give each channel its own emission wavelength so its PSF is correct
psfselect view my_stack.tif --wavelengths 510,580,670

# just one channel, or a subset
psfselect view my_stack.tif --channel 1
psfselect view my_stack.tif --channels 0 2

# deeper stacks: crop for speed, more RL iterations, pick the model
psfselect view my_stack.tif --model vri_gibson_lanni \
    --crop-xy 256 --crop-z 48 --iters 15 --depth 40

# pull optical params from a metadata file, or set them inline
psfselect view my_stack.tif --metadata examples/example_metadata.yaml \
    --sample-id zf_cardiac_48hpf_s005
psfselect view my_stack.tif --na 0.95 --wavelength-nm 510 --ni 1.33 --ns 1.35 \
    --voxel 0.685,0.568,0.568

# save the three volumes to disk too (and optionally skip the GUI)
psfselect view my_stack.tif --save view_out [--no-show]
```

Requires `napari` (`pip install "napari[all]"`). The three windows stay open
until you close them. With the EPFL JAR enabled, add `--backend epfl`.

### Parameter sweep

```bash
psfselect generate examples/example_metadata.yaml -o out_sweep \
    --sweep configs/sweep.example.yaml --nx 80 --nz 40
psfselect report out_sweep/candidates_manifest.json -o out_sweep
```

### With your own data + the EPFL JAR + reblur test

```bash
export PSF_GENERATOR_JAR=/path/to/PSFGenerator.jar
psfselect run my_metadata.csv -o out --backend epfl --with-reblur
```
`--with-reblur` runs the deconvolve→reblur consistency test on each sample's
`image_path` (TIFF/OME-TIFF/NumPy).

---

## Metadata schema

One record per imaged volume. Only `sample_id` is required; everything else is
optional and defaulted-with-provenance if missing. See
[`examples/example_metadata.yaml`](examples/example_metadata.yaml) and
[`examples/example_metadata.csv`](examples/example_metadata.csv).

| field | meaning |
|---|---|
| `sample_id` | unique name (**required**) |
| `modality` | confocal / widefield / two-photon / lightsheet |
| `wavelength_nm` | emission wavelength |
| `na` | numerical aperture |
| `ni`, `ni0` | immersion medium RI (in-use / design) |
| `ns` | specimen RI (zebrafish myocardium ≈ 1.35) |
| `ng`, `ng0` | coverslip RI |
| `coverslip_um`, `coverslip_design_um` | coverslip thickness |
| `working_distance_um` | objective working distance |
| `voxel_x_um`, `voxel_y_um`, `voxel_z_um` | voxel size (auto-read from TIFF if `image_path` set) |
| `image_x/y/z` | image dimensions (voxels) |
| `imaging_depth_um` | acquisition depth / z-position into specimen |
| `objective` | free-text objective |
| `image_path` | optional raw volume (for the reblur test) |
| `notes` | free text |

Common aliases are accepted (`NA`, `emission_wavelength`, `dz`, `depth`, …).
Missing fields are listed in `metadata_resolved.json` under `missing_fields`,
and the defaults that were applied under `applied_defaults`.

---

## How candidates are compared *without* a ground-truth PSF

Since there is no measured bead PSF, each candidate is scored by **indirect
criteria**, each normalised to `[0, 1]` (higher = better):

- **Plausibility** — does the rendered PSF agree with the diffraction limit
  implied by the metadata? Lateral FWHM vs `0.51·λ/NA`, axial FWHM vs
  `1.77·n·λ/NA²`; plus Nyquist sampling adequacy, containment in the grid,
  centering, and lateral symmetry.
- **Stability** — how little the shape metrics move under small (±5%)
  perturbations of NA / RI / depth. Robust models score higher.
- **Reblur consistency** *(optional, needs raw image)* — a few Richardson-Lucy
  iterations followed by reblur should reproduce the raw volume: low residual,
  high correlation, positive sharpness gain.

These combine into a weighted **composite** (default weights
`plausibility=0.5, stability=0.25, reblur=0.25`; override with
`compare --weights`). Sub-scores that can't be computed (e.g. no raw data) are
dropped and the weights renormalised.

**Depth sensitivity** (relative axial-FWHM change under a depth perturbation) is
reported separately — it is a *diagnostic* that drives the recommendation, not a
quality axis.

Shape summaries reported per candidate: lateral/axial FWHM, anisotropy
(axial/lateral), energy concentration, axial side-lobe ratio, lateral symmetry,
peak offset, containment.

---

## Recommendation strategy (default)

Encoded in [`psfselect/ranking.py`](psfselect/ranking.py), tuned for zebrafish 3D
cardiac tissue (axial blur matters; RI mismatch and depth may matter; but a
simple robust baseline comes first):

1. **Start with Gibson & Lanni** — robust scalar baseline with stratified-media
   aberration.
2. **Compare against Born & Wolf** — if axial FWHM is nearly identical,
   stratified-media aberration is mild and Born & Wolf is an acceptable simpler
   baseline; otherwise prefer GL.
3. **Test Richards & Wolf** when vectorial effects are likely relevant —
   `NA ≥ 1.0`, or a clear axial side-lobe difference vs GL.
4. **Escalate to Variable-RI Gibson & Lanni** when depth/RI indicators trigger:
   `|ni − ns| ≥ 0.04`, imaging depth `≥ 30 µm`, or depth-sensitivity of axial
   FWHM `≥ 5%`.

Each recommendation states the numbers behind every decision and a confidence
level (driven by the score margin and whether data-driven reblur was available).
Thresholds are module-level constants you can tune.

---

## Outputs

Written under the chosen output directory:

```
out/
  metadata_resolved.json        # samples with missing_fields + applied_defaults
  candidates_manifest.json      # all candidates: params, metrics, paths
  candidates_scored.json        # + sub-scores and composite
  comparison_table.csv          # tidy ranked table
  ranking.csv                   # (from `rank`)
  recommendations.json          # per-sample recommendation + reasons + flags
  report.md                     # ranked Markdown report
  psfs/
    <candidate_id>.tif          # saved PSF volume (ImageJ TIFF, ZYX)
    <candidate_id>.config.txt   # EPFL config (when the JAR backend was used)
  figures/
    compare_<sample>.png        # FWHM / anisotropy / composite per model
    scores_<sample>.png         # plausibility / stability / reblur per model
    sweep_<sample>_<axis>.png   # trends vs swept parameter
    orthoviews/<id>_ortho.png   # XY / XZ / YZ views per PSF
```

---

## CLI reference

```
psfselect backends                       # which backends are available
psfselect ingest   META -o OUT
psfselect generate META -o OUT [--models ...] [--sweep S.yaml]
                                [--backend auto|epfl|psfmodels|gaussian]
                                [--jar PSFGenerator.jar] [--nx N] [--nz N]
                                [--no-stability] [--with-reblur]
psfselect compare  MANIFEST -o OUT [--weights plausibility=..,stability=..,reblur=..]
psfselect rank     MANIFEST -o OUT
psfselect report   MANIFEST -o OUT [--no-orthoviews]
psfselect run      META -o OUT [same options as generate]
```

`META` is JSON / CSV / YAML; `MANIFEST` is a candidates manifest (scored or not —
unscored manifests are scored on the fly).

---

## Assumptions

- **No ground-truth PSF.** All rankings are *relative*, from physical plausibility,
  stability, and optional data consistency — not from a measured PSF.
- **Defaults** for missing optical fields target a generic water-immersion confocal
  objective and zebrafish tissue (`ns ≈ 1.35`); every applied default is recorded
  per sample. Edit `psfselect/metadata.py:DEFAULTS` or document yours in
  `configs/optics.example.yaml`.
- **Fallback fidelity.** Without the EPFL JAR, `psfmodels` provides the PSFs:
  Born & Wolf ≈ matched-RI scalar (no stratified aberration), Gibson & Lanni =
  scalar stratified, Richards & Wolf = vectorial, Variable-RI GL ≈ scalar with a
  depth-adjusted specimen RI. For publication-grade Variable-RI / Born & Wolf
  specifically, use the EPFL JAR (`--backend epfl`).
- **EPFL config keys.** The config writer emits the common + per-model key blocks;
  if your PSF Generator version expects different keys, adjust
  `psfselect/psfgenerator.py:build_config` (you can dump a config from the GUI to
  compare).

---

## Tests

```bash
cd psf
pytest -q
```

The pipeline smoke test uses the always-available gaussian backend, so it runs
fast and offline.

---

## Project layout

```
psf/
  psfselect/
    __init__.py        # model registry + labels
    io_utils.py        # TIFF/OME-TIFF/NumPy/JSON IO, logging
    metadata.py        # schema, ingestion, validation, defaults
    parameters.py      # PSFParams, sweeps, perturbations
    psfgenerator.py    # EPFL PSF Generator config writer + JAR subprocess
    backends.py        # unified render_psf: epfl / psfmodels / gaussian
    candidates.py      # candidate generation + manifests
    metrics.py         # FWHM, anisotropy, energy, side-lobes, symmetry
    reblur.py          # minimal RL deconvolve + reblur consistency (eval only)
    compare.py         # ground-truth-free scoring framework
    ranking.py         # ranking + recommendation decision strategy
    viz.py             # ortho views, comparison/sweep figures
    report.py          # comparison tables + ranked Markdown report
    cli.py             # command-line interface
  configs/             # example optics + sweep specs
  examples/            # example metadata + run_example.sh
  tests/               # pytest suite
  pyproject.toml
  requirements.txt
  README.md
```
