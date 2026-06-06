"""psfselect — PSF estimation and model selection for 3D fluorescence microscopy.

Focused toolkit to:
  * ingest and validate microscopy metadata,
  * generate candidate theoretical 3D PSFs (EPFL PSF Generator JAR, or a
    ``psfmodels`` fallback so the pipeline is runnable without Java),
  * compare candidate PSFs *without* a measured ground-truth PSF,
  * rank candidates and recommend a starting PSF model.

The scientific target is anisotropic, depth-degraded 3D fluorescence volumes of
zebrafish cardiac tissue. Deconvolution / GAN training are intentionally out of
scope; only a minimal reblur/deconv consistency utility lives in
:mod:`psfselect.reblur` for PSF evaluation.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The four physically-motivated PSF models this project knows about. These names
# are used throughout (config files, manifests, reports) and map onto both the
# EPFL PSF Generator short-names and the psfmodels fallback.
MODELS = ("born_wolf", "gibson_lanni", "richards_wolf", "vri_gibson_lanni")

MODEL_LABELS = {
    "born_wolf": "Born & Wolf (scalar, no stratified aberration)",
    "gibson_lanni": "Gibson & Lanni (scalar, stratified media)",
    "richards_wolf": "Richards & Wolf (vectorial, high-NA)",
    "vri_gibson_lanni": "Variable-RI Gibson & Lanni (depth-dependent aberration)",
}

# EPFL PSF Generator config short-names for each model.
EPFL_SHORTNAMES = {
    "born_wolf": "BW",
    "gibson_lanni": "GL",
    "richards_wolf": "RW",
    "vri_gibson_lanni": "VRIGL",
}
