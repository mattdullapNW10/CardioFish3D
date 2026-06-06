import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell("""# PSF Estimation and Deconvolution Pipeline
This notebook demonstrates how to use the python java wrapper (`psfselect` package) to estimate PSFs using different techniques (like Born & Wolf, Gibson-Lanni, etc.) via the EPFL PSF Generator JAR. Afterwards, we apply Richardson-Lucy deconvolution using `skimage`.
"""))

nb.cells.append(new_code_cell("""import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.restoration import richardson_lucy

# 1. Setup paths
psf_pkg_dir = "/Users/matthieupallud/Downloads/FYP/code/psf"
if psf_pkg_dir not in sys.path:
    sys.path.append(psf_pkg_dir)

# Set the environment variable for the JAR file so the Python wrapper can find it
os.environ["PSF_GENERATOR_JAR"] = os.path.join(psf_pkg_dir, "psfgenerator.jar")

# Import from the custom python wrapper
from psfselect.parameters import PSFParams
from psfselect.backends import render_psf, MODELS
from psfselect import MODEL_LABELS"""))

nb.cells.append(new_markdown_cell("""## 1. Define Optical Parameters
We create a `PSFParams` object which contains all the required microscopy metadata (NA, wavelength, voxel sizes, etc.)."""))

nb.cells.append(new_code_cell("""# You can adjust these based on your specific microscope settings
params = PSFParams(
    na=1.0, 
    wavelength_nm=510.0,
    ni=1.33,
    ns=1.33,
    voxel_xy_um=0.1,
    voxel_z_um=0.3,
    nx=63,  # odd numbers are usually recommended to center the PSF perfectly
    nz=31
)
print("Parameters:", params)"""))

nb.cells.append(new_markdown_cell("""## 2. Generate PSFs with Different Models
We iterate through all available models in `MODELS` (which uses `psfgenerator.jar` behind the scenes via the python wrapper)."""))

nb.cells.append(new_code_cell("""psfs = {}

for model in MODELS:
    print(f"Generating PSF for {model}...")
    try:
        # We use backend="auto" which prefers "epfl" but falls back to "psfmodels" if needed
        vol, backend, cfg = render_psf(model, params, backend="auto")
        psfs[model] = vol
        print(f"  Success using backend: {backend}")
    except Exception as e:
        print(f"  Failed: {e}")

# Let us visualize the central Z-slice of the generated PSFs
fig, axes = plt.subplots(1, len(psfs), figsize=(15, 3))
if len(psfs) == 1:
    axes = [axes]
for ax, (model, vol) in zip(axes, psfs.items()):
    ax.imshow(vol[vol.shape[0]//2], cmap="magma")
    ax.set_title(MODEL_LABELS.get(model, model))
    ax.axis("off")
plt.tight_layout()
plt.show()"""))

nb.cells.append(new_markdown_cell("""## 3. Load Sample Image
We load one of the raw `.tif` stacks. Note that to speed up Richardson-Lucy deconvolution for this demonstration, we crop a small Region of Interest (ROI)."""))

nb.cells.append(new_code_cell("""image_path = "/Users/matthieupallud/Downloads/FYP/code/data/raw/cmlc2_lifeactXnuclear/48hpf/16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
image = tifffile.imread(image_path)
print(f"Original image shape: {image.shape}")

# Take a central crop to make the deconvolution run fast in the notebook
# Adjust these indices depending on the actual image size
z_mid = image.shape[0] // 2
image_crop = image[max(0, z_mid-15) : z_mid+16, 200:400, 200:400]
print(f"Cropped image shape: {image_crop.shape}")

plt.imshow(image_crop[image_crop.shape[0]//2], cmap="gray")
plt.title("Original Cropped Image (Central Z-slice)")
plt.axis("off")
plt.show()"""))

nb.cells.append(new_markdown_cell("""## 4. Richardson-Lucy Deconvolution
Now, we apply the `richardson_lucy` algorithm from `skimage.restoration` using each of our generated PSFs."""))

nb.cells.append(new_code_cell("""deconvolved_images = {}
num_iter = 15  # 15-20 iterations is a typical starting point

for model, psf in psfs.items():
    print(f"Deconvolving with {model} PSF...")
    # Perform Richardson-Lucy deconvolution
    # NOTE: skimage richardson_lucy expects floating point arrays in [0, 1] range usually
    img_norm = image_crop.astype(float) / image_crop.max()
    
    # Run algorithm
    deconv = richardson_lucy(img_norm, psf, num_iter=num_iter)
    deconvolved_images[model] = deconv

print("Deconvolution finished!")"""))

nb.cells.append(new_markdown_cell("""## 5. Results Comparison
Let's compare the central slice of the deconvolved volumes."""))

nb.cells.append(new_code_cell("""fig, axes = plt.subplots(1, len(deconvolved_images) + 1, figsize=(20, 5))

# Original
z_mid_crop = image_crop.shape[0] // 2
axes[0].imshow(image_crop[z_mid_crop], cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

# Deconvolved
for i, (model, deconv_img) in enumerate(deconvolved_images.items()):
    axes[i+1].imshow(deconv_img[z_mid_crop], cmap="gray")
    axes[i+1].set_title(f"Deconv: {model}")
    axes[i+1].axis("off")

plt.tight_layout()
plt.show()"""))

with open("/Users/matthieupallud/Downloads/FYP/code/PSF-estimation/psf_estimation.ipynb", "w") as f:
    nbformat.write(nb, f)
