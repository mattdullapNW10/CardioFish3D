import argparse
import numpy as np
from pathlib import Path
import tifffile
import psfmodels as psfm
from skimage.restoration import richardson_lucy
import time


def detect_channel_axis(image, axes):
    """Return index of C axis for 4D ZCYX-like stacks."""
    if image.ndim != 4:
        return None
    if axes and "C" in axes:
        return axes.index("C")
    spatial = set(image.shape[-2:])
    candidates = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
    return min(candidates, key=lambda x: x[1])[0]


def extract_channel(image, axes, target_channel_index=1):
    """
    Extracts a 3D volume (Z, Y, X) for the given channel.
    Defaults to channel 1 (nuclear).
    """
    if image.ndim == 4:
        channel_axis = detect_channel_axis(image, axes)
        volume = image.take(target_channel_index, axis=channel_axis)
        return volume
    elif image.ndim == 3:
        return image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

def get_metadata(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        tags = tif.pages[0].tags
        dx = 0.5675  # fallback
        if "XResolution" in tags:
            xr = tags["XResolution"].value
            if isinstance(xr, tuple) and len(xr) == 2 and xr[0] != 0:
                dx = xr[1] / xr[0]
        
        ij_meta = tif.imagej_metadata or {}
        dz = ij_meta.get("spacing", 0.6849)
        
        # We know from earlier metadata extraction:
        na = 0.75  # NumericalAperture
        mag = 20.0  # Magnification
        # Assumption for dry objective
        ni = 1.0  # Refractive index for air (dry objective)
        ns = 1.33 # Refractive index of sample (water/tissue)
        
    return dx, dz, na, ni, ns


def insert_channel_volume(image, axes, channel_index, volume_deconvolved):
    """Return a copy of `image` with `volume_deconvolved` written into `channel_index`."""
    merged = np.array(image, copy=True, dtype=image.dtype)
    if merged.ndim != 4:
        return volume_deconvolved
    channel_axis = detect_channel_axis(image, axes)
    idx = [slice(None)] * merged.ndim
    idx[channel_axis] = channel_index
    merged[tuple(idx)] = np.asarray(volume_deconvolved, dtype=merged.dtype)
    return merged


def write_merged_hyperstack(path, data_zcyx, dxy_um, dz_um, ij_meta=None):
    """Save ZCYX uint8 hyperstack with ImageJ-compatible tags."""
    z, c, y, x = data_zcyx.shape
    meta = {
        "axes": "ZCYX",
        "unit": "micron",
        "spacing": float(dz_um),
        "mode": "grayscale",
    }
    if ij_meta:
        loop = ij_meta.get("loop")
        if loop is not None:
            meta["loop"] = loop
    photometric = "minisblack"
    tifffile.imwrite(
        path,
        data_zcyx,
        imagej=True,
        photometric=photometric,
        resolution=(1.0 / dxy_um, 1.0 / dxy_um),
        metadata=meta,
    )


def deconvolve_image(tif_path, output_dir, channel=1, iterations=5):
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {tif_path.name}...")
    with tifffile.TiffFile(tif_path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None
        ij_meta = tif.imagej_metadata

    # Extract single channel (Z, Y, X)
    volume = extract_channel(image, axes, target_channel_index=channel)
    print(f"Volume shape for deconvolution: {volume.shape}")
    
    # Get metadata
    dxy, dz, na, ni, ns = get_metadata(tif_path)
    print(f"Metadata: dxy={dxy:.4f} um, dz={dz:.4f} um, NA={na}, ni={ni}")
    
    # The original script used a PSF shape. Let's make it 31x63x63 for speed/accuracy.
    # psfmodels nx and nz must be odd.
    nx = 63  # XY size
    nz = 31  # Z size
    
    # We'll use emission wavelength ~0.6 um (600nm) since the nuclear stain is often red
    wvl = 0.6 
    
    print("Generating PSF with psfmodels...")
    t0 = time.time()
    psf = psfm.make_psf(nz, nx, dxy=dxy, dz=dz, NA=na, wvl=wvl,
                        model='gaussian', ni=ni, ni0=ni, ns=ns,
                        oversample_factor=3, normalize=True)
    print(f"PSF generated in {time.time()-t0:.2f}s. Shape: {psf.shape}")
    
    # Save PSF
    psf_out = output_dir / f"{tif_path.stem}_psf.tif"
    tifffile.imwrite(psf_out, psf.astype(np.float32))
    print(f"Saved PSF to {psf_out}")
    
    # Run Richardson-Lucy Deconvolution
    # For performance, we'll convert volume to float in range [0, 1]
    print(f"Running Richardson-Lucy deconvolution ({iterations} iterations)...")
    volume_norm = volume.astype(np.float32) / 255.0
    
    t0 = time.time()
    deconvolved = richardson_lucy(volume_norm, psf, num_iter=iterations)
    print(f"Deconvolution finished in {time.time()-t0:.2f}s")
    
    # Scale back to 8-bit
    deconvolved_8bit = np.clip(deconvolved * 255, 0, 255).astype(np.uint8)

    out_file = output_dir / f"{tif_path.stem}_deconvolved.tif"
    tifffile.imwrite(out_file, deconvolved_8bit)
    print(f"Saved deconvolved single-channel volume to {out_file}")

    if image.ndim == 4:
        merged = insert_channel_volume(image, axes, channel, deconvolved_8bit)
        out_stack = output_dir / f"{tif_path.stem}_deconvolved_merged.tif"
        write_merged_hyperstack(out_stack, merged, dxy, dz, ij_meta=ij_meta)
        print(f"Saved full hyperstack (channel {channel} replaced) to {out_stack}")

if __name__ == "__main__":
    DEFAULT_TIF = (
        Path(__file__).parent.parent
        / "data/raw/cmlc2_lifeactXnuclear/48hpf"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
    )
    
    parser = argparse.ArgumentParser(description="Deconvolve a 3D TIF image using estimated PSF.")
    parser.add_argument("path", type=str, nargs="?", default=str(DEFAULT_TIF), help="Path to .tif file")
    parser.add_argument("--outdir", type=str, default="deconvolution_results", help="Output directory")
    parser.add_argument("--iter", type=int, default=5, help="Number of RL iterations")
    parser.add_argument("--channel", type=int, default=1, help="Channel to deconvolve (default 1 = nuclear)")
    args = parser.parse_args()
    
    deconvolve_image(args.path, args.outdir, args.channel, args.iter)
