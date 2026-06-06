#!/usr/bin/env python3
"""
Iterates over all raw data, extracts metadata, generates the PSF, 
and deconvolves the specified channel (default: 1 for nuclei) using Richardson-Lucy.
The deconvolved volumes are saved to a processed directory to act as targets
for GAN training (PairedDeconvDataset).

Usage:
    python create_paired_dataset.py
    python create_paired_dataset.py --channel 1 --iters 15
"""

import argparse
import sys
from pathlib import Path

# Setup paths to import from src/ and psf/ directories
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "psf"))

import numpy as np
from psfselect.metadata_leica import extract_metadata
from psfselect.metadata import validate_samples
from psfselect.parameters import params_from_metadata
from psfselect.visualize_napari import deconvolve_multichannel
from psfselect.io_utils import save_volume
from microscopy_io import iter_raw_tifs, load_hyperstack_volume

def main():
    parser = argparse.ArgumentParser(description="Create paired dataset for GAN training.")
    parser.add_argument("--raw_root", type=str, default="data/raw", help="Path to raw data")
    parser.add_argument("--out_dir", type=str, default="data/processed/deconvolved", help="Where to save deconvolved files")
    parser.add_argument("--channel", type=int, default=1, help="Channel to deconvolve (0=actin, 1=nuclei typically)")
    parser.add_argument("--model", type=str, default="gibson_lanni", help="PSF model to use")
    parser.add_argument("--iters", type=int, default=10, help="Richardson-Lucy iterations")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()

    raw_root = project_root / args.raw_root
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = iter_raw_tifs(raw_root)
    print(f"Found {len(tifs)} raw .tif files in {raw_root}")

    success_count = 0

    for raw_path in tifs:
        target_path = out_dir / f"{raw_path.stem}_deconvolved.tif"
        if target_path.exists() and not args.force:
            print(f"Skipping {raw_path.name}, already exists at {target_path.relative_to(project_root)}")
            continue

        print(f"\nProcessing {raw_path.name}...")
        
        try:
            # 1. Load the specific channel (handles CZYX / ZYX finding and extracts 1 channel)
            try:
                vol_3d, axes, (dxy, dz) = load_hyperstack_volume(raw_path, channel_index=args.channel)
            except ValueError as e:
                print(f"  [Skip] Could not load channel {args.channel}: {e}")
                continue

            # 2. Extract metadata
            try:
                meta, info = extract_metadata(raw_path)
                # Ensure no critical failures in mapping
                sample = validate_samples([meta])[0]
                base = params_from_metadata(sample, nx=63, nz=63)
            except Exception as e:
                print(f"  [Warn] Metadata extraction failed: {e}. Using fallback defaults.")
                # Fallback base params if metadata fails completely
                from psfselect.parameters import PSFParams
                base = PSFParams(
                    na=0.8, wavelength_nm=600.0, ni=1.33, ni0=1.33, ns=1.35,
                    nx=63, nz=63, voxel_xy_um=dxy, voxel_z_um=dz, sample_id=raw_path.stem
                )

            # 3. Deconvolve
            # deconvolve_multichannel expects (C, Z, Y, X)
            raw_czyx = vol_3d[None, ...] # shape (1, Z, Y, X)
            
            print(f"  Volume shape: {vol_3d.shape}, voxel(dz,dy,dx) = ({dz:.4f}, {dxy:.4f}, {dxy:.4f})")
            print(f"  Generating PSF ({args.model}) and running {args.iters} iterations of R-L...")
            
            dec_czyx, psf_czyx = deconvolve_multichannel(
                raw_czyx, 
                model=args.model, 
                params=base, 
                backend="psfmodels", 
                iters=args.iters
            )
            
            # dec_czyx is (1, Z, Y, X). We want to save a 3D ZYX TIFF.
            dec_vol = dec_czyx[0]
            
            # 4. Save the single channel deconvolved volume
            save_volume(target_path, dec_vol, voxel_um=(dz, dxy, dxy))
            print(f"  [Success] Saved -> {target_path.relative_to(project_root)}")
            success_count += 1

        except Exception as e:
            print(f"  [Error] Failed on {raw_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Processed and saved {success_count} deconvolved files to {out_dir.relative_to(project_root)}")

if __name__ == "__main__":
    main()
