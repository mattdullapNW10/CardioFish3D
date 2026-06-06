import tifffile
from pathlib import Path

DEFAULT_TIF = (
    Path(__file__).parent.parent
    / "data/raw/cmlc2_lifeactXnuclear/48hpf"
    / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
)

with tifffile.TiffFile(DEFAULT_TIF) as tif:
    tags = tif.pages[0].tags
    
    if "XResolution" in tags:
        xr = tags["XResolution"].value
        if isinstance(xr, tuple) and len(xr) == 2:
            x_res = xr[1] / xr[0] if xr[0] != 0 else 1.0
            print(f"dx (from XResolution): {x_res:.4f} microns")
    
    ij_meta = tif.imagej_metadata
    if ij_meta and "spacing" in ij_meta:
        print(f"dz (from ImageJ spacing): {ij_meta['spacing']} microns")
