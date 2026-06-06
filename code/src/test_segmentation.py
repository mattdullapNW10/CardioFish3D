import argparse
from pathlib import Path
import napari
import tifffile
import numpy as np

# Import the segmentation function
from segment_nuclei_3d import extract_channel, segment_3d

DEFAULT_TIF = (
    Path(__file__).parent.parent
    / "data/raw/cmlc2_lifeactXnuclear/48hpf"
    / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
)

def view_segmentation(path: Path):
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None

    print(f"Loaded: {path.name}")
    print(f"Shape: {image.shape} | dtype: {image.dtype} | axes: {axes}")

    # Segment the nuclei
    volume = extract_channel(image, axes, target_channel_index=1)
    print("Running segmentation...")
    labels = segment_3d(volume)

    viewer = napari.Viewer(title=f"Segmentation: {path.name}")

    # Add the raw channel
    viewer.add_image(
        volume,
        name="Nuclear Channel",
        colormap="magenta",
        blending="additive",
    )
    
    # Add the segmentation labels
    viewer.add_labels(
        labels,
        name="Segmented Nuclei",
        opacity=0.7
    )

    napari.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View 3D nuclei segmentation in napari")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_TIF),
        help="Path to a .tif file",
    )
    args = parser.parse_args()

    tif_path = Path(args.path)
    if not tif_path.exists():
        raise FileNotFoundError(f"File not found: {tif_path}")

    view_segmentation(tif_path)
