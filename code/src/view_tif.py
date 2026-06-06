import argparse
from pathlib import Path

import napari
import tifffile


DEFAULT_TIF = (
    Path(__file__).parent.parent
    / "deconvolution_results"
    / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005_deconvolved_merged.tif"
)

# Colormaps assigned per channel index
CHANNEL_COLORMAPS = ["green", "magenta", "cyan"]
CHANNEL_NAMES = ["lifeact (actin)", "nuclear", "ch3"]


def setup_viewer_layers(viewer: napari.Viewer, path: Path, segment: bool = False) -> None:
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None

    print(f"Loaded: {path.name}")
    print(f"Shape: {image.shape} | dtype: {image.dtype} | axes: {axes}")

    if image.ndim == 4:
        spatial = set(image.shape[-2:])
        candidates = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
        if axes and "C" in axes:
            channel_axis = axes.index("C")
        else:
            channel_axis = min(candidates, key=lambda x: x[1])[0]

        n_channels = image.shape[channel_axis]
        z_axis = 1 - channel_axis
        print(f"Channel axis: {channel_axis} ({n_channels} channels) | Z axis: {z_axis}")

        for c in range(n_channels):
            channel_data = image.take(c, axis=channel_axis)
            name = CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"
            cmap = CHANNEL_COLORMAPS[c] if c < len(CHANNEL_COLORMAPS) else "gray"
            viewer.add_image(
                channel_data,
                name=name,
                colormap=cmap,
                blending="additive",
            )

        if segment:
            print("Running segmentation...")
            from segment_nuclei_3d import segment_3d, extract_channel

            volume = extract_channel(image, axes, target_channel_index=1)
            labels = segment_3d(volume)
            viewer.add_labels(labels, name="Segmented Nuclei (Blobs)")

    elif image.ndim == 3 and image.shape[-1] not in (3, 4):
        viewer.add_image(image, name=path.stem, colormap="green", blending="additive")
        if segment:
            print("Running segmentation...")
            from segment_nuclei_3d import segment_3d

            labels = segment_3d(image)
            viewer.add_labels(labels, name="Segmented Nuclei (Blobs)")

    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        viewer.add_image(image, name=path.stem, rgb=True)


def load_and_view(path: Path, segment: bool = False) -> None:
    viewer = napari.Viewer(title=path.name)
    setup_viewer_layers(viewer, path, segment=segment)
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open a .tif file in napari")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_TIF),
        help="Path to a .tif file (defaults to a sample file)",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Run 3D nuclei segmentation on the image and add it as a labels layer",
    )
    args = parser.parse_args()

    tif_path = Path(args.path)
    if not tif_path.exists():
        raise FileNotFoundError(f"File not found: {tif_path}")

    load_and_view(tif_path, segment=args.segment)
