import argparse
import numpy as np
import pandas as pd
from skimage import filters, measure, morphology, feature, segmentation
from scipy import ndimage as ndi
import tifffile
from pathlib import Path
import time

def extract_channel(image, axes, target_channel_index=1):
    """
    Extracts a 3D volume (Z, Y, X) for the given channel.
    Defaults to channel 1 (nuclear).
    """
    if image.ndim == 4:
        if axes and "C" in axes:
            channel_axis = axes.index("C")
        else:
            # Default fallback for shape (Z, C, Y, X)
            spatial = set(image.shape[-2:])
            candidates = [(i, s) for i, s in enumerate(image.shape[:2]) if s not in spatial]
            channel_axis = min(candidates, key=lambda x: x[1])[0]
            
        print(f"Detected channel axis at index {channel_axis}.")
        # Extract the target channel
        volume = image.take(target_channel_index, axis=channel_axis)
        return volume
    elif image.ndim == 3:
        print("Image is 3D, assuming it's already a single channel (Z, Y, X).")
        return image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

def segment_3d(volume):
    print("1/5 Applying 3D Gaussian blur...")
    t0 = time.time()
    # Blur slightly more in XY than Z since Z resolution is often lower
    # We'll just use a uniform sigma for now, or slightly less in Z if desired.
    blurred = filters.gaussian(volume, sigma=(1, 2, 2))
    print(f"Done in {time.time()-t0:.2f}s")
    
    print("2/5 Thresholding (Otsu)...")
    t0 = time.time()
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    print(f"Done in {time.time()-t0:.2f}s")
    
    print("3/5 Morphological cleanup (removing small artifacts)...")
    t0 = time.time()
    # Remove very small components. In 3D, an object of 50 voxels is very small
    binary = morphology.remove_small_objects(binary, min_size=200)
    print(f"Done in {time.time()-t0:.2f}s")
    
    print("4/5 Distance transform and peak finding...")
    t0 = time.time()
    distance = ndi.distance_transform_edt(binary)
    # Find local maxima in 3D. min_distance limits how close nuclei centers can be.
    # We use min_distance=5 as a reasonable default for 3D nuclei
    local_max_coords = feature.peak_local_max(distance, min_distance=5, labels=binary)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    print(f"Done in {time.time()-t0:.2f}s")
    
    print("5/5 3D Watershed segmentation...")
    t0 = time.time()
    markers = measure.label(local_max_mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    print(f"Done in {time.time()-t0:.2f}s")
    
    return labels

def extract_features(labels, intensity_image):
    print("Extracting geometric features for each nucleus...")
    num_nuclei = labels.max()
    print(f"Total nuclei segmented: {num_nuclei}")
    
    # Base properties
    props = ['label', 'centroid', 'area', 'bbox', 'equivalent_diameter_area', 'extent', 'solidity']
    
    features = measure.regionprops_table(labels, intensity_image=intensity_image, properties=props)
    df = pd.DataFrame(features)
    
    # Rename for clarity
    df.rename(columns={'area': 'volume_voxels', 'equivalent_diameter_area': 'equivalent_diameter'}, inplace=True)
    
    # Compute bounding box dimensions
    # Bbox returns: min_z, min_y, min_x, max_z, max_y, max_x
    df['bbox_z_length'] = df['bbox-3'] - df['bbox-0']
    df['bbox_y_length'] = df['bbox-4'] - df['bbox-1']
    df['bbox_x_length'] = df['bbox-5'] - df['bbox-2']
    
    # Calculate principal axes lengths from inertia tensor (rough approximation of shape)
    # The inertia tensor gives variances. For a solid ellipsoid, var = L^2 / 10
    # Therefore, L = sqrt(10 * eigval) gives the semi-axis length
    print("Calculating principal axes...")
    region_props = measure.regionprops(labels)
    
    z_axes, y_axes, x_axes = [], [], []
    for prop in region_props:
        # Sort eigenvalues in descending order (largest = major axis, smallest = minor axis)
        # However, we want them assigned roughly to z, y, x based on the general spread if possible, 
        # but eigenvalues are rotation invariant. Let's just output them as major, intermediate, minor lengths.
        # Alternatively, we can project to x, y, z by just using bounding box or moments.
        # Eigvals are sorted smallest to largest or vice versa depending on numpy version
        # Usually: smallest eigenvalue -> shortest axis.
        eigvals = np.sort(prop.inertia_tensor_eigvals)[::-1] # descending
        # Lengths of a solid ellipsoid (full length of axis = 2 * semi-axis, but here we just give relative size)
        # We'll provide the 'diameter' length = 2 * sqrt(5 * eigval) which is approx length
        # Actually standard equation: full length = 2 * sqrt(5 * eigval)
        # Let's just output the scaled eigenvalues to represent shape
        lengths = np.sqrt(10 * eigvals) 
        z_axes.append(lengths[2] if len(lengths) > 2 else 0) # Minor
        y_axes.append(lengths[1] if len(lengths) > 1 else 0) # Intermediate
        x_axes.append(lengths[0] if len(lengths) > 0 else 0) # Major

    df['shape_major_axis'] = x_axes
    df['shape_intermediate_axis'] = y_axes
    df['shape_minor_axis'] = z_axes
    
    return df

def main(path, output_csv, target_channel=1):
    path = Path(path)
    print(f"Loading {path.name}...")
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes if tif.series else None
        
    print(f"Original image shape: {image.shape}, axes: {axes}")
    
    # Extract nuclear channel volume
    volume = extract_channel(image, axes, target_channel_index=target_channel)
    print(f"Extracted 3D volume shape: {volume.shape}")
    
    # Segment
    labels = segment_3d(volume)
    
    # Extract features
    df = extract_features(labels, volume)
    
    print(f"Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    DEFAULT_TIF = (
        Path(__file__).parent.parent
        / "data/raw/cmlc2_lifeactXnuclear/48hpf"
        / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
    )
    
    parser = argparse.ArgumentParser(description="Segment 3D nuclei from a .tif file and extract geometric features.")
    parser.add_argument("path", type=str, nargs="?", default=str(DEFAULT_TIF), help="Path to the .tif file")
    parser.add_argument("--output", type=str, default="nuclei_3d_features.csv", help="Output CSV file path")
    parser.add_argument("--channel", type=int, default=1, help="Channel index for nuclei (default 1)")
    args = parser.parse_args()
    
    main(args.path, args.output, args.channel)
