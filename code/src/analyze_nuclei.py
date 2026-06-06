import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
import argparse
from pathlib import Path

def analyze_nuclei(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate additional features
    df['aspect_ratio'] = df['shape_major_axis'] / df['shape_minor_axis'].replace(0, np.nan)
    df['z_centroid'] = df['centroid-0']
    
    # Calculate nearest neighbor distance
    coords = df[['centroid-0', 'centroid-1', 'centroid-2']].values
    dist_mat = distance_matrix(coords, coords)
    # Fill diagonal with infinity so we don't pick the cell itself
    np.fill_diagonal(dist_mat, np.inf)
    df['nearest_neighbor_dist'] = dist_mat.min(axis=1)

    print(f"Loaded {len(df)} nuclei.")
    print("Generating plots...")

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create a single figure with multiple subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Nuclei Morphological Analysis', fontsize=16, y=0.98)

    # 1. Volume Distribution
    sns.histplot(df['volume_voxels'], bins=30, kde=True, color='skyblue', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Nuclei Volumes')
    axes[0, 0].set_xlabel('Volume (voxels)')
    axes[0, 0].set_ylabel('Count')

    # 2. Aspect Ratio Distribution
    sns.histplot(df['aspect_ratio'].dropna(), bins=30, kde=True, color='salmon', ax=axes[0, 1])
    axes[0, 1].set_title('Aspect Ratio (Major / Minor Axis)')
    axes[0, 1].set_xlabel('Aspect Ratio (1 = Spherical, >1 = Elongated)')
    
    # 3. Solidity Distribution
    sns.histplot(df['solidity'], bins=30, kde=True, color='lightgreen', ax=axes[1, 0])
    axes[1, 0].set_title('Solidity Distribution')
    axes[1, 0].set_xlabel('Solidity (1 = Perfectly Convex)')

    # 4. Z-Depth Distribution
    sns.histplot(df['z_centroid'], bins=30, kde=True, color='purple', ax=axes[1, 1])
    axes[1, 1].set_title('Nuclei Distribution along Z-axis (Depth)')
    axes[1, 1].set_xlabel('Z-slice index')
    axes[1, 1].set_ylabel('Number of Nuclei')

    # 5. Nearest Neighbor Distance
    sns.histplot(df['nearest_neighbor_dist'], bins=30, kde=True, color='orange', ax=axes[2, 0])
    axes[2, 0].set_title('Distance to Nearest Neighbor Nucleus')
    axes[2, 0].set_xlabel('Distance (voxels)')
    axes[2, 0].set_ylabel('Count')

    # 6. Scatter Plot: Volume vs. Solidity
    sns.scatterplot(data=df, x='volume_voxels', y='solidity', alpha=0.6, color='darkblue', ax=axes[2, 1])
    axes[2, 1].set_title('Volume vs. Solidity')
    axes[2, 1].set_xlabel('Volume (voxels)')
    axes[2, 1].set_ylabel('Solidity')

    plt.tight_layout()
    plt.savefig(out_path / 'combined_analysis_subplots.png')
    
    # Also show the plot interactively
    plt.show()
    plt.close()

    # 7. Pairplot of key morphological features (this one is complex, better to keep separate if we still want it)
    key_features = ['volume_voxels', 'solidity', 'aspect_ratio', 'nearest_neighbor_dist']
    g = sns.pairplot(df[key_features].dropna(), corner=True, diag_kind='kde')
    g.fig.suptitle('Pairwise Relationships of Key Morphological Features', y=1.02)
    plt.savefig(out_path / 'morphology_pairplot.png')
    plt.close()

    print(f"Done! Plots saved in: {out_path.absolute()}")
    
    # Print some summary statistics
    print("\n--- Summary Statistics ---")
    print(df[['volume_voxels', 'solidity', 'aspect_ratio', 'nearest_neighbor_dist']].describe().T)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze geometric features of segmented nuclei.")
    parser.add_argument("--csv", type=str, default="nuclei_3d_features.csv", help="Input CSV file from segmentation")
    parser.add_argument("--outdir", type=str, default="analysis_plots", help="Directory to save plots")
    args = parser.parse_args()
    
    analyze_nuclei(args.csv, args.outdir)
