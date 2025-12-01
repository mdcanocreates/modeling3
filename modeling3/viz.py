"""
Visualization utilities for Modeling 3.

Generates publication-ready figures (300 DPI, colorblind-safe):
- Image grid (original + generated)
- PCA/UMAP clustering scatter plot
- Hierarchical clustering dendrogram
- Feature correlation matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import seaborn as sns
from skimage.io import imread
import modeling3.config as config


# Colorblind-safe palette (ColorBrewer Set2)
COLORBLIND_PALETTE = [
    '#66c2a5',  # Teal
    '#fc8d62',  # Orange
    '#8da0cb',  # Purple
    '#e78ac3',  # Pink
    '#a6d854',  # Green
    '#ffd92f',  # Yellow
    '#e5c494',  # Tan
    '#b3b3b3'   # Gray
]


def plot_image_grid(
    image_records: List,
    output_path: Path,
    n_cols: int = 6,
    dpi: int = None
) -> None:
    """
    Create grid of original + generated images.
    
    Parameters
    ----------
    image_records : list
        List of ImageRecord objects (or dicts with path_actin, etc.)
    output_path : Path
        Path to save figure
    n_cols : int
        Number of columns in grid
    dpi : int, optional
        DPI for figure. Defaults to config.FIGURE_DPI.
    """
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    # Filter to show originals + sample of generated
    def get_sample_id(r):
        if hasattr(r, 'sample_id'):
            return r.sample_id.lower()
        elif isinstance(r, dict):
            return str(r.get('sample_id', '')).lower()
        else:
            return ''
    
    originals = [r for r in image_records if 'original' in get_sample_id(r)]
    generated = [r for r in image_records if 'original' not in get_sample_id(r)]
    
    # Sample generated images (max 15)
    if len(generated) > 15:
        import random
        random.seed(config.RANDOM_SEED)
        generated = random.sample(generated, 15)
    
    # Combine: originals first, then generated
    to_plot = originals + generated
    
    n_images = len(to_plot)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, record in enumerate(to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Load actin channel (or phase if actin not available)
        img_path = None
        if hasattr(record, 'path_actin'):
            img_path = record.path_actin
        elif hasattr(record, 'path_phase'):
            img_path = record.path_phase
        elif isinstance(record, dict):
            img_path = record.get('path_actin') or record.get('path_phase')
        
        if img_path and Path(img_path).exists():
            img = imread(img_path)
            ax.imshow(img, cmap='gray')
        
        # Title
        if hasattr(record, 'sample_id'):
            sample_id = record.sample_id
        elif isinstance(record, dict):
            sample_id = record.get('sample_id', '')
        else:
            sample_id = ''
        ax.set_title(sample_id, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_pca_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: List[str],
    output_path: Path,
    dpi: int = None,
    n_components: int = 2
) -> None:
    """
    Create PCA scatter plot with cluster coloring.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : np.ndarray
        Cluster labels (n_samples,)
    sample_ids : list
        List of sample IDs for labeling
    output_path : Path
        Path to save figure
    dpi : int, optional
        DPI for figure. Defaults to config.FIGURE_DPI.
    n_components : int
        Number of PCA components (2 or 3)
    """
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        
        ax.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=color,
            label=f'Cluster {int(label)}',
            alpha=0.6,
            s=50
        )
    
    # Label originals with stars
    for i, sid in enumerate(sample_ids):
        if 'original' in sid.lower():
            ax.scatter(
                features_pca[i, 0],
                features_pca[i, 1],
                marker='*',
                s=300,
                c='black',
                edgecolors='white',
                linewidths=1,
                zorder=10
            )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA Clustering Visualization', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
    dpi: int = None,
    max_display: int = 50
) -> None:
    """
    Create hierarchical clustering dendrogram.
    
    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    labels : list
        List of sample labels
    output_path : Path
        Path to save figure
    dpi : int, optional
        DPI for figure. Defaults to config.FIGURE_DPI.
    max_display : int
        Maximum number of leaves to display
    """
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Truncate if too many samples
    if len(labels) > max_display:
        # Use truncated dendrogram
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=max_display,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax
        )
    else:
        dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax
        )
    
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(
    features_df: pd.DataFrame,
    output_path: Path,
    dpi: int = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    Create feature correlation heatmap.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features as columns
    output_path : Path
        Path to save figure
    dpi : int, optional
        DPI for figure. Defaults to config.FIGURE_DPI.
    figsize : tuple
        Figure size (width, height)
    """
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    # Compute correlation matrix
    corr_matrix = features_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_clustering_comparison(
    features: np.ndarray,
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    sample_ids: List[str],
    output_path: Path,
    dpi: int = None
) -> None:
    """
    Create side-by-side comparison of K-Means and Hierarchical clustering.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    kmeans_labels : np.ndarray
        K-Means cluster labels
    hierarchical_labels : np.ndarray
        Hierarchical cluster labels
    sample_ids : list
        List of sample IDs
    output_path : Path
        Path to save figure
    dpi : int, optional
        DPI for figure. Defaults to config.FIGURE_DPI.
    """
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    # Compute PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot K-Means
    unique_labels = np.unique(kmeans_labels)
    for i, label in enumerate(unique_labels):
        mask = kmeans_labels == label
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        ax1.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=color,
            label=f'Cluster {int(label)}',
            alpha=0.6,
            s=50
        )
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax1.set_title('K-Means Clustering (k=3)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Hierarchical
    unique_labels = np.unique(hierarchical_labels)
    for i, label in enumerate(unique_labels):
        mask = hierarchical_labels == label
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        ax2.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=color,
            label=f'Cluster {int(label)}',
            alpha=0.6,
            s=50
        )
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax2.set_title('Hierarchical Clustering (Ward)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

