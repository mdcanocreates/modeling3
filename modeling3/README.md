# Modeling 3: Endothelial Cell Generation, Metrics, and Clustering

This package extends Modeling 2 to generate new endothelial cell images, compute extended metrics, and perform clustering analysis.

## Overview

Modeling 3 implements:
1. **Image Generation**: ≥30 new plausible images using 2 algorithms
   - Algorithm 1: Classical biological augmentation (rotation, flip, elastic deformation, noise, brightness/contrast, zoom)
   - Algorithm 2: Thin-Plate Spline (TPS) warping for structural variants
2. **Extended Metrics**: 4 metric families
   - Family 1: Cell morphology (area, perimeter, circularity, aspect ratio)
   - Family 2: Adhesion properties (bright spot count, adhesion polarity)
   - Family 3: Actin cytoskeleton (GLCM texture, alignment metrics)
   - Family 4: Nucleus properties (area, aspect ratio, polarity)
3. **Clustering & Evaluation**: K-Means (k=3), Hierarchical (Ward), parent-match rates
4. **Publication-Ready Outputs**: Figures (300 DPI, colorblind-safe), CSVs, summaries

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy, pandas, scikit-image, matplotlib, scipy
- scikit-learn, seaborn, tqdm
- (and all Modeling 2 dependencies)

## Usage

### Basic Usage

Run the complete pipeline:

```bash
python -m modeling3.main_mc3
```

### Options

```bash
python -m modeling3.main_mc3 [OPTIONS]

Options:
  --dry-run          Preview steps without heavy computation
  --verbose          Verbose logging
  --output-dir DIR   Output directory (default: modeling3_outputs/)
  --data-root DIR    Data root directory (default: img_model/)
  --n-per-cell N     Number of images per algorithm per cell (default: 10)
```

### Example

```bash
# Dry run to preview
python -m modeling3.main_mc3 --dry-run

# Full run with verbose logging
python -m modeling3.main_mc3 --verbose

# Custom output directory
python -m modeling3.main_mc3 --output-dir my_results/
```

## Output Files

The pipeline generates the following outputs in `modeling3_outputs/` (or custom `--output-dir`):

### Data Files
- `manifest_mc3.csv`: Manifest of all images (originals + generated)
- `features_mc3.csv`: Extended metrics for all images
- `summary_mc3.txt`: Text summary of results
- `summary_mc3.json`: JSON summary with detailed metrics
- `errors_mc3.log`: Error log file

### Generated Images
- `originals/`: Original cell images (one per channel)
- `alg1/`: Algorithm 1 generated images
- `alg2/`: Algorithm 2 generated images

### Figures (300 DPI, colorblind-safe)
- `figures/image_grid.png`: Grid of original + generated images
- `figures/pca_clustering_kmeans.png`: PCA scatter plot (K-Means)
- `figures/pca_clustering_hierarchical.png`: PCA scatter plot (Hierarchical)
- `figures/clustering_comparison.png`: Side-by-side clustering comparison
- `figures/dendrogram.png`: Hierarchical clustering dendrogram
- `figures/correlation_matrix.png`: Feature correlation heatmap

## Architecture

```
modeling3/
├── __init__.py          # Package initialization
├── config.py            # Configuration (seed, image size, channels, etc.)
├── manifest.py          # ImageRecord dataclass & manifest management
├── data_loading.py      # Load original cells (reuses Modeling 2's io_utils)
├── preprocessing.py     # Image normalization and channel extraction
├── gen_alg1.py         # Algorithm 1: Classical augmentation
├── gen_alg2.py         # Algorithm 2: TPS warping
├── metrics.py          # Extended metrics (GLCM, alignment, etc.)
├── clustering.py       # K-Means, Hierarchical, Silhouette
├── evaluation.py       # Parent-match rate, intra-cluster distances
├── viz.py              # Publication-ready figures
└── main_mc3.py         # Main CLI entry point
```

## Configuration

Edit `modeling3/config.py` to customize:

- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `IMAGE_SIZE`: Target image size (default: (256, 256))
- `N_IMAGES_PER_CELL`: Images per algorithm per cell (default: 10)
- `N_CLUSTERS`: K-Means k value (default: 3)
- `FIGURE_DPI`: Figure resolution (default: 300)

## Metric Families

### Family 1: Cell Morphology
- `cell_area`: Cell spread area (µm²)
- `circularity`: Circularity (0-1)
- `aspect_ratio`: Aspect ratio (≥1.0)
- `perimeter`: Cell perimeter (pixels)

### Family 2: Adhesion Properties
- `bright_spot_count`: Number of bright spots (adhesion sites)
- `bright_spot_area`: Total area of bright spots
- `adhesion_polarity`: Distance from cell centroid to adhesion centroid

### Family 3: Actin Cytoskeleton
- `glcm_contrast`: GLCM contrast
- `glcm_homogeneity`: GLCM homogeneity
- `glcm_energy`: GLCM energy
- `glcm_correlation`: GLCM correlation
- `glcm_entropy`: GLCM entropy
- `glcm_smoothness`: Smoothness (1 - variance)
- `glcm_skewness`: Skewness
- `alignment_index`: Alignment index (1 - circular variance)
- `orientation_mean`: Mean orientation angle
- `orientation_std`: Standard deviation of orientations

### Family 4: Nucleus Properties
- `nuclear_area`: Total nuclear area (µm²)
- `nuclear_count`: Number of nuclei
- `nucleus_aspect_ratio`: Nucleus aspect ratio
- `nucleus_polarity`: Distance from cell centroid to nucleus centroid
- `nc_ratio`: Nuclear to cytoplasmic area ratio

## Clustering

The pipeline performs two clustering methods:

1. **K-Means** (k=3): Fast, partition-based clustering
2. **Hierarchical** (Ward linkage): Tree-based clustering with dendrogram

Both methods compute:
- Silhouette score
- Parent-match rate (% of generated images that cluster with their parent)
- Intra-cluster distances

## Troubleshooting

### Import Errors
If you get import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### File Not Found
Verify that `img_model/` contains:
- `CellA/` with 5 channels (Actin, BF, Combo, Microtubules, Nuclei)
- `CellB/` with 5 channels
- `CellC/` with 5 channels

### Memory Issues
If you run out of memory:
- Reduce `IMAGE_SIZE` in `config.py`
- Reduce `N_IMAGES_PER_CELL` via `--n-per-cell`

### NaNs in Metrics
If you see NaNs in `features_mc3.csv`:
- Check that segmentation masks are valid
- Verify images load correctly
- Check `errors_mc3.log` for details

## Sanity Checks

The pipeline includes automatic sanity checks:

**Pre-generation:**
- ✓ Found all 3 original cells
- ✓ Each has 4 channels
- ✓ Channels load successfully
- ✓ Dimensions reasonable

**Post-generation:**
- ✓ ≥30 images created
- ✓ All sample_ids unique
- ✓ Images visually distinct from originals

**Post-metrics:**
- ✓ No NaNs or Infs in features_mc3.csv
- ✓ Circularity in [0,1]
- ✓ Dropped constant features

## Citation

If you use this code, please cite:
- Modeling 2 pipeline (base metrics and segmentation)
- This Modeling 3 extension

## License

Same as parent project.

