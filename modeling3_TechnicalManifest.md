# Modeling 3 Technical Manifest

## 1. High level overview

### 1.1 Objective

Modeling 3 extends the Modeling 2 pipeline to generate synthetic endothelial cell images using two distinct algorithms, extract extended biologically meaningful metrics, and evaluate whether synthetically generated images cluster with their parent cell phenotypes. The pipeline processes 3 original endothelial cell images (each with 4 channels: BF/phase, Actin, Microtubules, Nuclei) and generates ≥30 new plausible images (≥15 per algorithm) for downstream analysis.

### 1.2 Pipeline summary

The pipeline executes the following stages in sequence:

1. **Data Loading**: Load 3 original cell images (CellA, CellB, CellC) with 4 channels each from `img_model/` directory
2. **Preprocessing**: Normalize images to float [0,1], resize to 256×256 pixels, extract channels
3. **Image Generation**: 
   - Algorithm 1: Generate 10 images per cell using classical biological augmentation (rotation, flip, elastic deformation, noise, brightness/contrast jitter, zoom/crop)
   - Algorithm 2: Generate 10 images per cell using Thin-Plate Spline (TPS) warping for structural variants
4. **Segmentation**: Segment cell masks (from actin channel), nucleus masks (from nuclei channel), and cytoplasm masks
5. **Metric Extraction**: Compute extended metrics including morphology, adhesion proxies, actin texture (GLCM), alignment, and nucleus properties
6. **Clustering**: Perform K-Means (k=3) and Hierarchical clustering (Ward linkage) on standardized features
7. **Evaluation**: Compute silhouette scores, parent-match rates, and intra-cluster distances
8. **Visualization**: Generate publication-ready figures (300 DPI) including PCA plots, dendrograms, correlation matrices, and image grids
9. **Output**: Save manifest CSV, features CSV, summary files (JSON and TXT), error log, and all figures

### 1.3 Input data description

The pipeline loads 3 original endothelial cell images from `img_model/` directory structure. Each cell has 4 fluorescence microscopy channels:

- **BF (Brightfield/Phase)**: Phase contrast channel, internally aliased as `phase`
- **Actin**: Actin cytoskeleton channel
- **Microtubules**: Microtubule network channel, internally aliased as `mt`
- **Nuclei**: Nuclear staining channel, internally aliased as `nuc`

Images are assumed to be 8-bit (0-255) and are normalized to float [0,1] during preprocessing. All images are resized to 256×256 pixels for consistency.

## 2. Code modules and their responsibilities

### 2.1 Configuration and shared constants

**Module**: `modeling3/config.py`

Centralizes all configuration parameters including:
- `RANDOM_SEED = 42`: Ensures reproducibility across all random operations
- `IMAGE_SIZE = (256, 256)`: Target size for all images
- `CHANNEL_ORDER`: Maps original channel names to internal aliases
- `DATA_ROOT`: Path to input data directory (`img_model`)
- `OUTPUT_ROOT`: Path to output directory (`modeling3_outputs`)
- Generation parameters (rotation ranges, noise levels, TPS control points)
- Clustering parameters (`N_CLUSTERS = 3`)
- Figure settings (`FIGURE_DPI = 300`)

### 2.2 Data loading and preprocessing

**Module**: `modeling3/data_loading.py`

- `load_original_cells()`: Loads all 3 original cell images using Modeling 2's `io_utils.load_cell_images()`, normalizes sizes, and returns dictionary mapping cell_id to channel images
- `verify_cell_data()`: Validates that all cells have 4 channels with reasonable dimensions

**Module**: `modeling3/preprocessing.py`

- `normalize_to_float()`: Converts images to float32 [0,1] range (handles 8-bit, 16-bit, or already normalized)
- `resize_to_target()`: Resizes images to target size using skimage with anti-aliasing
- `extract_channels()`: Extracts and normalizes channels, maps BF→phase, Microtubules→mt, Nuclei→nuc
- `preprocess_for_generation()`: Full preprocessing pipeline (extract, normalize, resize)

### 2.3 Image generation algorithms

#### 2.3.1 Algorithm 1 classical augmentation

**Module**: `modeling3/gen_alg1.py`

Generates plausible variants using classical biological augmentation techniques. All transformations are applied synchronously to all channels to maintain biological consistency.

- `generate_classical_augs()`: Main function that generates n_per_cell images per cell
- **Transformations applied** (in sequence, same parameters for all channels):
  - Rotation: ±15° (uniform random)
  - Flip: Random left-right and/or up-down
  - Elastic deformation: 70% probability, alpha=50, sigma=5 (Gaussian-filtered random displacement fields)
  - Gaussian noise: Standard deviation up to 0.02
  - Brightness jitter: Multiplier in [0.8, 1.2]
  - Contrast jitter: Multiplier in [0.8, 1.2]
  - Zoom/crop: Factor in [0.9, 1.1] (center crop and resize)

All generated images are saved as PNG files in `output_dir/alg1/` with naming convention: `{parent_id}_alg1_{001-010}_{channel}.png`

#### 2.3.2 Algorithm 2 TPS warping

**Module**: `modeling3/gen_alg2.py`

Generates structural variants using Thin-Plate Spline (TPS) warping, which creates smooth, biologically plausible deformations distinct from Algorithm 1.

- `generate_structural_variants()`: Main function that generates n_per_cell images per cell
- `generate_tps_control_points()`: Creates 10 control points on a regular grid with random displacements (max 20 pixels)
- `apply_tps_warp()`: Applies TPS transformation using scikit-image's `PiecewiseAffineTransform` and `warp()`

Same TPS control points are applied to all channels synchronously. Generated images saved in `output_dir/alg2/` with naming: `{parent_id}_alg2_{001-010}_{channel}.png`

### 2.4 Metric computation

**Module**: `modeling3/metrics.py`

Extends Modeling 2's `compute_all_metrics()` with additional metric families. The function `compute_extended_metrics()` orchestrates:

1. **Base metrics** (from Modeling 2): `cell_area`, `circularity`, `aspect_ratio`, `nuclear_count`, `nuclear_area`, `nc_ratio`, `actin_mean_intensity`, `actin_anisotropy`, `mtub_mean_intensity`

2. **GLCM texture features** (from actin channel, computed on cytoplasm mask):
   - `compute_glcm_features()`: Computes Gray-Level Co-occurrence Matrix features at distances=[1] and angles=[0, π/4, π/2, 3π/4]
   - Features: `glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `glcm_entropy`, `glcm_smoothness`, `glcm_skewness`

3. **Alignment metrics** (from actin channel, computed on cytoplasm mask):
   - `compute_alignment_metrics()`: Uses Sobel gradients to compute orientation statistics
   - Features: `alignment_index` (1 - circular variance, 0-1 scale), `orientation_mean`, `orientation_std`

4. **Adhesion proxies** (from phase/BF channel, computed on cell mask):
   - `compute_adhesion_proxies()`: Detects bright spots (90th percentile threshold) as adhesion site proxies
   - Features: `bright_spot_count`, `bright_spot_area`, `adhesion_polarity` (distance from cell centroid to bright spot centroid)

5. **Additional nucleus properties**:
   - `nucleus_aspect_ratio`: Major/minor axis ratio of largest nucleus
   - `nucleus_polarity`: Distance from cell centroid to nucleus centroid

### 2.5 Clustering and dimensionality reduction

**Module**: `modeling3/clustering.py`

- `perform_kmeans_clustering()`: K-Means clustering with k=3, random_state=42, n_init=10. Returns labels, cluster centers, and inertia.
- `perform_hierarchical_clustering()`: Hierarchical clustering using Ward linkage, extracts k=3 clusters. Returns labels and linkage matrix for dendrogram.
- `compute_silhouette_score()`: Computes silhouette score (-1 to 1, higher is better) using sklearn's `silhouette_score()`
- `compute_distance_matrix()`: Computes pairwise Euclidean distance matrix for intra-cluster distance analysis

Features are standardized using `StandardScaler` before clustering to ensure all metrics contribute equally.

### 2.6 Evaluation and parent match analysis

**Module**: `modeling3/evaluation.py`

- `compute_parent_match_rate()`: Determines what percentage of generated images cluster with their parent cell. For each generated image, checks if its cluster label matches the cluster label of its parent's original image. Returns overall rate, rates by parent (CellA, CellB, CellC), and rates by algorithm (alg1, alg2).

- `compute_intra_cluster_distances()`: Computes mean, std, and max pairwise distances within each cluster. Returns per-cluster statistics and overall statistics.

- `compare_algorithms()`: Compares Algorithm 1 vs Algorithm 2 in terms of parent-match rate and mean intra-cluster distance to assess which algorithm produces more realistic variants.

### 2.7 Visualization and figure export

**Module**: `modeling3/viz.py`

Generates publication-ready figures (300 DPI, colorblind-safe palette):

- `plot_image_grid()`: Creates grid of original + generated images (max 15 generated shown), displays actin channel
- `plot_pca_clustering()`: 2D PCA scatter plot colored by cluster labels, originals marked with stars. Shows PC1 and PC2 with explained variance percentages.
- `plot_dendrogram()`: Hierarchical clustering dendrogram using Ward linkage matrix. Truncates if >50 samples.
- `plot_correlation_matrix()`: Heatmap of feature correlation matrix using seaborn, shows correlation coefficients
- `plot_clustering_comparison()`: Side-by-side PCA plots comparing K-Means vs Hierarchical clustering results

All figures use ColorBrewer Set2 palette for colorblind accessibility.

### 2.8 Quality control and filtering

**Module**: `modeling3/quality_filter.py`

Filters exported RGB composite images based on biological plausibility criteria. Does not modify generation, metrics, or clustering logic.

- `filter_color_images()`: Main filtering function that:
  1. Splits RGB composites into R=MT, G=Actin, B=Nuclei channels
  2. Computes edge density using Sobel edge detection (rejects if <0.005)
  3. Thresholds B channel to find nucleus mask (requires 1-15 components, area 300-45000 px)
  4. Derives cell mask from union of high-intensity R and G pixels (requires 1-5 components, area 3000-65500 px)
  5. Adaptively relaxes thresholds if fewer than `min_keep` images pass

- `score_image()`: Scores individual RGB composite for plausibility, returns pass/fail and detailed score dictionary

**Module**: `modeling3/export_images.py`

Self-contained utility for exporting assignment deliverables:

- `export_sample_images()`: Exports ≥9 grayscale images (random channel selection) and ≥21 RGB composite images
- `make_color_composite()`: Creates RGB composite with R=MT, G=Actin, B=Nuclei using percentile-based normalization (2nd-98th percentile) for enhanced contrast
- `normalize_channel()`: Normalizes grayscale images using percentile-based scaling for better visual quality

## 3. CSV file reference

### 3.1 manifest_mc3.csv

**Purpose**: Tracks all images (originals + generated) with metadata and file paths.

**Location**: `{output_dir}/manifest_mc3.csv`

**Columns**:

| Column Name | Type | Description | Notes |
|------------|------|-------------|-------|
| `sample_id` | string | Unique identifier | Format: `{parent_id}_original`, `{parent_id}_alg1_{001-010}`, or `{parent_id}_alg2_{001-010}` |
| `parent_id` | string | Parent cell ID | `CellA`, `CellB`, or `CellC` |
| `algorithm` | categorical | Generation algorithm | `original`, `alg1`, or `alg2` |
| `path_phase` | string | File path to phase/BF channel | Relative path to PNG file |
| `path_actin` | string | File path to actin channel | Relative path to PNG file |
| `path_mt` | string | File path to microtubules channel | Relative path to PNG file |
| `path_nuc` | string | File path to nuclei channel | Relative path to PNG file |

**Row count**: 63 (3 originals + 30 alg1 + 30 alg2)

### 3.2 features_mc3.csv

**Purpose**: Contains all computed metrics for each image sample.

**Location**: `{output_dir}/features_mc3.csv`

**Columns**:

| Column Name | Type | Description | Units/Scale | Metric Family |
|------------|------|-------------|-------------|---------------|
| `cell_area` | numeric | Cell spread area | pixels² (or µm² if pixel_size_um specified) | Morphology |
| `circularity` | numeric | Cell circularity | 0-1 (1 = perfect circle) | Morphology |
| `aspect_ratio` | numeric | Cell aspect ratio | ≥1.0 (major/minor axis) | Morphology |
| `nuclear_count` | numeric | Number of nuclei | count | Nucleus |
| `nuclear_area` | numeric | Total nuclear area | pixels² (or µm²) | Nucleus |
| `nc_ratio` | numeric | Nuclear to cytoplasmic area ratio | ratio (0-1) | Nucleus |
| `actin_mean_intensity` | numeric | Mean cytoplasmic actin intensity | 0-1 (normalized) | Actin |
| `actin_anisotropy` | numeric | Actin orientation order parameter | 0-1 (1 = perfectly aligned) | Actin |
| `mtub_mean_intensity` | numeric | Mean cytoplasmic microtubule intensity | 0-1 (normalized) | Microtubule |
| `glcm_contrast` | numeric | GLCM contrast (texture variation) | ≥0 (higher = more contrast) | Actin texture |
| `glcm_homogeneity` | numeric | GLCM homogeneity (texture uniformity) | 0-1 (1 = uniform) | Actin texture |
| `glcm_energy` | numeric | GLCM energy (texture order) | 0-1 (1 = ordered) | Actin texture |
| `glcm_correlation` | numeric | GLCM correlation (linear dependency) | 0-1 (1 = correlated) | Actin texture |
| `glcm_entropy` | numeric | GLCM entropy (texture randomness) | ≥0 (higher = more random) | Actin texture |
| `glcm_smoothness` | numeric | Texture smoothness (1/(1+variance)) | 0-1 (1 = smooth) | Actin texture |
| `glcm_skewness` | numeric | Intensity distribution skewness | real (0 = symmetric) | Actin texture |
| `alignment_index` | numeric | Actin alignment index (1 - circular variance) | 0-1 (1 = aligned) | Actin alignment |
| `orientation_mean` | numeric | Mean orientation angle | radians [-π, π] | Actin alignment |
| `orientation_std` | numeric | Standard deviation of orientations | radians ≥0 | Actin alignment |
| `bright_spot_count` | numeric | Number of bright spots (adhesion proxies) | count | Adhesion |
| `bright_spot_area` | numeric | Total area of bright spots | pixels² | Adhesion |
| `adhesion_polarity` | numeric | Distance from cell centroid to bright spot centroid | pixels | Adhesion |
| `nucleus_aspect_ratio` | numeric | Nucleus major/minor axis ratio | ≥1.0 | Nucleus |
| `nucleus_polarity` | numeric | Distance from cell centroid to nucleus centroid | pixels | Nucleus |
| `sample_id` | string | Sample identifier | - | Metadata |
| `parent_id` | string | Parent cell ID | - | Metadata |
| `algorithm` | categorical | Generation algorithm | - | Metadata |

**Row count**: 63 (one row per image in manifest)

**Note**: All numeric features are computed after segmentation. GLCM features use 8-bit quantization (0-255 levels) with distances=[1] and angles=[0, π/4, π/2, 3π/4], averaged across angles.

### 3.3 filtering_results.csv

**Purpose**: Results from quality filtering of exported RGB composite images.

**Location**: `{output_dir}/exported_images/filtering_results.csv`

**Columns**:

| Column Name | Type | Description | Units/Scale | Notes |
|------------|------|-------------|-------------|-------|
| `filename` | string | RGB composite filename | - | Format: `color_{sample_id}.png` |
| `passes` | boolean | Whether image passed plausibility criteria | True/False | Overall pass/fail |
| `edge_density` | numeric | Fraction of pixels that are edges (Sobel) | 0-1 | Higher = more edges |
| `nucleus_area` | numeric | Area of largest nucleus component | pixels² | 0 if no nucleus detected |
| `nucleus_num_components` | numeric | Number of connected nucleus components | count | 0-15 allowed |
| `cell_area` | numeric | Area of largest cell component | pixels² | 0 if no cell detected |
| `cell_num_components` | numeric | Number of connected cell components | count | 0-5 allowed |
| `reasons` | string | Failure reasons (if rejected) | - | Semicolon-separated list, or "PASS" |

**Row count**: 21 (number of RGB composites checked)

**Filtering criteria** (relaxed thresholds):
- Edge density: ≥0.005
- Nucleus: 1-15 components, area 300-45000 px
- Cell: 1-5 components, area 3000-65500 px
- Special handling: If nucleus or cell has 0 components but other criteria pass, image is accepted (weak signal case)

### 3.4 summary_mc3.json

**Purpose**: JSON summary of pipeline results with clustering and evaluation metrics.

**Location**: `{output_dir}/summary_mc3.json`

**Structure**:

```json
{
  "n_images": {
    "originals": 3,
    "alg1": 30,
    "alg2": 30,
    "total": 63
  },
  "clustering": {
    "kmeans": {
      "silhouette_score": 0.334,
      "parent_match_rate": 0.767,
      "mean_intra_cluster_distance": 4.348
    },
    "hierarchical": {
      "silhouette_score": 0.336,
      "parent_match_rate": 0.733,
      "mean_intra_cluster_distance": 4.140
    }
  },
  "algorithm_comparison": {
    "alg1_match_rate": 0.533,
    "alg2_match_rate": 1.0,
    "alg1_mean_intra_distance": 6.241,
    "alg2_mean_intra_distance": 3.371,
    "alg1_n_samples": 30,
    "alg2_n_samples": 30
  }
}
```

**Key values** (from `modeling3_outputs_latest/`):
- Total images: 63
- K-Means silhouette: 0.334
- K-Means parent-match: 76.7%
- Hierarchical silhouette: 0.336
- Hierarchical parent-match: 73.3%
- Algorithm 1 match rate: 53.3%
- Algorithm 2 match rate: 100.0%

### 3.5 summary_mc3.txt

**Purpose**: Human-readable text summary of pipeline results.

**Location**: `{output_dir}/summary_mc3.txt`

**Content**: Plain text summary with image counts and clustering results. Format matches the JSON summary but in readable text form.

## 4. Metric glossary

### 4.1 Morphology metrics

**`cell_area`**: Total area of the cell mask in pixels² (or µm² if `pixel_size_um` is specified). Computed as sum of all pixels in the binary cell mask. Larger values indicate more spread cells.

**`circularity`**: Measure of how circular the cell shape is. Computed as `4π × area / perimeter²`. Range: 0-1, where 1 = perfect circle, 0 = highly elongated or irregular. Lower values indicate more elongated or irregular shapes.

**`aspect_ratio`**: Ratio of major axis length to minor axis length of the cell's fitted ellipse. Computed from `skimage.measure.regionprops`. Range: ≥1.0, where 1.0 = circular, higher values = more elongated.

### 4.2 Adhesion related metrics

**`bright_spot_count`**: Number of connected components detected as bright spots in the phase/BF channel. Bright spots are defined as pixels above the 90th percentile intensity within the cell mask. Used as a proxy for adhesion sites since no dedicated adhesion channel exists.

**`bright_spot_area`**: Total area (in pixels²) of all bright spots combined. Sum of all pixels above the 90th percentile threshold within the cell mask.

**`adhesion_polarity`**: Distance (in pixels) from the cell centroid to the centroid of bright spots. Computed as `||cell_centroid - bright_spot_centroid||` using Euclidean distance. Higher values indicate more polarized adhesion distribution. If no bright spots are detected, value is 0.0.

### 4.3 Actin texture and intensity metrics

**`actin_mean_intensity`**: Mean pixel intensity of the actin channel within the cytoplasm mask. Normalized to [0,1] range. Higher values indicate stronger actin staining.

**`actin_anisotropy`**: Orientation order parameter for actin filaments. Computed from gradient orientation statistics. Range: 0-1, where 1 = perfectly aligned filaments, 0 = random orientation. Measures how directionally organized the actin cytoskeleton is.

**`glcm_contrast`**: GLCM contrast measures local intensity variation. Higher values indicate more texture contrast (edges, boundaries). Computed as `Σ(i-j)² × P(i,j)` where P is the co-occurrence probability matrix.

**`glcm_homogeneity`**: GLCM homogeneity measures texture uniformity. Range: 0-1, where 1 = uniform texture. Computed as `Σ P(i,j) / (1 + |i-j|)`. Higher values indicate smoother, more uniform texture.

**`glcm_energy`**: GLCM energy (also called angular second moment) measures texture order. Range: 0-1, where 1 = highly ordered texture. Computed as `Σ P(i,j)²`. Higher values indicate more regular, periodic patterns.

**`glcm_correlation`**: GLCM correlation measures linear dependency of gray levels. Range: 0-1, where 1 = perfectly correlated. Computed using mean and standard deviation of row and column sums of the GLCM matrix.

**`glcm_entropy`**: GLCM entropy measures texture randomness. Range: ≥0, higher values = more random texture. Computed as `-Σ P(i,j) × log(P(i,j))`. High entropy indicates complex, irregular texture.

**`glcm_smoothness`**: Texture smoothness measure. Computed as `1 / (1 + variance)` where variance is computed from masked pixels. Range: 0-1, where 1 = perfectly smooth. Normalized to avoid division issues.

**`glcm_skewness`**: Skewness of the intensity distribution within the cytoplasm mask. Computed using `scipy.stats.skew()`. Value of 0 indicates symmetric distribution, positive = right-skewed, negative = left-skewed.

### 4.4 Microtubule related metrics

**`mtub_mean_intensity`**: Mean pixel intensity of the microtubule channel within the cytoplasm mask. Normalized to [0,1] range. Higher values indicate stronger microtubule staining.

### 4.5 Nuclear metrics

**`nuclear_count`**: Number of distinct nuclei detected in the nucleus mask. Computed by counting connected components in the thresholded nucleus channel.

**`nuclear_area`**: Total area (in pixels² or µm²) of all nuclei combined. Sum of areas of all connected components in the nucleus mask.

**`nc_ratio`**: Nuclear to cytoplasmic area ratio. Computed as `nuclear_area / (cell_area - nuclear_area)`. Range: 0-1, where higher values indicate larger nucleus relative to cytoplasm.

**`nucleus_aspect_ratio`**: Aspect ratio of the largest nucleus (major/minor axis ratio). Computed from `skimage.measure.regionprops` of the largest connected component. Range: ≥1.0, where 1.0 = circular nucleus.

**`nucleus_polarity`**: Distance (in pixels) from the cell centroid to the nucleus centroid. Computed as `||cell_centroid - nucleus_centroid||` using Euclidean distance. Higher values indicate more eccentric nucleus position. If no nucleus is detected, value is 0.0.

### 4.6 GLCM texture metrics

See Section 4.3 for detailed descriptions. All GLCM metrics are computed from the actin channel within the cytoplasm mask using:
- Distances: [1] pixel
- Angles: [0, π/4, π/2, 3π/4] radians
- Levels: 256 (8-bit quantization)
- Averaged across all angles

### 4.7 Alignment and orientation metrics

**`alignment_index`**: Measure of actin filament alignment. Computed as `1 - circular_variance` where circular variance = `1 - |mean(exp(iθ))|` from gradient orientations. Range: 0-1, where 1 = perfectly aligned, 0 = random orientation. Higher values indicate more organized cytoskeleton.

**`orientation_mean`**: Mean orientation angle of actin filaments (in radians). Computed from gradient orientations using circular statistics: `angle(mean(exp(iθ)))`. Range: [-π, π]. Represents the dominant direction of actin alignment.

**`orientation_std`**: Standard deviation of orientation angles (in radians). Computed as `std(valid_orientations)` where only pixels with gradient magnitude > 0.1 are included. Higher values indicate more variable filament directions.

### 4.8 Derived or aggregate metrics

**`nc_ratio`**: Derived metric combining `nuclear_area` and `cell_area`. See Section 4.5.

**`glcm_smoothness`**: Derived from variance of intensity distribution. See Section 4.3.

**`alignment_index`**: Derived from circular variance of orientations. See Section 4.7.

All metrics are computed after segmentation (cell mask, nucleus mask, cytoplasm mask) to ensure measurements are restricted to biologically relevant regions.

## 5. Clustering and evaluation outputs

### 5.1 K means and hierarchical clustering

**K-Means Clustering**:
- Algorithm: `sklearn.cluster.KMeans` with k=3, random_state=42, n_init=10
- Input: Standardized feature matrix (all numeric columns from `features_mc3.csv` except metadata)
- Output: Cluster labels (0, 1, or 2) for each of 63 samples
- Silhouette Score: 0.334 (from `modeling3_outputs_latest/`)
- Interpretation: Positive silhouette score indicates reasonable cluster separation. Score of 0.334 suggests moderate clustering quality.

**Hierarchical Clustering**:
- Algorithm: `sklearn.cluster.AgglomerativeClustering` with Ward linkage, n_clusters=3
- Input: Same standardized feature matrix as K-Means
- Output: Cluster labels (0, 1, or 2) and linkage matrix for dendrogram visualization
- Silhouette Score: 0.336 (from `modeling3_outputs_latest/`)
- Interpretation: Similar performance to K-Means, suggesting consistent cluster structure.

**Why k=3?**: Three original cells (CellA, CellB, CellC) naturally suggest three phenotypic clusters. The goal is to determine if synthetically generated images cluster with their parent cells.

### 5.2 Parent match rate and silhouette score

**Parent-Match Rate**: Percentage of generated images that cluster with their parent cell's original image.

- **Computation**: For each generated image, find its cluster label. Find the cluster label of its parent's original image. If they match, count as a match.
- **K-Means parent-match rate**: 76.7% (from `modeling3_outputs_latest/`)
- **Hierarchical parent-match rate**: 73.3%
- **Interpretation**: High parent-match rates (>70%) suggest that synthetic images preserve phenotypic characteristics of their parent cells, indicating successful generation.

**Silhouette Score**: Measures how well samples are clustered (how similar samples are to their own cluster vs. other clusters).

- **Range**: -1 to 1, where 1 = perfect clustering, 0 = overlapping clusters, -1 = wrong clustering
- **K-Means silhouette**: 0.334
- **Hierarchical silhouette**: 0.336
- **Interpretation**: Positive scores indicate reasonable cluster separation. Scores around 0.3 suggest moderate but meaningful clustering structure.

**Intra-Cluster Distances**: Mean pairwise Euclidean distance between samples within the same cluster.

- **K-Means mean intra-cluster distance**: 4.348 (standardized feature space)
- **Hierarchical mean intra-cluster distance**: 4.140
- **Interpretation**: Lower values indicate tighter, more cohesive clusters. Hierarchical clustering produces slightly tighter clusters.

### 5.3 How synthetic images relate to parent cells

**Algorithm Comparison** (from `modeling3_outputs_latest/summary_mc3.json`):

- **Algorithm 1 (Classical Augmentation)**:
  - Parent-match rate: 53.3%
  - Mean intra-cluster distance: 6.241
  - Interpretation: Lower match rate suggests classical augmentations may introduce more variation that moves images away from parent phenotype. Higher intra-cluster distance indicates more diverse variants.

- **Algorithm 2 (TPS Warping)**:
  - Parent-match rate: 100.0%
  - Mean intra-cluster distance: 3.371
  - Interpretation: Perfect match rate indicates TPS warping preserves phenotypic characteristics extremely well. Lower intra-cluster distance suggests more consistent, tighter variants.

**Biological Implication**: TPS warping appears to generate more biologically plausible variants that maintain parent cell phenotypes, while classical augmentation produces more diverse but potentially less realistic variants.

## 6. Figures and visual outputs

All figures are saved in `{output_dir}/figures/` at 300 DPI resolution using PNG format. Colorblind-safe ColorBrewer Set2 palette is used throughout.

### 6.1 PCA clustering plots

**Files**: 
- `pca_clustering_kmeans.png`
- `pca_clustering_hierarchical.png`

**Content**: 2D scatter plots showing first two principal components (PC1 and PC2) of the standardized feature matrix. Points are colored by cluster labels (0, 1, 2). Original cell images are marked with black stars. Axes show explained variance percentages (e.g., "PC1 (45.2% variance)").

**Purpose**: Visualize cluster separation in reduced-dimensional space. Helps assess whether clusters correspond to parent cells and whether synthetic images group with their parents.

### 6.2 Dendrogram

**File**: `dendrogram.png`

**Content**: Hierarchical clustering dendrogram showing the tree structure of sample relationships. Uses Ward linkage. If >50 samples, dendrogram is truncated to show top-level structure. Leaf labels show sample IDs.

**Purpose**: Visualize hierarchical relationships between all samples. Can identify which samples are most similar and how clusters merge.

### 6.3 Feature correlation matrix

**File**: `correlation_matrix.png`

**Content**: Heatmap showing pairwise Pearson correlation coefficients between all numeric features. Uses seaborn's `heatmap()` with coolwarm colormap (blue = negative, red = positive). Correlation values are annotated on each cell.

**Purpose**: Identify highly correlated features (potential redundancy) and feature relationships. Helps understand which metrics capture similar biological information.

### 6.4 Image grid of originals and synthetics

**File**: `image_grid.png`

**Content**: Grid layout showing original cell images (3) plus a sample of generated images (up to 15). Each subplot displays the actin channel as grayscale. Titles show sample IDs. Layout: 6 columns, variable rows.

**Purpose**: Visual inspection of generated images to assess plausibility and diversity. Allows comparison of originals vs. synthetics.

### 6.5 Clustering comparison

**File**: `clustering_comparison.png`

**Content**: Side-by-side PCA plots comparing K-Means (left) vs. Hierarchical (right) clustering results. Same PCA projection, different cluster colorings. Helps visualize differences between clustering methods.

**Purpose**: Compare clustering methods visually. Assess consistency between K-Means and Hierarchical results.

## 7. Quality control criteria

### 7.1 Edge density thresholds

**Criterion**: Edge density must be ≥0.005 (relaxed from 0.01).

**Computation**: 
1. Convert RGB composite to grayscale using luminance formula: `0.299×R + 0.587×G + 0.114×B`
2. Compute Sobel edge magnitude
3. Threshold edges at 90th percentile
4. Edge density = (number of edge pixels) / (total pixels)

**Rationale**: Biologically plausible cells should have sufficient structure (edges, boundaries). Very low edge density suggests blurry, featureless, or corrupted images.

### 7.2 Nucleus detection and mask checks

**Criterion**: Nucleus mask must have 1-15 connected components, with largest component area between 300-45000 pixels² (relaxed from 500-8000).

**Computation**:
1. Threshold B channel (nuclei) using Otsu method
2. Apply morphological closing (disk radius 3) and opening (disk radius 2)
3. Remove small objects (<100 pixels)
4. Count connected components and measure largest area

**Rationale**: Endothelial cells should have one primary nucleus (or a few fragments). Too many components suggests over-segmentation or noise. Area range ensures nucleus is reasonably sized (not too small = noise, not too large = entire image).

**Special handling**: If 0 components detected but edge and cell criteria pass, image is accepted (weak nucleus signal case).

### 7.3 Cell mask validation

**Criterion**: Cell mask must have 1-5 connected components, with largest component area between 3000-65500 pixels² (relaxed from 5000-40000).

**Computation**:
1. Threshold R channel (MT) and G channel (Actin) separately using Otsu method
2. Union of high-intensity regions from both channels
3. Apply morphological closing (disk radius 5) and opening (disk radius 3)
4. Remove small objects (<500 pixels)
5. Count connected components and measure largest area

**Rationale**: Cells should be contiguous or have a few main fragments. Too many components suggests over-segmentation. Area range ensures cell fills reasonable portion of image (not too small = fragment, not too large = entire image detected as cell).

**Special handling**: If 0 components detected but edge and nucleus criteria pass, image is accepted (weak cell signal case).

### 7.4 Accepted versus rejected images

**From `modeling3_outputs_latest/exported_images/filtering_results.csv`**:
- **Total checked**: 21 RGB composite images
- **Accepted**: 21 (100%)
- **Rejected**: 0

**Relaxed thresholds used**: All images passed using relaxed criteria (edge density ≥0.005, nucleus area 300-45000 px, cell area 3000-65500 px, up to 15 nucleus components, up to 5 cell components).

**Failure reasons** (if any): Stored in `reasons` column as semicolon-separated strings. Common reasons: "Low edge density", "Nucleus area outside range", "Cell area outside range", "Too many components".

## 8. Reproducibility notes

### 8.1 How to rerun the full pipeline

**Main command**:
```bash
python -m modeling3.main_mc3 [--dry-run] [--verbose] [--output-dir DIR] [--data-root DIR] [--n-per-cell N]
```

**Arguments**:
- `--dry-run`: Preview steps without heavy computation (generates manifest only)
- `--verbose`: Enable detailed logging
- `--output-dir`: Override output directory (default: `modeling3_outputs`)
- `--data-root`: Override data root directory (default: `img_model`)
- `--n-per-cell`: Number of images per algorithm per cell (default: 10)

**Example**:
```bash
# Full pipeline run
python -m modeling3.main_mc3 --verbose

# Dry run to preview
python -m modeling3.main_mc3 --dry-run --verbose
```

**Output location**: Results saved to `{output_dir}/` (default: `modeling3_outputs/` or `modeling3_outputs_latest/`).

### 8.2 Important configuration settings

**File**: `modeling3/config.py`

**Critical settings for reproducibility**:
- `RANDOM_SEED = 42`: Must be set consistently for reproducible random number generation (used in numpy, sklearn, random modules)
- `IMAGE_SIZE = (256, 256)`: All images resized to this size
- `N_CLUSTERS = 3`: Number of clusters for K-Means and Hierarchical
- `N_IMAGES_PER_CELL = 10`: Images generated per algorithm per cell (total: 3×10×2 = 60 generated + 3 originals = 63)

**Generation parameters** (affect image diversity):
- `ALG1_ROTATION_RANGE = (-15, 15)`: Rotation angle range in degrees
- `ALG1_ELASTIC_ALPHA = 50`: Elastic deformation strength
- `ALG1_NOISE_STD = 0.02`: Gaussian noise standard deviation
- `ALG2_N_CONTROL_POINTS = 10`: Number of TPS control points
- `ALG2_DISPLACEMENT_MAX = 20`: Maximum TPS displacement in pixels

**Figure settings**:
- `FIGURE_DPI = 300`: Publication-ready resolution
- `FIGURE_FORMAT = "png"`: Output format

### 8.3 Random seed usage and determinism

**Random seed**: `RANDOM_SEED = 42` (from `config.py`)

**Where seed is used**:
1. **Image generation** (`gen_alg1.py`, `gen_alg2.py`): `np.random.seed(config.RANDOM_SEED)` before generating each cell's variants
2. **Clustering** (`clustering.py`): `random_state=config.RANDOM_SEED` in K-Means
3. **Image export** (`export_images.py`): `random.seed(config.RANDOM_SEED)` and `np.random.seed(config.RANDOM_SEED)` for reproducible image selection
4. **Visualization** (`viz.py`): `random.seed(config.RANDOM_SEED)` for reproducible image grid sampling

**Determinism**: With the same random seed, the pipeline should produce identical results across runs (same generated images, same cluster labels, same figures). However, note that:
- Image generation uses random parameters per image, so results are deterministic only if the entire pipeline runs from scratch
- Clustering results are deterministic due to fixed random_state
- Figure generation (image grid sampling) is deterministic due to fixed seed

**To reproduce exact results**: Use the same random seed (42) and ensure no changes to configuration parameters or input data.

---

**Document Version**: 1.0  
**Last Updated**: Based on `modeling3_outputs_latest/` outputs  
**Pipeline Version**: Modeling 3 (extends Modeling 2)

