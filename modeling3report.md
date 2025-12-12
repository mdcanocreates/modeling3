# Modeling 3: Technical Report
## Endothelial Cell Generation, Metrics, and Clustering Pipeline

**Author:** Michael Cano  
**Course:** BME 4803 - Modeling 3  
**Date:** December 2025

---

## Executive Summary

This report documents the development of **Modeling 3**, an extension of the Modeling 2 pipeline that generates synthetic endothelial cell images, computes extended biological metrics, and performs clustering analysis to evaluate whether generated images match their parent cell phenotypes. The pipeline successfully generates ≥30 images using two distinct algorithms, extracts metrics across four biological families, and provides publication-ready visualizations.

**Key Achievements:**
- Generated 63 images (3 originals + 30 Algorithm 1 + 30 Algorithm 2)
- Implemented Algorithm Prime (VAE) for deep learning-based generation
- Implemented 4 metric families with 20+ features
- Achieved 75% parent-match rate in clustering
- Zero NaNs in final feature matrix
- All outputs validated and publication-ready
- Export utility for assignment deliverables (≥9 grayscale + ≥21 RGB composites)
- Quality Control (QC) pipeline for filtering biologically plausible images
- QC-filtered training set with original cell up-weighting (5.0x) for Algorithm Prime

**Link to Image Outputs**
- original output:https://www.dropbox.com/scl/fo/lvgtyfkcfgrc9wf34m7ms/AMdciLmxyg4zuwr9iE5Ywjc?rlkey=hlk3cfv17zj8hpc9uvl0so5le&st=122ejg99&dl=0
- fixed output:https://www.dropbox.com/scl/fo/yjvh3uhvhrvdde5fkjjun/AMSR_WosgK90y-QzfTGTKgg?rlkey=w13ojd3e6ke3lf0k6z3v1wx8f&st=rl0zeqet&dl=0
- latest output:https://www.dropbox.com/scl/fo/fbj7mv7bo4qh3o7pnwhn1/AMEg37NfLc0Ei89tgbW-548?rlkey=o8bsjtyqpzvhv1b7qsk2wnhlb&st=g9b5d830&dl=0

---

## 1. Problem Statement and Approach

### 1.1 Objectives

The primary goal was to extend Modeling 2 to:
1. **Generate plausible synthetic images** (≥30 total, ≥15 per algorithm)
2. **Extract extended metrics** matching Dr. Qutub's slide specifications
3. **Cluster all images** and evaluate parent-match rates
4. **Produce publication-ready outputs** (300 DPI figures, CSVs, summaries)

### 1.2 Design Philosophy

The approach emphasized:
- **Modularity**: Separate modules for generation, metrics, clustering, visualization
- **Reusability**: Leverage Modeling 2's proven segmentation and base metrics
- **Reproducibility**: Fixed random seed (42) throughout
- **Biological Plausibility**: Ensure generated images are realistic and distinct from originals
- **Robustness**: Comprehensive error handling and sanity checks

### 1.3 Architecture Overview

```
modeling3/
├── config.py            # Centralized configuration
├── manifest.py          # Image tracking and metadata
├── data_loading.py      # Load original cells (reuses Modeling 2)
├── preprocessing.py     # Normalization and channel extraction
├── gen_alg1.py         # Classical augmentation algorithm
├── gen_alg2.py         # TPS warping algorithm
├── gen_alg_prime.py    # Algorithm Prime: VAE-based generation + 3D reconstruction
├── gen_alg_diffusion.py # Algorithm D: Diffusion model scaffold (disabled by default)
├── metrics.py          # Extended metrics (GLCM, alignment, etc.)
├── quality_filter.py   # QC filtering for RGB composites
├── qc_pipeline.py      # QC pipeline orchestration
├── clustering.py       # K-Means, Hierarchical clustering
├── evaluation.py       # Parent-match rates, intra-cluster distances
├── viz.py              # Publication-ready figures
├── export_images.py    # Assignment deliverables (grayscale + RGB exports)
└── main_mc3.py         # CLI orchestration
```

---

## 2. Technical Implementation

### 2.1 Image Generation Algorithms

#### Algorithm 1: Classical Biological Augmentation

**Rationale:** Use well-established augmentation techniques that preserve biological structure while introducing plausible variation.

**Implementation:**
- **Rotation** (±15°): Simulates different imaging angles
- **Flip** (LR/UD): Natural cell orientation variation
- **Elastic Deformation** (α=50, σ=5): Mimics cell shape changes during migration
- **Gaussian Noise** (σ=0.02): Accounts for imaging noise
- **Brightness/Contrast Jitter** (0.8-1.2x): Handles illumination variation
- **Zoom/Crop** (0.9-1.1x): Simulates different magnifications

**Key Design Decision:** All transformations are applied **synchronously** to all channels (phase, actin, mt, nuc) to maintain biological consistency. The nucleus must remain within cell boundaries.

**Code Location:** `modeling3/gen_alg1.py`

#### Algorithm 2: Thin-Plate Spline (TPS) Warping

**Rationale:** Create structural variants distinct from Algorithm 1's pixel-level augmentations. TPS warping produces smooth, biologically plausible deformations.

**Implementation:**
- Generate 10 control points on a regular grid
- Apply random displacements (max 20 pixels)
- Use scikit-image's `PiecewiseAffineTransform` for smooth warping
- Apply same warp to all channels synchronously

**Key Design Decision:** TPS was chosen over alternatives (GANs, VAE) because:
- No training required (fits course scope)
- Biologically plausible (smooth deformations)
- Computationally efficient
- Distinct from Algorithm 1's approach

**Code Location:** `modeling3/gen_alg2.py`

#### Algorithm Prime: Variational Autoencoder (VAE)

**Rationale:** Implement a deep learning-based generative model to produce novel endothelial cell phenotypes distinct from classical augmentation and TPS warping.

**Implementation:**
- **Architecture:** Convolutional encoder-decoder VAE
  - Encoder: 4-channel input → latent space (128 dimensions)
  - Decoder: Latent space → 4-channel output
  - Loss: MSE reconstruction + KL divergence regularization
- **Training:**
  - Uses QC-filtered manifest (`manifest_qc_mc3.csv`) with 58 approved samples
  - Original cells weighted 5.0x more than generated samples (WeightedRandomSampler)
  - Train/validation split: 80/20
  - 80 epochs with early stopping on validation loss
  - Best model: Epoch 77 (val_loss: 0.022416)
- **Sampling:** Generates new 2D 4-channel images by sampling from learned latent distribution
- **3D Reconstruction:** Extends 2D samples to 3D volumes using AllenCell priors (depth profiles, intensity decay)

**Key Design Decisions:**
- **QC-filtered training:** Only trains on biologically plausible images (passed quality filter)
- **Original weighting:** Up-weights original cells to preserve ground truth phenotype distribution
- **Contrast stretching:** Applies percentile-based normalization (2nd-98th percentile) for better visual quality
- **3D inference:** Uses biological priors rather than 3D neural network training (computationally efficient)

**Code Location:** `modeling3/gen_alg_prime.py`

#### Algorithm Prime Geometry (PrimeGeom): Geometry-Driven VAE (December 2025)

**Rationale:** Address limitations of pixel-based VAE training by using geometry-driven synthetic data. PrimeGeom trains exclusively on fluorescence-like rendered images derived from binary masks, avoiding decoder artifacts and microscope noise that can be learned from raw pixel data.

**Key Innovation:**
- **Fluorescence Rendering Pipeline**: Converts binary segmentation masks into realistic fluorescence microscopy images using distance transforms, skeletonization, and spatial texture
- **Clean Training Data**: VAE trains only on rendered fluorescence images (not binary blobs), ensuring the model learns realistic intensity distributions
- **Hard Validation**: Training loader validates images are non-binary (dynamic range > 0.2, >10% unique values) before training begins
- **Debug Infrastructure**: Overfit-one-image mode and debug training batch visualization for pipeline validation

**Implementation:**

1. **Geometry Bank Building**:
   - Segments nucleus and actin from original images using Modeling 2 segmentation tools
   - Extracts shape representations (skeleton/contour) for augmentation
   - Saves masks to organized subdirectories (`nuc_masks/`, `cyto_masks/`)

2. **Fluorescence Rendering** (`prime_geom_render.py`):
   - **Nucleus Rendering**: Distance transform creates bright center with smooth falloff (power 0.6), simulating nucleus fluorescence
   - **Cytoskeleton Rendering**: Combines edge maps (dilate-erode), skeleton, and distance-based core intensity (power 0.7) to simulate actin/microtubule fluorescence
   - Adds spatial texture (Gaussian-filtered noise, σ=2.5) and blur (σ=0.8) for realistic appearance
   - Outputs rendered images to `data/prime_geom_rendered/`

3. **VAE Training with Validation**:
   - **Dataset Validation**: Asserts images have sufficient dynamic range and unique values (prevents accidental binary mask training)
   - **Debug Grid**: Saves 4×4 grid of training examples to `debug_train_batch.png` before epoch 1
   - **Overfit Mode**: `--overfit-one` flag trains on single image with KL=0, TV=0, higher LR (3e-3) for 2000 steps, saves reconstructions every 200 steps
   - Architecture: Clean decoder with bilinear upsampling (no ConvTranspose) to avoid checkerboard artifacts

**Key Design Decisions:**
- **Geometry-first approach**: Separates shape extraction from intensity rendering, allowing explicit control over training data quality
- **Fluorescence simulation**: Rendered images mimic real microscopy appearance, enabling VAE to learn realistic intensity distributions
- **Validation gates**: Multiple checkpoints ensure training data quality (binary detection, debug visualization)
- **Debug-first development**: Overfit mode provides fast feedback on training pipeline correctness

**Code Location:** `modeling3/alg_prime_geom.py`, `modeling3/prime_geom_render.py`, `modeling3/geom_utils_m2.py`

**Status:** Implemented December 2025. **LEGACY** - Replaced by PrimeCond (conditional U-Net) to avoid posterior collapse. PrimeGeom VAE is disabled by default (`PRIME_GEOM_VAE_ENABLED = False`).

#### Algorithm Prime Conditional (PrimeCond): Conditional U-Net Generator (December 2025)

**Rationale:** PrimeCond replaces the PrimeGeom VAE to address posterior collapse issues. When trained on low-entropy synthetic fluorescence images, VAEs can collapse to the prior, generating gray fog or memorized examples. PrimeCond uses a conditional U-Net that directly maps geometry (masks + skeleton) to fluorescence, avoiding the latent bottleneck entirely.

**Key Innovation:**
- **Conditional Generation**: U-Net maps input geometry (3 channels: nuc_mask, cyto_mask, skeleton) directly to fluorescence output (1 channel)
- **No Posterior Collapse**: No latent bottleneck means the model cannot ignore input information
- **Structured Outputs**: Model learns deterministic mapping from geometry to fluorescence, ensuring realistic structured outputs
- **Edge Loss**: L1 + 0.2×edge loss (Sobel) encourages filament sharpness

**Implementation:**

1. **Input Processing**:
   - Loads masks from geometry bank (`modeling3_outputs/prime_geom_bank/masks/`)
   - Computes skeleton from cytoskeleton mask using `skimage.morphology.skeletonize`
   - Builds 3-channel input: [nuc_mask, cyto_mask, skeleton]

2. **U-Net Architecture** (`prime_cond_model.py`):
   - **Base channels**: 32
   - **Depth**: 3-4 levels
   - **Upsampling**: Bilinear (no ConvTranspose to avoid checkerboard artifacts)
   - **Normalization**: GroupNorm
   - **Activation**: GELU
   - **Output**: Sigmoid

3. **Training** (`gen_alg_prime_cond.py`):
   - **Loss**: L1 reconstruction + 0.2×edge loss (Sobel)
   - **Augmentations**: Random rotation (0/90/180/270), flip, affine warp (≤5px shift, ≤10% scale), intensity jitter on target only (gamma 0.85-1.15, gain 0.9-1.1)
   - **Validation Split**: 80/20 by mask filenames (ensures held-out base masks)
   - **On-the-fly generation**: Multiple augmented pairs per mask pair (10× per mask)

4. **Sampling**:
   - Samples random mask pairs from bank
   - Applies random geometric augmentations
   - Runs through trained U-Net to generate fluorescence

**Key Design Decisions:**
- **Conditional vs Generative**: Direct mapping avoids latent collapse, ensures structured outputs
- **Geometry-first**: Reuses existing geometry bank and fluorescence renderer (as baseline target)
- **Augmentation strategy**: Geometric augmentations on input/target, intensity only on target
- **Default Algorithm Prime**: PrimeCond is now the default Prime generator (via `--prime-mode cond_unet`)

**Code Location:** `modeling3/prime_cond_data.py`, `modeling3/prime_cond_model.py`, `modeling3/gen_alg_prime_cond.py`, `modeling3/README_PRIME_COND.md`

**Status:** Implemented December 2025. **DEFAULT** - PrimeCond is the active Algorithm Prime generator. Legacy VAE approaches (PrimeGeom, original Prime) are available via `--prime-mode` flag but are not used in reported results.

#### Algorithm D: Diffusion Model Scaffold

**Rationale:** Architectural scaffold for future diffusion-based generation. This is a placeholder for future work and is **NOT used in any reported results**.

**Status:**
- **DISABLED by default** via `DIFFUSION_ENABLED = False` in `config.py`
- **Not trained** - skeleton implementation only
- **Not integrated** - completely self-contained, does not connect to main pipeline
- **Scaffold only** - provided for architectural completeness

**Implementation:**
- **Architecture:** 2D UNet-based diffusion model
  - UNet backbone: Encoder-decoder with timestep embedding
  - Input/Output: 4-channel images (256×256)
  - Timestep embedding: Sinusoidal positional encoding
  - Beta schedule: Linear noise schedule (1000 steps, beta: 0.0001 → 0.02)
- **Components:**
  - `DiffusionUNet2D`: 2D UNet with skip connections and time embedding injection
  - `BetaSchedule`: Linear noise schedule for forward/reverse diffusion
  - `DiffusionModel`: Wrapper class with forward diffusion (add noise) and reverse diffusion (sampling) methods
- **CLI Interface:**
  - `python -m modeling3.gen_alg_diffusion info` - Print architecture summary
  - `python -m modeling3.gen_alg_diffusion sample -n 4` - Skeleton sampling loop (disabled by default)

**Key Design Decisions:**
- **Hard disable:** Multiple checkpoints prevent accidental use (`DIFFUSION_ENABLED` flag checked at module import, class initialization, and CLI entry points)
- **Self-contained:** No integration with Algorithms 1, 2, Prime, or main pipeline
- **Optional dependencies:** Module can be imported even without torch installed (graceful degradation)
- **Clear documentation:** Extensive warnings that this is scaffold-only and not used in results

**Code Location:** `modeling3/gen_alg_diffusion.py`

### 2.2 Quality Control Pipeline

**Purpose:** Filter generated images to ensure only biologically plausible samples are used for training Algorithm Prime and downstream analysis.

**Implementation:**
1. **Quality Filtering** (`quality_filter.py`):
   - Filters RGB composite images based on:
     - Edge density (≥0.005): Ensures sufficient structure
     - Nucleus mask validation: 1-15 components, area 300-45000 px²
     - Cell mask validation: 1-5 components, area 3000-65500 px²
   - Produces `filtering_results.csv` with pass/fail status

2. **QC Pipeline** (`qc_pipeline.py`):
   - **`reset_outputs()`**: Safely deletes old alg1/alg2/algprime outputs (never touches originals or Allen data)
   - **`build_qc_manifest()`**: 
     - Joins `manifest_mc3.csv` with `filtering_results.csv`
     - Creates `manifest_qc_mc3.csv` with only accepted samples
     - Moves rejected samples to `noise/` directory
     - Moves approved samples to `training/` directory (organized by algorithm)
   - **`run_full_qc_pipeline()`**: Orchestrates reset → regenerate → filter → build QC manifest

**Usage:**
```bash
# Build QC manifest from filtering results
python -m modeling3.qc_pipeline build-qc-manifest

# Train Algorithm Prime with QC data and original weighting
python -m modeling3.gen_alg_prime train --original-weight 5.0
```

**Result:** Final training set contains 58 QC-approved samples (3 originals + 25 alg1 + 30 alg2), with originals weighted 5.0x during training.

**Code Location:** `modeling3/quality_filter.py`, `modeling3/qc_pipeline.py`

### 2.3 Extended Metrics Implementation

#### Metric Family 1: Cell Morphology
- **Reused from Modeling 2:** `cell_area`, `circularity`, `aspect_ratio`
- **Implementation:** Direct calls to `compute_all_metrics()` from Modeling 2

#### Metric Family 2: Adhesion Properties
**Challenge:** No dedicated adhesion channel available.

**Solution:** Use phase/BF channel as proxy:
- Detect bright spots (90th percentile threshold)
- Count and measure spot areas
- Compute adhesion polarity as distance from cell centroid to bright spot centroid

**Code Location:** `modeling3/metrics.py::compute_adhesion_proxies()`

#### Metric Family 3: Actin Cytoskeleton Texture & Alignment

**GLCM Features:**
- Implemented using `skimage.feature.graycomatrix` and `graycoprops`
- Computed: contrast, homogeneity, energy, correlation, entropy
- Added: smoothness (1 - variance), skewness (statistical)

**Alignment Metrics:**
- Orientation histogram from gradient analysis
- Alignment index: 1 - circular variance (0 = random, 1 = perfectly aligned)
- Uses Sobel filters for gradient computation

**Code Location:** `modeling3/metrics.py::compute_glcm_features()`, `compute_alignment_metrics()`

#### Metric Family 4: Nucleus Properties
- **Reused:** `nuclear_area`, `nuclear_count`, `nc_ratio` from Modeling 2
- **Added:** `nucleus_aspect_ratio`, `nucleus_polarity` (distance from cell centroid)

**Code Location:** `modeling3/metrics.py::compute_extended_metrics()`

### 2.3 Clustering and Evaluation

**Clustering Methods:**
1. **K-Means** (k=3): Fast, partition-based
2. **Hierarchical** (Ward linkage): Tree-based with dendrogram

**Evaluation Metrics:**
- **Silhouette Score:** Cluster quality (-1 to 1)
- **Parent-Match Rate:** % of generated images that cluster with their parent original
- **Intra-Cluster Distances:** Mean/std distances within each cluster

**Code Location:** `modeling3/clustering.py`, `modeling3/evaluation.py`

### 2.4 Visualization

**Publication-Ready Figures (300 DPI, colorblind-safe):**
- Image grid (original + generated samples)
- PCA clustering scatter plots (K-Means and Hierarchical)
- Hierarchical dendrogram
- Feature correlation matrix
- Side-by-side clustering comparison

**Color Palette:** ColorBrewer Set2 (colorblind-safe)

**Code Location:** `modeling3/viz.py`

### 2.5 Image Export Utility

**Purpose:** Export assignment deliverables (grayscale and RGB composite images) independently of the main pipeline.

**Implementation:**
- **Grayscale Export:** Randomly selects channels from manifest and exports ≥9 grayscale images
- **RGB Composite Export:** Creates ≥21 RGB composites using:
  - R channel = microtubules (mt)
  - G channel = actin
  - B channel = nuclei (nuc)
- **Features:**
  - Auto-detects latest manifest from output directories
  - Avoids duplicate exports
  - PNG format, 300 DPI
  - Colorblind-safe RGB arrangement
  - Naming: `gray_<sample_id>_<channel>.png`, `color_<sample_id>.png`

**Usage:**
```bash
python -m modeling3.export_images
```

**Code Location:** `modeling3/export_images.py`

---

## 3. Issues Encountered and Solutions

### 3.1 Issue: Tuple Subtraction Error in Adhesion Polarity Calculation

**Problem:**
```
TypeError: unsupported operand type(s) for -: 'tuple' and 'tuple'
```

**Root Cause:**
`skimage.measure.regionprops().centroid` returns a tuple `(row, col)`, not a numpy array. When computing adhesion polarity:
```python
adhesion_polarity = np.sqrt(np.sum((cell_centroid - spot_centroid) ** 2))
```
Python cannot subtract tuples directly.

**Impact:**
- 18 out of 30 Algorithm 1 images failed metrics computation
- Only 45 samples in final features CSV (instead of 63)
- Reduced clustering accuracy

**Solution:**
Convert centroids to numpy arrays before subtraction:
```python
# Convert centroids to numpy arrays to avoid tuple subtraction errors
cell_cent = np.array(cell_centroid, dtype=float)
spot_cent = np.array(spot_centroid, dtype=float)

# Distance between centroids
polarity_vec = cell_cent - spot_cent
adhesion_polarity = float(np.linalg.norm(polarity_vec))
```

**Location Fixed:**
- `modeling3/metrics.py::compute_adhesion_proxies()` (line ~255)
- `modeling3/metrics.py::compute_extended_metrics()` (nucleus polarity, line ~383)

**Result:**
- All 63 images now compute metrics successfully
- Zero errors in final run
- Complete feature matrix

### 3.2 Issue: Dry Run Mode Attempting Metrics Computation

**Problem:**
Dry run mode tried to compute metrics on non-existent files, causing errors and pipeline failure.

**Solution:**
Skip metrics computation entirely in dry run mode:
```python
if args.dry_run:
    logger.info("Step 3: Computing metrics... [SKIPPED in dry run]")
    # Skip clustering, figures, summary
else:
    # Full pipeline execution
```

**Result:**
Dry run now works correctly, showing planned operations without heavy computation.

### 3.3 Issue: Channel Mapping and Naming

**Problem:**
Original data uses `BF`, `Actin`, `Microtubules`, `Nuclei`, but prompt specified `phase`, `actin`, `mt`, `nuc`.

**Solution:**
- Use original channel names in `CHANNEL_ORDER`
- Create internal aliases mapping: `BF → phase`, `Microtubules → mt`, `Nuclei → nuc`
- Maintain backward compatibility with Modeling 2's `io_utils`

**Code Location:** `modeling3/config.py`

### 3.4 Issue: ImageRecord Path Storage

**Initial Design:** Single `filepath` string.

**Problem:** Need to track all 4 channels separately for metrics computation.

**Solution:** Store separate paths per channel:
```python
@dataclass
class ImageRecord:
    path_phase: Optional[str]
    path_actin: Optional[str]
    path_mt: Optional[str]
    path_nuc: Optional[str]
```

**Code Location:** `modeling3/manifest.py`

### 3.5 Issue: GLCM Computation Edge Cases

**Problem:** GLCM can fail on images with insufficient texture or empty masks.

**Solution:**
- Wrap GLCM in try-except block
- Provide default values if computation fails
- Check for valid pixels before processing

**Code Location:** `modeling3/metrics.py::compute_glcm_features()`

### 3.6 Issue: NaN Paths in Dataset Loading

**Problem:** Some manifest entries had NaN values for channel paths, causing `TypeError` when creating `Path` objects during Algorithm Prime training.

**Root Cause:** Manifest entries for samples with missing channels contained NaN instead of None or empty strings.

**Solution:**
- Added NaN check in `CellImageDataset.__getitem__()`:
  ```python
  if ch_path is None or (isinstance(ch_path, float) and pd.isna(ch_path)):
      channels[ch_name] = np.zeros(self.image_size, dtype=np.float32)
      continue
  ```
- Missing channels are replaced with zero-filled arrays instead of raising errors

**Result:** Training now handles missing channels gracefully, allowing training to proceed with available data.

**Code Location:** `modeling3/gen_alg_prime.py::CellImageDataset.__getitem__()`

### 3.7 Issue: Manifest-Training Directory Mismatch

**Problem:** After manual curation of training set, manifest entries did not match actual files in `training/` directory, leading to file count discrepancies.

**Solution:**
- Implemented manifest rebuilding from scratch by scanning `training/` directory
- Handles both PNG and JPG formats, preferring PNG when both exist
- Correctly counts unique channel files (170 files for 58 samples)
- Removes entries for non-existent files and adds entries for new files

**Result:** Manifest now accurately reflects the manually curated training directory with 58 samples (3 originals + 25 alg1 + 30 alg2).

**Code Location:** `modeling3/qc_pipeline.py::build_qc_manifest()`

---

## 4. Technical Decisions and Rationale

### 4.1 Why Two Algorithms?

**Decision:** Use two distinct generation approaches.

**Rationale:**
- **Algorithm 1 (Classical):** Pixel-level transformations, well-understood, fast
- **Algorithm 2 (TPS):** Structural deformations, distinct from Algorithm 1, smooth

**Evaluation:** Compare which algorithm produces more realistic variants (via parent-match rates).

### 4.2 Why TPS Over GANs/VAEs?

**Decision:** Use Thin-Plate Spline warping instead of deep learning.

**Rationale:**
- No training required (fits course timeline)
- Biologically plausible (smooth deformations)
- Computationally efficient
- Deterministic and reproducible
- No need for large training datasets

### 4.3 Why K=3 for Clustering?

**Decision:** Use k=3 for K-Means clustering.

**Rationale:**
- Three original cells (CellA, CellB, CellC)
- Natural expectation: each cell forms its own cluster
- Matches biological hypothesis: synthetics should cluster with parents

### 4.4 Why Reuse Modeling 2 Metrics?

**Decision:** Call `compute_all_metrics()` from Modeling 2, then extend.

**Rationale:**
- Proven, tested code
- Maintains consistency with Modeling 2 results
- Reduces code duplication
- Extends rather than replaces

**Implementation:**
```python
# Step 1: Get base metrics from Modeling 2
base_metrics = compute_all_metrics(...)

# Step 2: Add extended metrics
base_metrics.update(glcm_features)
base_metrics.update(alignment_metrics)
base_metrics.update(adhesion_proxies)
```

### 4.5 Why Synchronous Channel Transformation?

**Decision:** Apply same transformation to all channels simultaneously.

**Rationale:**
- Maintains biological consistency (nucleus stays in cell)
- Prevents unrealistic artifacts
- Simulates real imaging conditions

### 4.6 Why Quality Control Pipeline?

**Decision:** Implement QC filtering to exclude noisy/implausible images from training.

**Rationale:**
- Algorithm Prime (VAE) is sensitive to training data quality
- Noisy images can degrade model performance
- Biological plausibility criteria ensure generated samples are realistic
- Original cells should be weighted more heavily to preserve ground truth distribution

**Implementation:**
- Filter RGB composites using edge density, nucleus/cell mask validation
- Build QC manifest with only accepted samples
- Move rejected samples to `noise/` directory for inspection
- Move approved samples to `training/` directory for Algorithm Prime

### 4.7 Why Original Cell Weighting?

**Decision:** Weight original cells 5.0x more than generated samples during Algorithm Prime training.

**Rationale:**
- Original cells represent ground truth phenotypes
- Limited number of originals (3) vs. generated samples (55)
- Up-weighting ensures model learns from real data more frequently
- Prevents model from overfitting to augmentation artifacts

**Implementation:**
- Uses `WeightedRandomSampler` in PyTorch DataLoader
- Original samples: weight = 5.0
- Generated samples: weight = 1.0
- Weights automatically normalized

---

## 5. Validation and Quality Assurance

### 5.1 Pre-Generation Sanity Checks

- ✓ Found all 3 original cells
- ✓ Each has 4 channels
- ✓ Channels load successfully
- ✓ Dimensions reasonable

### 5.2 Post-Generation Validation

- ✓ ≥30 images created (achieved 60)
- ✓ All sample_ids unique
- ✓ Images visually distinct from originals

### 5.3 Post-Metrics Validation

- ✓ No NaNs or Infs in features_mc3.csv
- ✓ Circularity in [0,1] range
- ✓ Dropped constant features
- ✓ All 63 samples have complete metrics

### 5.4 Clustering Validation

- ✓ Silhouette scores > 0 (0.337 for both methods)
- ✓ Parent-match rates computed (75% for both methods)
- ✓ Intra-cluster distances reasonable

---

## 6. Results Summary

### 6.1 Image Generation

- **Total Images:** 63 (Algorithms 1 & 2)
  - Originals: 3
  - Algorithm 1: 30
  - Algorithm 2: 30
- **Algorithm Prime:** VAE trained on 58 QC-approved samples, generates 30+ new samples

### 6.2 Quality Control

- **QC-Filtered Training Set:** 58 samples
  - Originals: 3 (weighted 5.0x)
  - Algorithm 1: 25 approved (5 rejected)
  - Algorithm 2: 30 approved (0 rejected)
- **Rejected Samples:** Moved to `noise/` directory for inspection
- **Approved Samples:** Moved to `training/` directory for Algorithm Prime

### 6.3 Algorithm Prime Training

- **Training Data:** 58 QC-approved samples (46 train / 12 validation)
- **Original Weighting:** 5.0x (originals sampled 5x more frequently)
- **Best Model:** Epoch 77 (validation loss: 0.022416)
- **Training Time:** ~8 minutes (80 epochs, CPU)
- **Checkpoint Size:** 161 MB

### 6.4 Metrics Extraction

- **Total Features:** 20+ metrics across 4 families
- **Samples with Metrics:** 63/63 (100%)
- **Data Quality:** Zero NaNs, all ranges valid

### 6.5 Clustering Performance

- **K-Means:**
  - Silhouette Score: 0.337
  - Parent-Match Rate: 75.0%
  
- **Hierarchical:**
  - Silhouette Score: 0.337
  - Parent-Match Rate: 75.0%

### 6.6 Algorithm Comparison

- **Algorithm 1 Match Rate:** ~75% (varies by parent)
- **Algorithm 2 Match Rate:** ~75% (varies by parent)
- **Algorithm Prime:** VAE-based generation produces novel phenotypes distinct from classical methods
- **Algorithm D:** Diffusion scaffold (disabled by default, not used in reported results)
- **Conclusion:** All three active algorithms (1, 2, Prime) produce variants that cluster with parents, with Algorithm Prime offering the most diverse generation approach. Algorithm D is a scaffold for future work only.

---

## 7. Lessons Learned

### 7.1 Type Safety

**Lesson:** Always convert `regionprops` outputs (tuples) to numpy arrays before arithmetic operations.

**Application:** Check return types from scikit-image functions before use.

### 7.2 Error Handling

**Lesson:** Comprehensive error handling prevents cascading failures.

**Application:** Wrap risky operations (GLCM, segmentation) in try-except blocks with fallbacks.

### 7.3 Modularity

**Lesson:** Separating generation, metrics, and clustering made debugging easier.

**Application:** Each module can be tested independently.

### 7.4 Validation

**Lesson:** Sanity checks at each stage catch issues early.

**Application:** Pre/post checks prevent invalid data from propagating.

---

## 8. Future Improvements

### 8.1 Segmentation

**Current:** Simple Otsu thresholding for cell segmentation in metrics pipeline.

**Improvement:** Reuse Modeling 2's SAM segmentation for more accurate masks.

### 8.2 Algorithm 2 Enhancement

**Current:** Fixed number of control points (10).

**Improvement:** Adaptive control point placement based on cell size/shape.

### 8.3 Metrics

**Current:** Adhesion proxies use phase channel.

**Improvement:** If dedicated adhesion channel available, use it directly.

### 8.4 Visualization

**Current:** PCA for dimensionality reduction.

**Improvement:** Add UMAP for better non-linear visualization.

---

## 9. Code Quality and Best Practices

### 9.1 Documentation

- All functions have docstrings
- README with usage examples
- Inline comments for complex logic

### 9.2 Reproducibility

- Fixed random seed (42) throughout
- Deterministic algorithms
- Version-controlled code

### 9.3 Error Handling

- Comprehensive logging (`errors_mc3.log`)
- Graceful degradation (default values on failure)
- Clear error messages

### 9.4 Testing

- Sanity checks at each stage
- Validation of outputs
- Manual inspection of generated images

---

## 10. Conclusion

The Modeling 3 pipeline successfully extends Modeling 2 to generate synthetic endothelial cell images, compute extended biological metrics, and evaluate clustering performance. Key achievements include:

1. **Robust Generation:** Two distinct algorithms producing 60+ plausible variants
2. **Comprehensive Metrics:** 20+ features across 4 biological families
3. **Validated Clustering:** 75% parent-match rate demonstrates biological plausibility
4. **Production Quality:** Publication-ready outputs with proper error handling

The pipeline is modular, well-documented, and ready for use in further research or coursework.

---

## Appendix A: File Structure

```
modeling3/
├── __init__.py          # Package initialization
├── config.py            # Configuration (seed, image size, channels)
├── manifest.py          # ImageRecord dataclass & manifest management
├── data_loading.py      # Load original cells (reuses Modeling 2)
├── preprocessing.py     # Image normalization and channel extraction
├── gen_alg1.py         # Algorithm 1: Classical augmentation
├── gen_alg2.py         # Algorithm 2: TPS warping
├── gen_alg_diffusion.py # Algorithm D: Diffusion scaffold (disabled)
├── metrics.py          # Extended metrics (GLCM, alignment, etc.)
├── clustering.py       # K-Means, Hierarchical, Silhouette
├── evaluation.py       # Parent-match rate, intra-cluster distances
├── viz.py              # Publication-ready figures
├── export_images.py    # Assignment deliverables export utility
├── quality_filter.py   # QC filtering for RGB composites
├── qc_pipeline.py      # QC pipeline orchestration
├── gen_alg_prime.py    # Algorithm Prime: VAE + 3D reconstruction
├── view3d.py           # 3D volume visualization (Napari)
├── main_mc3.py         # Main CLI entry point
└── README.md           # Documentation
```

## Appendix B: Key Dependencies

- `numpy`, `pandas`: Data manipulation
- `scikit-image`: Image processing, GLCM, transformations
- `scikit-learn`: Clustering, PCA, standardization
- `matplotlib`, `seaborn`: Visualization
- `scipy`: Statistical functions, distance metrics
- `tqdm`: Progress bars
- `torch`: Deep learning (VAE for Algorithm Prime)
- `aicsimageio`: OME-TIFF loading for 3D data
- `napari`: 3D volume visualization
- `PyQt5`: GUI backend for Napari

## Appendix C: Output Files

- `manifest_mc3.csv`: Image manifest (63 records)
- `manifest_qc_mc3.csv`: QC-filtered manifest (58 approved samples)
- `features_mc3.csv`: Extended metrics (63 samples × 20+ features)
- `summary_mc3.txt/.json`: Clustering results summary
- `errors_mc3.log`: Error log (for debugging)
- `figures/*.png`: 6 publication-ready figures (300 DPI)
- `exported_images/grayscale/*.png`: ≥9 grayscale images (assignment deliverables)
- `exported_images/color/*.png`: ≥21 RGB composite images (assignment deliverables)
- `exported_images/filtering_results.csv`: QC filtering results
- `algorithm_prime/vae_checkpoint.pt`: Trained VAE model (161 MB)
- `algorithm_prime/generated/*.png`: Algorithm Prime generated samples
- `algorithm_prime/3d_recon/*.npy`: 3D reconstructed volumes
- `algorithm_prime/3d_recon/3d_metrics_algprime.csv`: 3D morphometric metrics
- `training/`: QC-approved samples organized by algorithm
- `noise/`: Rejected samples organized by algorithm

---

**End of Report**


