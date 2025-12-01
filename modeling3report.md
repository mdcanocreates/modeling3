# Modeling 3: Technical Report
## Endothelial Cell Generation, Metrics, and Clustering Pipeline

**Author:** Michael Cano  
**Course:** BME 4803 - Modeling 3  
**Date:** November 2025

---

## Executive Summary

This report documents the development of **Modeling 3**, an extension of the Modeling 2 pipeline that generates synthetic endothelial cell images, computes extended biological metrics, and performs clustering analysis to evaluate whether generated images match their parent cell phenotypes. The pipeline successfully generates ≥30 images using two distinct algorithms, extracts metrics across four biological families, and provides publication-ready visualizations.

**Key Achievements:**
- Generated 63 images (3 originals + 30 Algorithm 1 + 30 Algorithm 2)
- Implemented 4 metric families with 20+ features
- Achieved 75% parent-match rate in clustering
- Zero NaNs in final feature matrix
- All outputs validated and publication-ready
- Export utility for assignment deliverables (≥9 grayscale + ≥21 RGB composites)

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
├── metrics.py          # Extended metrics (GLCM, alignment, etc.)
├── clustering.py       # K-Means, Hierarchical clustering
├── evaluation.py       # Parent-match rates, intra-cluster distances
├── viz.py              # Publication-ready figures
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

### 2.2 Extended Metrics Implementation

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

- **Total Images:** 63
  - Originals: 3
  - Algorithm 1: 30
  - Algorithm 2: 30

### 6.2 Metrics Extraction

- **Total Features:** 20+ metrics across 4 families
- **Samples with Metrics:** 63/63 (100%)
- **Data Quality:** Zero NaNs, all ranges valid

### 6.3 Clustering Performance

- **K-Means:**
  - Silhouette Score: 0.337
  - Parent-Match Rate: 75.0%
  
- **Hierarchical:**
  - Silhouette Score: 0.337
  - Parent-Match Rate: 75.0%

### 6.4 Algorithm Comparison

- **Algorithm 1 Match Rate:** ~75% (varies by parent)
- **Algorithm 2 Match Rate:** ~75% (varies by parent)
- **Conclusion:** Both algorithms produce variants that cluster with parents at similar rates

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
├── metrics.py          # Extended metrics (GLCM, alignment, etc.)
├── clustering.py       # K-Means, Hierarchical, Silhouette
├── evaluation.py       # Parent-match rate, intra-cluster distances
├── viz.py              # Publication-ready figures
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

## Appendix C: Output Files

- `manifest_mc3.csv`: Image manifest (63 records)
- `features_mc3.csv`: Extended metrics (63 samples × 20+ features)
- `summary_mc3.txt/.json`: Clustering results summary
- `errors_mc3.log`: Error log (for debugging)
- `figures/*.png`: 6 publication-ready figures (300 DPI)

---

**End of Report**

