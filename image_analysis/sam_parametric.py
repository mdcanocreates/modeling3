"""
Parametric SAM-based cell segmentation for interactive refinement.

This module provides a parameterized version of SAM cell segmentation
that can be tuned via a UI (e.g., Streamlit) for mask refinement.
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from skimage import io, color, morphology
from skimage.morphology import remove_small_objects, disk, binary_closing, binary_opening, binary_dilation
from skimage.measure import label, regionprops
from scipy import ndimage

# Add the local SAM repo to the path
SAM_REPO_PATH = Path(__file__).parent.parent / "segment-anything-main"
if SAM_REPO_PATH.exists():
    sys.path.insert(0, str(SAM_REPO_PATH))

# Try to import SAM (optional dependency)
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


def segment_cell_with_sam(
    cell_id: str,
    actin_path: str,
    nuclei_path: Optional[str] = None,
    margin: int = 150,
    dilate_radius: int = 8,
    close_radius: int = 3,
    actin_percentile: float = 0.3,
    band_frac: float = 0.03,
    band_coverage: float = 0.4,
    device: Optional[str] = None,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run SAM-based segmentation for a single cell with tunable parameters.
    
    This function refactors the existing SAM-first pipeline to be parametric,
    allowing interactive tuning via a UI.
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell (e.g., "CellA")
    actin_path : str
        Path to Actin image
    nuclei_path : str, optional
        Path to Nuclei image for ROI estimation
    margin : int
        ROI expansion margin in pixels (default: 150)
    dilate_radius : int
        Dilation radius for including actin-rich periphery (default: 8)
    close_radius : int
        Closing radius for morphological smoothing (default: 3)
    actin_percentile : float
        Percentile threshold for high-actin region (default: 0.3)
    band_frac : float
        Fraction of image height for top/bottom band removal (default: 0.03)
    band_coverage : float
        Threshold for band removal (default: 0.4)
    device : str, optional
        Device for inference ("cuda" or "cpu")
    model_type : str
        SAM model type: "vit_b", "vit_l", or "vit_h" (default: "vit_b")
    checkpoint_path : str, optional
        Path to SAM model checkpoint file
    
    Returns
    -------
    dict
        Dictionary containing:
        - "cell_mask": boolean array in full image coordinates
        - "actin_img": the actin image (normalized)
        - "nuclei_img": the nuclei image (if available)
        - "rough_nuclei": rough nuclei mask (if available)
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install segment-anything package.")
    
    # Get checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.getenv('SAM_CHECKPOINT_PATH')
        if not checkpoint_path:
            raise ValueError("SAM_CHECKPOINT_PATH not set. Set environment variable or provide checkpoint_path.")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load main image
    image = io.imread(actin_path)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # Drop alpha channel
    
    # Ensure uint8 format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    
    # Convert to grayscale for intensity analysis
    if len(image.shape) == 3:
        actin_gray = color.rgb2gray(image)
    else:
        actin_gray = image.copy()
    if actin_gray.max() > 1.0:
        actin_gray = actin_gray.astype(np.float32) / 255.0
    else:
        actin_gray = actin_gray.astype(np.float32)
    
    # Step 1: Derive rough nuclei mask and tight ROI
    box = None
    positive_points = []
    negative_points = []
    rough_nuclei = None
    nuclei_img = None
    
    if nuclei_path and Path(nuclei_path).exists():
        # Use nuclei to estimate bounding box
        nuclei_image = io.imread(nuclei_path)
        if len(nuclei_image.shape) == 3:
            nuclei_image = color.rgb2gray(nuclei_image)
        
        # Resize nuclei image to match main image dimensions
        if nuclei_image.shape != (h, w):
            from skimage.transform import resize
            nuclei_image = resize(nuclei_image, (h, w), order=1, preserve_range=True)
        
        # Normalize nuclei image
        if nuclei_image.max() > 1.0:
            nuclei_image = nuclei_image.astype(np.float32) / 255.0
        else:
            nuclei_image = nuclei_image.astype(np.float32)
        
        nuclei_img = nuclei_image.copy()
        
        # Get rough nuclei mask: Gaussian or median blur + threshold
        from skimage import filters
        
        # Try median filter first, then Gaussian
        try:
            nuclei_blurred = filters.median(nuclei_image, footprint=morphology.disk(3))
        except:
            nuclei_blurred = ndimage.gaussian_filter(nuclei_image, sigma=1.5)
        
        # Try Otsu threshold, fallback to percentile
        try:
            nuclei_threshold = filters.threshold_otsu(nuclei_blurred)
        except:
            nuclei_threshold = np.percentile(nuclei_blurred, 85)
        
        nuclei_binary = nuclei_blurred > nuclei_threshold
        
        # Remove small objects (min_size=200 for rough nuclei)
        nuclei_binary = remove_small_objects(nuclei_binary, min_size=200)
        
        # Get largest connected component as rough_nuclei
        labeled_nuclei = label(nuclei_binary)
        if labeled_nuclei.max() > 0:
            regions = regionprops(labeled_nuclei)
            if regions:
                # Find largest component
                largest_region = max(regions, key=lambda r: r.area)
                rough_nuclei = (labeled_nuclei == largest_region.label)
                
                # Get bounding box of largest component
                min_row, min_col, max_row, max_col = largest_region.bbox
                
                # Expand bounding box by margin (relaxed: 25% of image dimensions, up to margin pixels)
                margin_y = min(int(0.25 * h), margin)
                margin_x = min(int(0.25 * w), margin)
                min_row = max(0, min_row - margin_y)
                min_col = max(0, min_col - margin_x)
                max_row = min(h, max_row + margin_y)
                max_col = min(w, max_col + margin_x)
                
                box = np.array([min_col, min_row, max_col, max_row])  # SAM expects (x1, y1, x2, y2)
                
                # Generate positive points inside box (from nuclei centroids)
                for region in regions:
                    centroid = (int(region.centroid[0]), int(region.centroid[1]))
                    if min_row <= centroid[0] < max_row and min_col <= centroid[1] < max_col:
                        positive_points.append((centroid[0], centroid[1]))
    
    # If no nuclei or box estimation failed, use brightest regions in Actin
    if box is None:
        # Find brightest regions (top 20% of pixels)
        threshold = np.percentile(actin_gray, 80)
        bright_mask = actin_gray > threshold
        
        # Get bounding box of bright regions
        labeled = label(bright_mask)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            if regions:
                # Get largest bright region
                largest = max(regions, key=lambda r: r.area)
                min_row, min_col, max_row, max_col = largest.bbox
                
                # Expand bounding box
                margin_y = min(int(0.25 * h), margin)
                margin_x = min(int(0.25 * w), margin)
                min_row = max(0, min_row - margin_y)
                min_col = max(0, min_col - margin_x)
                max_row = min(h, max_row + margin_y)
                max_col = min(w, max_col + margin_x)
                
                box = np.array([min_col, min_row, max_col, max_row])
                
                # Generate positive points from bright regions inside box
                bright_pixels = np.where(bright_mask[min_row:max_row, min_col:max_col])
                if len(bright_pixels[0]) > 0:
                    num_samples = min(5, len(bright_pixels[0]))
                    indices = np.random.choice(len(bright_pixels[0]), size=num_samples, replace=False)
                    for idx in indices:
                        row = bright_pixels[0][idx] + min_row
                        col = bright_pixels[1][idx] + min_col
                        positive_points.append((row, col))
    
    # Fallback: use center of image if no box found
    if box is None:
        center_row, center_col = h // 2, w // 2
        margin_y = min(int(0.25 * h), margin)
        margin_x = min(int(0.25 * w), margin)
        box = np.array([
            max(0, center_col - margin_x),
            max(0, center_row - margin_y),
            min(w, center_col + margin_x),
            min(h, center_row + margin_y)
        ])
        positive_points = [(center_row, center_col)]
    
    # Generate additional positive points if needed
    if len(positive_points) < 5:
        # Sample random points inside box
        box_min_row, box_min_col = int(box[1]), int(box[0])
        box_max_row, box_max_col = int(box[3]), int(box[2])
        
        for _ in range(5 - len(positive_points)):
            row = np.random.randint(box_min_row, box_max_row)
            col = np.random.randint(box_min_col, box_max_col)
            positive_points.append((row, col))
    
    # Generate negative points outside box (clear background)
    num_negative = 3
    for _ in range(num_negative):
        # Sample from edges of image
        if np.random.rand() < 0.5:
            # Top or bottom edge
            row = np.random.choice([0, h-1])
            col = np.random.randint(0, w)
        else:
            # Left or right edge
            row = np.random.randint(0, h)
            col = np.random.choice([0, w-1])
        negative_points.append((row, col))
    
    # Step 2: Load SAM model
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    
    # Create predictor
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Step 3: Prepare prompts
    all_points = positive_points + negative_points
    point_coords = np.array([[p[1], p[0]] for p in all_points])  # (col, row) for SAM
    point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
    
    # Step 4: Run SAM prediction
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],  # Add batch dimension
        multimask_output=True,
    )
    
    # Step 5: Select best mask
    best_mask = None
    best_score = -1
    
    # Compute background intensity (from edges)
    edge_pixels = np.concatenate([
        actin_gray[0, :].flatten(),
        actin_gray[-1, :].flatten(),
        actin_gray[:, 0].flatten(),
        actin_gray[:, -1].flatten()
    ])
    background_intensity = np.median(edge_pixels)
    
    # Get crude nuclei mask for overlap calculation if available
    nuclei_mask = None
    if rough_nuclei is not None:
        nuclei_mask = rough_nuclei
    
    for i, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        
        # Filter by area
        mask_area = mask_bool.sum()
        if mask_area < 500 or mask_area > h * w * 0.9:  # Not too small, not almost full image
            continue
        
        # Compute metrics for this mask
        actin_mean = actin_gray[mask_bool].mean()
        actin_above_bg = actin_mean - background_intensity
        
        # Compute overlap with nuclei if available
        nuclei_overlap = 0.0
        if nuclei_mask is not None:
            overlap = (mask_bool & nuclei_mask).sum()
            nuclei_area = nuclei_mask.sum()
            if nuclei_area > 0:
                nuclei_overlap = overlap / nuclei_area
        
        # Score this mask
        score = (
            float(scores[i]) * 0.4 +  # SAM confidence
            min(actin_above_bg * 2, 1.0) * 0.3 +  # Actin intensity (normalized)
            nuclei_overlap * 0.2 +  # Nuclei overlap
            min(mask_area / (h * w * 0.3), 1.0) * 0.1  # Area score (prefer ~30% of image)
        )
        
        if score > best_score:
            best_score = score
            best_mask = mask_bool
    
    # If no mask selected, use the highest scoring one
    if best_mask is None and len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(bool)
    
    if best_mask is None:
        # Fallback: return empty mask
        best_mask = np.zeros((h, w), dtype=bool)
    
    # Step 6: Post-processing - keep only component overlapping nuclei
    # Remove small objects first
    best_mask = remove_small_objects(best_mask, min_size=500)
    
    # If we have rough_nuclei, keep only component with maximum overlap
    if rough_nuclei is not None and np.any(rough_nuclei):
        labeled_mask = label(best_mask)
        if labeled_mask.max() > 0:
            regions = regionprops(labeled_mask)
            if regions:
                # Find component with maximum overlap with rough_nuclei
                max_overlap = -1
                best_component_label = None
                
                for region in regions:
                    # Get mask for this component
                    component_mask = (labeled_mask == region.label)
                    # Compute overlap
                    overlap = np.sum(component_mask & rough_nuclei)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_component_label = region.label
                
                # Keep only the best component
                if best_component_label is not None:
                    best_mask = (labeled_mask == best_component_label)
                else:
                    # Fallback: keep largest component
                    largest = max(regions, key=lambda r: r.area)
                    best_mask = (labeled_mask == largest.label)
    
    # Morphological smoothing (for smoothing, not shrinking)
    best_mask = binary_closing(best_mask, disk(5))
    best_mask = binary_opening(best_mask, disk(close_radius))
    
    # Fill holes
    best_mask = ndimage.binary_fill_holes(best_mask)
    
    # Gentle expansion to include more actin-rich periphery
    best_mask = binary_dilation(best_mask, disk(dilate_radius))
    
    # AND with high-actin region to avoid grabbing dark background
    # Compute percentile of actin intensity within the current mask
    mask_pixels = actin_gray[best_mask]
    if len(mask_pixels) > 0:
        p_threshold = np.percentile(mask_pixels, int(actin_percentile * 100))
        # Build high-actin region
        actin_high = actin_gray > p_threshold
        # Final mask: intersection of dilated mask and high-actin region
        best_mask = best_mask & actin_high
    
    # Re-fill holes after dilation and AND operation
    best_mask = ndimage.binary_fill_holes(best_mask)
    
    # Remove top/bottom pure-background bands (less aggressive)
    band_size = int(band_frac * h)
    if band_size > 0:
        # Check top band
        top_band_mask = best_mask[:band_size, :]
        top_band_ratio = np.sum(top_band_mask) / (band_size * w) if (band_size * w) > 0 else 0
        
        if top_band_ratio > band_coverage:  # More than threshold of mask in top band
            best_mask[:band_size, :] = False
        
        # Check bottom band
        bottom_band_mask = best_mask[-band_size:, :]
        bottom_band_ratio = np.sum(bottom_band_mask) / (band_size * w) if (band_size * w) > 0 else 0
        
        if bottom_band_ratio > band_coverage:  # More than threshold of mask in bottom band
            best_mask[-band_size:, :] = False
    
    return {
        "cell_mask": best_mask,
        "actin_img": actin_gray,
        "nuclei_img": nuclei_img,
        "rough_nuclei": rough_nuclei
    }

