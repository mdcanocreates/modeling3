"""
Segment Anything Model (SAM) wrapper for cell segmentation.

This module provides integration with Meta's Segment Anything Model (SAM)
for primary cell segmentation. SAM is used as the main segmentation method
due to its robustness to varying image quality and minimal parameter tuning.

The module provides two main functions:
1. sam_segment_cell(): Primary cell segmentation using SAM
2. refine_cell_mask_with_sam(): Optional refinement of existing masks

Requirements:
- PyTorch
- SAM model checkpoint (ViT-B or ViT-H)
- segment-anything package (from local repo)

Usage:
    Set SAM_CHECKPOINT_PATH environment variable to point to your model file.
    The pipeline automatically uses SAM for cell segmentation.
"""

import os
import sys
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
from skimage import io, color
from skimage.morphology import remove_small_objects
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
    print("Note: Segment Anything Model (SAM) not available. Install segment-anything package.")


def refine_cell_mask_with_sam(
    image_path: str,
    initial_mask: np.ndarray,
    points: Optional[List[Tuple[int, int]]] = None,
    device: Optional[str] = None,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None
) -> np.ndarray:
    """
    Use Meta's Segment Anything model to refine a given cell mask.
    
    This function uses SAM to refine a cell mask by:
    1. Loading the image (grayscale or RGB)
    2. Initializing the SAM predictor
    3. Generating seed points from the initial mask if not provided
    4. Running SAM prediction to generate a refined mask
    5. Applying morphological cleanup
    
    Parameters
    ----------
    image_path : str
        Path to Actin or Combo image (grayscale or RGB)
    initial_mask : np.ndarray
        Binary mask (boolean array, same shape as image)
    points : list of tuple, optional
        Optional seed points (e.g., nuclei centroids) as (row, col) tuples
        If None, automatically generates points from initial_mask
    device : str, optional
        Device for inference ("cuda" or "cpu")
        If None, auto-detects: "cuda" if available, else "cpu"
    model_type : str
        SAM model type: "vit_b", "vit_l", or "vit_h" (default: "vit_b")
    checkpoint_path : str, optional
        Path to SAM model checkpoint file
        If None, reads from SAM_CHECKPOINT_PATH environment variable
    
    Returns
    -------
    np.ndarray
        Refined binary mask (boolean array, same shape as initial_mask)
    """
    if not SAM_AVAILABLE:
        print("Warning: SAM not available. Returning original mask.")
        return initial_mask
    
    # Get checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.getenv('SAM_CHECKPOINT_PATH')
        if not checkpoint_path:
            print("Warning: SAM_CHECKPOINT_PATH not set. Returning original mask.")
            return initial_mask
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: SAM checkpoint not found at {checkpoint_path}. Returning original mask.")
            return initial_mask
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load image
        image = io.imread(image_path)
        
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
        
        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        
        # Create predictor
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        
        # Generate seed points if not provided
        if points is None:
            points = _generate_seed_points_from_mask(initial_mask)
        
        if len(points) == 0:
            print("Warning: No seed points generated. Returning original mask.")
            return initial_mask
        
        # Convert points to numpy array (row, col format -> SAM expects (x, y))
        point_coords = np.array([[p[1], p[0]] for p in points])  # (col, row) for SAM
        point_labels = np.ones(len(points), dtype=int)  # All foreground points
        
        # Predict mask
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        refined_mask = masks[best_idx].astype(bool)
        
        # Ensure mask has same shape as initial_mask
        if refined_mask.shape != initial_mask.shape:
            # Resize if needed (shouldn't happen, but just in case)
            from skimage.transform import resize
            refined_mask = resize(refined_mask, initial_mask.shape, order=0, preserve_range=True).astype(bool)
        
        # Apply morphological cleanup
        refined_mask = ndimage.binary_fill_holes(refined_mask)
        refined_mask = remove_small_objects(refined_mask, min_size=100)
        
        return refined_mask
        
    except Exception as e:
        print(f"Warning: SAM refinement failed: {e}. Returning original mask.")
        return initial_mask


def _generate_seed_points_from_mask(mask: np.ndarray, num_points: int = 5) -> List[Tuple[int, int]]:
    """
    Generate seed points from a binary mask.
    
    This function samples points from within the mask, prioritizing:
    1. Centroid of the mask
    2. Random points within the mask
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (boolean array)
    num_points : int
        Number of seed points to generate (default: 5)
    
    Returns
    -------
    list of tuple
        List of (row, col) tuples representing seed points
    """
    from skimage.measure import regionprops, label
    
    points = []
    
    # Get centroid of the mask
    labeled = label(mask)
    if labeled.max() > 0:
        regions = regionprops(labeled)
        if regions:
            # Use centroid of largest component
            largest = max(regions, key=lambda r: r.area)
            centroid = (int(largest.centroid[0]), int(largest.centroid[1]))
            points.append(centroid)
    
    # Add random points within the mask
    mask_pixels = np.where(mask)
    if len(mask_pixels[0]) > 0:
        num_random = min(num_points - len(points), len(mask_pixels[0]))
        if num_random > 0:
            indices = np.random.choice(len(mask_pixels[0]), size=num_random, replace=False)
            for idx in indices:
                points.append((int(mask_pixels[0][idx]), int(mask_pixels[1][idx])))
    
    return points


def sam_segment_cell(
    image_path: str,
    nuclei_image_path: Optional[str] = None,
    device: Optional[str] = None,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    box_margin: int = 100,
    num_points: int = 5,
    min_mask_area: int = 500
) -> np.ndarray:
    """
    Use Segment Anything as the PRIMARY method to segment the main cell.
    
    This function uses SAM to generate a cell mask by:
    1. Loading the Actin or Combo image
    2. Optionally using nuclei image to derive seed points and bounding box
    3. Building prompts (box + points) to guide SAM to the main cell
    4. Running SAM to get candidate masks
    5. Selecting the best mask based on criteria
    6. Post-processing the mask with morphological operations
    
    Parameters
    ----------
    image_path : str
        Path to Actin or Combo image (grayscale or RGB)
    nuclei_image_path : str, optional
        Path to Nuclei image for generating seed points and bounding box
        If None, estimates box from brightest regions in Actin
    device : str, optional
        Device for inference ("cuda" or "cpu")
        If None, auto-detects: "cuda" if available, else "cpu"
    model_type : str
        SAM model type: "vit_b", "vit_l", or "vit_h" (default: "vit_b")
    checkpoint_path : str, optional
        Path to SAM model checkpoint file
        If None, reads from SAM_CHECKPOINT_PATH environment variable
    box_margin : int
        Margin in pixels to expand around nuclei bounding box (default: 100)
    num_points : int
        Number of positive seed points to generate (default: 5)
    min_mask_area : int
        Minimum area in pixels for valid masks (default: 500)
    
    Returns
    -------
    np.ndarray
        Binary mask (boolean array) of the segmented cell
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
    image = io.imread(image_path)
    
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
    
    # Step 1: Estimate bounding box
    box = None
    positive_points = []
    negative_points = []
    
    if nuclei_image_path and Path(nuclei_image_path).exists():
        # Use nuclei to estimate bounding box
        nuclei_image = io.imread(nuclei_image_path)
        if len(nuclei_image.shape) == 3:
            nuclei_image = color.rgb2gray(nuclei_image)
        
        # Resize nuclei image to match main image dimensions
        if nuclei_image.shape != (h, w):
            from skimage.transform import resize
            nuclei_image = resize(nuclei_image, (h, w), order=1, preserve_range=True)
        
        # Normalize nuclei image
        if nuclei_image.max() > 1.0:
            nuclei_image = nuclei_image.astype(np.float32) / 255.0
        
        # Simple thresholding to get rough nuclei mask
        from scipy import ndimage
        nuclei_blurred = ndimage.gaussian_filter(nuclei_image, sigma=1.5)
        nuclei_threshold = np.percentile(nuclei_blurred, 85)
        nuclei_binary = nuclei_blurred > nuclei_threshold
        
        # Remove small objects
        from skimage.morphology import remove_small_objects
        nuclei_binary = remove_small_objects(nuclei_binary, min_size=50)
        
        # Get bounding box of nuclei
        from skimage.measure import label, regionprops
        labeled_nuclei = label(nuclei_binary)
        if labeled_nuclei.max() > 0:
            regions = regionprops(labeled_nuclei)
            if regions:
                # Get bounding box of all nuclei
                min_row = min(r.bbox[0] for r in regions)
                min_col = min(r.bbox[1] for r in regions)
                max_row = max(r.bbox[2] for r in regions)
                max_col = max(r.bbox[3] for r in regions)
                
                # Expand bounding box by margin
                min_row = max(0, min_row - box_margin)
                min_col = max(0, min_col - box_margin)
                max_row = min(h, max_row + box_margin)
                max_col = min(w, max_col + box_margin)
                
                box = np.array([min_col, min_row, max_col, max_row])  # SAM expects (x1, y1, x2, y2)
                
                # Generate positive points inside box (from nuclei centroids)
                for region in regions:
                    centroid = (int(region.centroid[0]), int(region.centroid[1]))
                    if min_row <= centroid[0] < max_row and min_col <= centroid[1] < max_col:
                        positive_points.append((centroid[0], centroid[1]))
    
    # If no nuclei or box estimation failed, use brightest regions in Actin
    if box is None:
        # Convert image to grayscale for intensity analysis
        if len(image.shape) == 3:
            actin_gray = color.rgb2gray(image)
        else:
            actin_gray = image
        
        if actin_gray.max() > 1.0:
            actin_gray = actin_gray.astype(np.float32) / 255.0
        
        # Find brightest regions (top 20% of pixels)
        threshold = np.percentile(actin_gray, 80)
        bright_mask = actin_gray > threshold
        
        # Get bounding box of bright regions
        from skimage.measure import label, regionprops
        labeled = label(bright_mask)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            if regions:
                # Get largest bright region
                largest = max(regions, key=lambda r: r.area)
                min_row, min_col, max_row, max_col = largest.bbox
                
                # Expand bounding box
                min_row = max(0, min_row - box_margin)
                min_col = max(0, min_col - box_margin)
                max_row = min(h, max_row + box_margin)
                max_col = min(w, max_col + box_margin)
                
                box = np.array([min_col, min_row, max_col, max_row])
                
                # Generate positive points from bright regions inside box
                bright_pixels = np.where(bright_mask[min_row:max_row, min_col:max_col])
                if len(bright_pixels[0]) > 0:
                    num_samples = min(num_points, len(bright_pixels[0]))
                    indices = np.random.choice(len(bright_pixels[0]), size=num_samples, replace=False)
                    for idx in indices:
                        row = bright_pixels[0][idx] + min_row
                        col = bright_pixels[1][idx] + min_col
                        positive_points.append((row, col))
    
    # Fallback: use center of image if no box found
    if box is None:
        center_row, center_col = h // 2, w // 2
        box = np.array([
            max(0, center_col - box_margin),
            max(0, center_row - box_margin),
            min(w, center_col + box_margin),
            min(h, center_row + box_margin)
        ])
        positive_points = [(center_row, center_col)]
    
    # Generate additional positive points if needed
    if len(positive_points) < num_points:
        # Sample random points inside box
        box_min_row, box_min_col = int(box[1]), int(box[0])
        box_max_row, box_max_col = int(box[3]), int(box[2])
        
        for _ in range(num_points - len(positive_points)):
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
    # Convert points to numpy array (SAM expects (x, y) format)
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
    
    # Convert nuclei image to grayscale for intensity analysis if available
    actin_gray = None
    if len(image.shape) == 3:
        actin_gray = color.rgb2gray(image)
    else:
        actin_gray = image.copy()
    if actin_gray.max() > 1.0:
        actin_gray = actin_gray.astype(np.float32) / 255.0
    
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
    if nuclei_image_path and Path(nuclei_image_path).exists():
        try:
            nuclei_img = io.imread(nuclei_image_path)
            if len(nuclei_img.shape) == 3:
                nuclei_img = color.rgb2gray(nuclei_img)
            # Resize nuclei image to match main image dimensions
            if nuclei_img.shape != (h, w):
                from skimage.transform import resize
                nuclei_img = resize(nuclei_img, (h, w), order=1, preserve_range=True)
            if nuclei_img.max() > 1.0:
                nuclei_img = nuclei_img.astype(np.float32) / 255.0
            nuclei_blurred = ndimage.gaussian_filter(nuclei_img, sigma=1.5)
            nuclei_threshold = np.percentile(nuclei_blurred, 85)
            nuclei_mask = nuclei_blurred > nuclei_threshold
            nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
        except:
            nuclei_mask = None
    
    for i, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        
        # Filter by area
        mask_area = mask_bool.sum()
        if mask_area < min_mask_area or mask_area > h * w * 0.9:  # Not too small, not almost full image
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
        # Prefer masks that:
        # 1. Have high SAM score
        # 2. Have high Actin intensity above background
        # 3. Overlap with nuclei (if available)
        # 4. Have reasonable area
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
        return np.zeros((h, w), dtype=bool)
    
    # Step 6: Post-processing
    from skimage.morphology import disk, binary_closing, binary_opening
    
    # Remove small objects
    best_mask = remove_small_objects(best_mask, min_size=min_mask_area)
    
    # Smooth edges
    best_mask = binary_closing(best_mask, disk(5))
    best_mask = binary_opening(best_mask, disk(3))
    
    # Fill holes
    best_mask = ndimage.binary_fill_holes(best_mask)
    
    # Ensure mask doesn't touch pure-black bands at top/bottom
    # Check top 10% and bottom 10% of image
    if actin_gray is not None:
        top_band = actin_gray[:h//10, :].mean()
        bottom_band = actin_gray[-h//10:, :].mean()
        
        if top_band < 0.1:  # Very dark top band
            # Remove mask if it touches top
            if best_mask[:h//10, :].any():
                best_mask[:h//10, :] = False
        
        if bottom_band < 0.1:  # Very dark bottom band
            # Remove mask if it touches bottom
            if best_mask[-h//10:, :].any():
                best_mask[-h//10:, :] = False
    
    return best_mask

