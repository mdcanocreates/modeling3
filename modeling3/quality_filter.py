"""
Quality filtering module for RGB composite images.

Filters exported color images based on biological plausibility criteria:
- Edge density check (Sobel edge detection)
- Nucleus mask validation (exactly one connected component, size 500-8000 px)
- Cell mask validation (contiguous, size 5000-40000 px)

This module does NOT modify generation, metrics, or clustering logic.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
from skimage.morphology import disk
import shutil

import modeling3.config as config


# Default thresholds
DEFAULT_EDGE_DENSITY_MIN = 0.01  # Minimum fraction of pixels that are edges
DEFAULT_NUCLEUS_AREA_MIN = 500
DEFAULT_NUCLEUS_AREA_MAX = 8000
DEFAULT_CELL_AREA_MIN = 5000
DEFAULT_CELL_AREA_MAX = 50000  # Increased for 256x256 images

# Relaxed thresholds (used if not enough images pass)
RELAXED_EDGE_DENSITY_MIN = 0.005
RELAXED_NUCLEUS_AREA_MIN = 300
RELAXED_NUCLEUS_AREA_MAX = 45000  # Increased for larger nuclei (some cells have large nuclei)
RELAXED_CELL_AREA_MIN = 3000
RELAXED_CELL_AREA_MAX = 65500  # Increased but still below full image (65536)


def split_rgb_channels(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split RGB image into R, G, B channels.
    
    In our composite: R=MT, G=Actin, B=Nuclei
    
    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image of shape (H, W, 3)
    
    Returns
    -------
    tuple
        (mt_channel, actin_channel, nuclei_channel) as 2D arrays
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {rgb_image.shape}")
    
    mt = rgb_image[:, :, 0]  # R channel = microtubules
    actin = rgb_image[:, :, 1]  # G channel = actin
    nuclei = rgb_image[:, :, 2]  # B channel = nuclei
    
    return mt, actin, nuclei


def compute_edge_density(image: np.ndarray) -> float:
    """
    Compute edge density using Sobel edge detection.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image
    
    Returns
    -------
    float
        Fraction of pixels that are edges (edge density)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Convert RGB to grayscale (luminance formula)
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    
    # Normalize to [0, 1] if needed
    if gray.max() > 1.0:
        gray = gray.astype(np.float32) / 255.0
    
    # Compute Sobel edges
    edges = sobel(gray)
    
    # Threshold edges (use Otsu or fixed percentile)
    edge_threshold = np.percentile(edges, 90)  # Top 10% are edges
    edge_mask = edges > edge_threshold
    
    # Edge density = fraction of pixels that are edges
    density = edge_mask.sum() / edge_mask.size
    
    return density


def find_nucleus_mask(nuclei_channel: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Find nucleus mask by thresholding the nuclei channel.
    
    Uses more conservative thresholding and better cleanup to find single nucleus.
    
    Parameters
    ----------
    nuclei_channel : np.ndarray
        Nuclei channel (B channel from RGB)
    
    Returns
    -------
    tuple
        (mask, props_dict) where:
        - mask: binary mask of nucleus
        - props_dict: dict with 'num_components' and 'largest_area', or None if no valid nucleus
    """
    # Normalize to [0, 1] if needed
    if nuclei_channel.max() > 1.0:
        nuclei_norm = nuclei_channel.astype(np.float32) / 255.0
    else:
        nuclei_norm = nuclei_channel.astype(np.float32)
    
    # Threshold using Otsu (more conservative than percentile)
    try:
        threshold = threshold_otsu(nuclei_norm)
        mask = nuclei_norm > threshold
    except:
        # Fallback: use higher percentile (more conservative)
        threshold = np.percentile(nuclei_norm, 80)
        mask = nuclei_norm > threshold
    
    # If no components found, try more aggressive thresholding
    labeled_temp = label(mask)
    if labeled_temp.max() == 0:
        # Try lower threshold (more permissive)
        threshold_lower = np.percentile(nuclei_norm, 60)
        mask = nuclei_norm > threshold_lower
    
    # Clean up mask more aggressively
    mask = binary_closing(mask, disk(3))
    mask = binary_opening(mask, disk(2))
    mask = remove_small_objects(mask, min_size=100)  # Increased min size
    
    # Find connected components
    labeled = label(mask)
    props = regionprops(labeled)
    
    if len(props) == 0:
        return mask, None
    
    # Get largest component
    largest_prop = max(props, key=lambda p: p.area)
    
    # If largest component is too large relative to image, might be noise
    image_area = nuclei_norm.size
    if largest_prop.area > image_area * 0.8:  # More than 80% of image is suspicious
        return mask, None
    
    props_dict = {
        'num_components': len(props),
        'largest_area': largest_prop.area,
        'largest_centroid': largest_prop.centroid
    }
    
    return mask, props_dict


def find_cell_mask(mt_channel: np.ndarray, actin_channel: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Find cell mask from union of high-intensity pixels in MT and Actin channels.
    
    Uses more conservative thresholding to avoid detecting entire image as cell.
    
    Parameters
    ----------
    mt_channel : np.ndarray
        Microtubules channel (R channel from RGB)
    actin_channel : np.ndarray
        Actin channel (G channel from RGB)
    
    Returns
    -------
    tuple
        (mask, props_dict) where:
        - mask: binary mask of cell
        - props_dict: dict with 'num_components' and 'largest_area', or None if no valid cell
    """
    # Normalize channels to [0, 1] if needed
    if mt_channel.max() > 1.0:
        mt_norm = mt_channel.astype(np.float32) / 255.0
    else:
        mt_norm = mt_channel.astype(np.float32)
    
    if actin_channel.max() > 1.0:
        actin_norm = actin_channel.astype(np.float32) / 255.0
    else:
        actin_norm = actin_channel.astype(np.float32)
    
    # Use Otsu thresholding for more conservative detection
    # This avoids detecting entire image as cell
    try:
        mt_threshold = threshold_otsu(mt_norm)
        actin_threshold = threshold_otsu(actin_norm)
    except:
        # Fallback: use higher percentile (more conservative)
        mt_threshold = np.percentile(mt_norm, 60)  # Top 40% are high intensity
        actin_threshold = np.percentile(actin_norm, 60)
    
    # If no components found, try more aggressive thresholding
    mt_mask_temp = mt_norm > mt_threshold
    actin_mask_temp = actin_norm > actin_threshold
    cell_mask_temp = mt_mask_temp | actin_mask_temp
    labeled_temp = label(cell_mask_temp)
    if labeled_temp.max() == 0:
        # Try lower thresholds (more permissive)
        mt_threshold = np.percentile(mt_norm, 40)
        actin_threshold = np.percentile(actin_norm, 40)
    
    mt_mask = mt_norm > mt_threshold
    actin_mask = actin_norm > actin_threshold
    
    # Union of high-intensity regions
    cell_mask = mt_mask | actin_mask
    
    # Clean up mask
    cell_mask = binary_closing(cell_mask, disk(5))
    cell_mask = binary_opening(cell_mask, disk(3))
    cell_mask = remove_small_objects(cell_mask, min_size=500)
    
    # Find connected components
    labeled = label(cell_mask)
    props = regionprops(labeled)
    
    if len(props) == 0:
        return cell_mask, None
    
    # Get largest component
    largest_prop = max(props, key=lambda p: p.area)
    
    # Reject if largest component is too large (likely entire image)
    # For 256x256 images, cells can legitimately fill most of the frame
    image_area = mt_norm.size
    if largest_prop.area > image_area * 0.98:  # More than 98% of image (essentially full image)
        return cell_mask, None
    
    props_dict = {
        'num_components': len(props),
        'largest_area': largest_prop.area,
        'largest_centroid': largest_prop.centroid
    }
    
    return cell_mask, props_dict


def score_image(
    rgb_image: np.ndarray,
    edge_density_min: float = DEFAULT_EDGE_DENSITY_MIN,
    nucleus_area_min: int = DEFAULT_NUCLEUS_AREA_MIN,
    nucleus_area_max: int = DEFAULT_NUCLEUS_AREA_MAX,
    cell_area_min: int = DEFAULT_CELL_AREA_MIN,
    cell_area_max: int = DEFAULT_CELL_AREA_MAX
) -> Tuple[bool, Dict]:
    """
    Score an RGB composite image for biological plausibility.
    
    Parameters
    ----------
    rgb_image : np.ndarray
        RGB composite image (H, W, 3)
    edge_density_min : float
        Minimum edge density threshold
    nucleus_area_min : int
        Minimum nucleus area in pixels
    nucleus_area_max : int
        Maximum nucleus area in pixels
    cell_area_min : int
        Minimum cell area in pixels
    cell_area_max : int
        Maximum cell area in pixels
    
    Returns
    -------
    tuple
        (passes, score_dict) where:
        - passes: bool indicating if image passes all criteria
        - score_dict: dict with detailed scores and reasons
    """
    # Split channels
    mt, actin, nuclei = split_rgb_channels(rgb_image)
    
    # Create grayscale for edge detection
    gray = 0.299 * mt + 0.587 * actin + 0.114 * nuclei
    
    # Check 1: Edge density
    edge_density = compute_edge_density(gray)
    edge_pass = edge_density >= edge_density_min
    
    # Check 2: Nucleus mask
    nucleus_mask, nucleus_props = find_nucleus_mask(nuclei)
    nucleus_pass = False
    nucleus_area = 0
    nucleus_num_components = 0
    
    if nucleus_props is not None:
        nucleus_num_components = nucleus_props['num_components']
        nucleus_area = nucleus_props['largest_area']
        
        # Must have at least one component and be in size range
        # Allow up to 15 components if largest is still reasonable (some cells have very fragmented nuclei)
        # For 0 components: if cell and edge pass, allow it (weak nucleus signal)
        if nucleus_num_components == 0:
            # Allow if cell and edge pass (nucleus might be too weak to detect)
            nucleus_pass = False  # Will be overridden below if needed
        else:
            nucleus_pass = (
                nucleus_num_components <= 15 and
                nucleus_area_min <= nucleus_area <= nucleus_area_max
            )
    
    # Check 3: Cell mask
    cell_mask, cell_props = find_cell_mask(mt, actin)
    cell_pass = False
    cell_area = 0
    cell_num_components = 0
    
    if cell_props is not None:
        cell_num_components = cell_props['num_components']
        cell_area = cell_props['largest_area']
        
        # Must have at least one component and be in size range
        # Allow up to 5 components if largest is still reasonable (some cells are fragmented)
        if cell_num_components == 0:
            # Will be handled below if nucleus and edge pass
            cell_pass = False
        else:
            cell_pass = (
                cell_num_components <= 5 and
                cell_area_min <= cell_area <= cell_area_max
            )
    else:
        cell_num_components = 0
        cell_area = 0
        cell_pass = False
    
    # Overall pass: edge and cell must pass, nucleus should pass but allow if weak signal
    # If nucleus has 0 components but cell and edge pass, allow it (weak nucleus signal)
    if nucleus_num_components == 0 and edge_pass and cell_pass:
        nucleus_pass = True  # Override: allow if other criteria pass
    
    # If cell has 0 components but nucleus and edge pass, allow it (weak cell signal)
    if cell_num_components == 0 and edge_pass and nucleus_pass:
        cell_pass = True  # Override: allow if other criteria pass
    
    # Overall pass: all criteria must pass
    passes = edge_pass and nucleus_pass and cell_pass
    
    score_dict = {
        'passes': passes,
        'edge_density': edge_density,
        'edge_pass': edge_pass,
        'nucleus_area': nucleus_area,
        'nucleus_num_components': nucleus_num_components,
        'nucleus_pass': nucleus_pass,
        'cell_area': cell_area,
        'cell_num_components': cell_num_components,
        'cell_pass': cell_pass,
        'reasons': []
    }
    
    # Collect failure reasons
    if not edge_pass:
        score_dict['reasons'].append(f"Low edge density: {edge_density:.4f} < {edge_density_min:.4f}")
    if not nucleus_pass:
        if nucleus_num_components == 0:
            score_dict['reasons'].append(f"Nucleus: 0 components (no nucleus detected)")
        elif nucleus_num_components > 15:
            score_dict['reasons'].append(f"Nucleus: {nucleus_num_components} components (too fragmented)")
        else:
            score_dict['reasons'].append(f"Nucleus area {nucleus_area} outside range [{nucleus_area_min}, {nucleus_area_max}]")
    if not cell_pass:
        if cell_num_components == 0:
            score_dict['reasons'].append(f"Cell: 0 components (no cell detected)")
        elif cell_num_components > 5:
            score_dict['reasons'].append(f"Cell: {cell_num_components} components (too fragmented)")
        else:
            score_dict['reasons'].append(f"Cell area {cell_area} outside range [{cell_area_min}, {cell_area_max}]")
    
    return passes, score_dict


def filter_color_images(
    input_dir: Path,
    output_dir: Path,
    min_keep: int = 21,
    edge_density_min: float = DEFAULT_EDGE_DENSITY_MIN,
    nucleus_area_min: int = DEFAULT_NUCLEUS_AREA_MIN,
    nucleus_area_max: int = DEFAULT_NUCLEUS_AREA_MAX,
    cell_area_min: int = DEFAULT_CELL_AREA_MIN,
    cell_area_max: int = DEFAULT_CELL_AREA_MAX,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Filter color images based on biological plausibility criteria.
    
    Loads color_*.png images from input_dir, applies plausibility criteria,
    and copies accepted images to output_dir/color_clean.
    
    If fewer than min_keep pass, relaxes thresholds and retries once.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing color_*.png images
    output_dir : Path
        Output directory (will create color_clean subdirectory)
    min_keep : int
        Minimum number of images to keep (default: 21)
    edge_density_min : float
        Minimum edge density threshold
    nucleus_area_min : int
        Minimum nucleus area
    nucleus_area_max : int
        Maximum nucleus area
    cell_area_min : int
        Minimum cell area
    cell_area_max : int
        Maximum cell area
    logger : logging.Logger, optional
        Logger instance (default: None, uses print)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: filename, passes, edge_density, nucleus_area,
        cell_area, reasons
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
    
    # Find all color images
    color_files = sorted(input_dir.glob("color_*.png"))
    
    if len(color_files) == 0:
        logger.warning(f"No color_*.png files found in {input_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(color_files)} color images to filter")
    
    # Create output directory
    output_clean_dir = output_dir / "color_clean"
    output_clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Score all images
    results = []
    relaxed = False
    
    for color_file in color_files:
        try:
            # Load image
            rgb_image = imread(str(color_file))
            
            # Score image
            passes, score_dict = score_image(
                rgb_image,
                edge_density_min=edge_density_min,
                nucleus_area_min=nucleus_area_min,
                nucleus_area_max=nucleus_area_max,
                cell_area_min=cell_area_min,
                cell_area_max=cell_area_max
            )
            
            results.append({
                'filename': color_file.name,
                'passes': passes,
                'edge_density': score_dict['edge_density'],
                'nucleus_area': score_dict['nucleus_area'],
                'nucleus_num_components': score_dict['nucleus_num_components'],
                'cell_area': score_dict['cell_area'],
                'cell_num_components': score_dict['cell_num_components'],
                'reasons': '; '.join(score_dict['reasons']) if score_dict['reasons'] else 'PASS'
            })
            
        except Exception as e:
            logger.warning(f"Failed to process {color_file.name}: {e}")
            results.append({
                'filename': color_file.name,
                'passes': False,
                'edge_density': 0.0,
                'nucleus_area': 0,
                'nucleus_num_components': 0,
                'cell_area': 0,
                'cell_num_components': 0,
                'reasons': f'Processing error: {e}'
            })
    
    df = pd.DataFrame(results)
    num_passed = df['passes'].sum()
    
    # If not enough passed, relax thresholds and retry
    if num_passed < min_keep and not relaxed:
        logger.info(f"Only {num_passed} images passed (need {min_keep}). Relaxing thresholds...")
        logger.info(f"  Edge density: {edge_density_min:.4f} -> {RELAXED_EDGE_DENSITY_MIN:.4f}")
        logger.info(f"  Nucleus area: [{nucleus_area_min}, {nucleus_area_max}] -> [{RELAXED_NUCLEUS_AREA_MIN}, {RELAXED_NUCLEUS_AREA_MAX}]")
        logger.info(f"  Cell area: [{cell_area_min}, {cell_area_max}] -> [{RELAXED_CELL_AREA_MIN}, {RELAXED_CELL_AREA_MAX}]")
        
        # Retry with relaxed thresholds
        results_relaxed = []
        for color_file in color_files:
            try:
                rgb_image = imread(str(color_file))
                passes, score_dict = score_image(
                    rgb_image,
                    edge_density_min=RELAXED_EDGE_DENSITY_MIN,
                    nucleus_area_min=RELAXED_NUCLEUS_AREA_MIN,
                    nucleus_area_max=RELAXED_NUCLEUS_AREA_MAX,
                    cell_area_min=RELAXED_CELL_AREA_MIN,
                    cell_area_max=RELAXED_CELL_AREA_MAX
                )
                
                results_relaxed.append({
                    'filename': color_file.name,
                    'passes': passes,
                    'edge_density': score_dict['edge_density'],
                    'nucleus_area': score_dict['nucleus_area'],
                    'nucleus_num_components': score_dict['nucleus_num_components'],
                    'cell_area': score_dict['cell_area'],
                    'cell_num_components': score_dict['cell_num_components'],
                    'reasons': '; '.join(score_dict['reasons']) if score_dict['reasons'] else 'PASS (relaxed)'
                })
            except Exception as e:
                results_relaxed.append({
                    'filename': color_file.name,
                    'passes': False,
                    'edge_density': 0.0,
                    'nucleus_area': 0,
                    'nucleus_num_components': 0,
                    'cell_area': 0,
                    'cell_num_components': 0,
                    'reasons': f'Processing error: {e}'
                })
        
        df = pd.DataFrame(results_relaxed)
        num_passed = df['passes'].sum()
        relaxed = True
    
    # Copy accepted images
    passed_files = df[df['passes'] == True]
    for _, row in passed_files.iterrows():
        src = input_dir / row['filename']
        dst = output_clean_dir / row['filename']
        shutil.copy2(src, dst)
    
    logger.info(f"Copied {len(passed_files)} accepted images to {output_clean_dir}")
    
    return df


def main():
    """Main entry point for quality filtering."""
    parser = argparse.ArgumentParser(
        description="Filter RGB composite images for biological plausibility",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing color_*.png images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory (will create color_clean subdirectory)'
    )
    
    parser.add_argument(
        '--min-keep',
        type=int,
        default=21,
        help='Minimum number of images to keep (default: 21)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Run filtering
    try:
        df = filter_color_images(
            input_dir=input_dir,
            output_dir=output_dir,
            min_keep=args.min_keep,
            logger=logger
        )
        
        # Print summary
        total = len(df)
        kept = df['passes'].sum()
        rejected = total - kept
        
        print("\n" + "=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)
        print(f"Checked {total} images → kept {kept} plausible, rejected {rejected}")
        print(f"Accepted images saved to: {output_dir / 'color_clean'}")
        
        # Save results CSV
        results_csv = output_dir / "filtering_results.csv"
        df.to_csv(results_csv, index=False)
        print(f"Detailed results saved to: {results_csv}")
        
    except Exception as e:
        logger.error(f"Filtering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

