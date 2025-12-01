"""
Extended metrics computation for Modeling 3.

Builds on Modeling 2's compute_all_metrics and adds:
- GLCM texture features (contrast, homogeneity, energy, correlation, entropy, smoothness, skewness)
- Alignment metrics (orientation histogram, alignment index)
- Additional morphology and adhesion metrics
"""

import numpy as np
from typing import Dict
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel_h, sobel_v
from scipy import stats
from image_analysis.metrics import compute_all_metrics
import modeling3.config as config


def compute_glcm_features(
    image: np.ndarray,
    mask: np.ndarray,
    distances: list = [1],
    angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
) -> Dict[str, float]:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) texture features.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image (normalized [0, 1])
    mask : np.ndarray
        Binary mask defining ROI
    distances : list
        List of pixel pair distances
    angles : list
        List of angles in radians
    
    Returns
    -------
    dict
        Dictionary with GLCM features:
        - contrast
        - homogeneity
        - energy
        - correlation
        - entropy
        - smoothness (1 - variance)
        - skewness
    """
    # Extract masked region
    masked_pixels = image[mask]
    
    if len(masked_pixels) == 0:
        return {
            'glcm_contrast': 0.0,
            'glcm_homogeneity': 1.0,
            'glcm_energy': 0.0,
            'glcm_correlation': 0.0,
            'glcm_entropy': 0.0,
            'glcm_smoothness': 1.0,
            'glcm_skewness': 0.0
        }
    
    # Quantize image to 8-bit for GLCM (0-255)
    img_quantized = (image * 255).astype(np.uint8)
    
    # Compute GLCM
    try:
        glcm = graycomatrix(
            img_quantized,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Compute properties
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Compute entropy manually (GLCM doesn't have it)
        # Entropy = -sum(p(i,j) * log(p(i,j)))
        glcm_flat = glcm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0]  # Remove zeros
        entropy = -np.sum(glcm_flat * np.log(glcm_flat + 1e-10))
        
    except Exception as e:
        # Fallback if GLCM fails
        contrast = 0.0
        homogeneity = 1.0
        energy = 0.0
        correlation = 0.0
        entropy = 0.0
    
    # Compute smoothness (1 - normalized variance)
    variance = np.var(masked_pixels)
    smoothness = 1.0 / (1.0 + variance)  # Normalized to [0, 1]
    
    # Compute skewness
    skewness = stats.skew(masked_pixels.flatten())
    if np.isnan(skewness):
        skewness = 0.0
    
    return {
        'glcm_contrast': float(contrast),
        'glcm_homogeneity': float(homogeneity),
        'glcm_energy': float(energy),
        'glcm_correlation': float(correlation),
        'glcm_entropy': float(entropy),
        'glcm_smoothness': float(smoothness),
        'glcm_skewness': float(skewness)
    }


def compute_alignment_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    gradient_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Compute alignment/orientation metrics for actin cytoskeleton.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image (e.g., actin channel)
    mask : np.ndarray
        Binary mask defining ROI
    gradient_threshold : float
        Minimum gradient magnitude to include
    
    Returns
    -------
    dict
        Dictionary with alignment metrics:
        - alignment_index: 1 - circular variance (0-1, 1 = perfectly aligned)
        - orientation_mean: Mean orientation angle
        - orientation_std: Standard deviation of orientations
    """
    # Normalize image if needed
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Compute gradients
    gx = sobel_h(image)
    gy = sobel_v(image)
    
    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)  # [-π, π]
    
    # Only consider pixels within mask and with sufficient gradient
    valid = mask & (magnitude > gradient_threshold)
    
    if not np.any(valid):
        return {
            'alignment_index': 0.0,
            'orientation_mean': 0.0,
            'orientation_std': 0.0
        }
    
    # Get orientations for valid pixels
    valid_orientations = orientation[valid]
    
    # Compute alignment index using circular variance
    # Circular variance = 1 - |mean(exp(iθ))|
    complex_values = np.exp(1j * valid_orientations)
    mean_complex = np.mean(complex_values)
    circular_variance = 1.0 - np.abs(mean_complex)
    alignment_index = 1.0 - circular_variance  # 1 = aligned, 0 = random
    
    # Compute mean and std of orientations
    # Handle circular statistics
    orientation_mean = np.angle(mean_complex)
    orientation_std = np.std(valid_orientations)
    
    return {
        'alignment_index': float(alignment_index),
        'orientation_mean': float(orientation_mean),
        'orientation_std': float(orientation_std)
    }


def compute_adhesion_proxies(
    image: np.ndarray,
    mask: np.ndarray,
    threshold_percentile: float = 90
) -> Dict[str, float]:
    """
    Compute adhesion property proxies.
    
    Since we don't have a dedicated adhesion channel, we use:
    - Bright spot count (adhesion sites)
    - Spot area
    - Adhesion polarity (centroid shift)
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image (could use phase/BF channel)
    mask : np.ndarray
        Binary mask (cell mask)
    threshold_percentile : float
        Percentile for bright spot detection
    
    Returns
    -------
    dict
        Dictionary with adhesion proxies:
        - bright_spot_count
        - bright_spot_area
        - adhesion_polarity
    """
    # Extract masked region
    masked_image = image[mask]
    
    if len(masked_image) == 0:
        return {
            'bright_spot_count': 0.0,
            'bright_spot_area': 0.0,
            'adhesion_polarity': 0.0
        }
    
    # Threshold for bright spots
    threshold = np.percentile(masked_image, threshold_percentile)
    
    # Find bright spots
    bright_spots = (image > threshold) & mask
    
    # Count and area
    from skimage.measure import label, regionprops
    labeled = label(bright_spots)
    regions = regionprops(labeled)
    
    bright_spot_count = len(regions)
    bright_spot_area = np.sum(bright_spots)
    
    # Adhesion polarity: distance from cell centroid to bright spot centroid
    from skimage.measure import regionprops as rp_cell
    cell_labeled = label(mask)
    cell_regions = rp_cell(cell_labeled)
    
    if len(cell_regions) > 0:
        cell_centroid = cell_regions[0].centroid
        
        if bright_spot_count > 0:
            spot_centroids = [r.centroid for r in regions]
            spot_centroid = np.mean(spot_centroids, axis=0)
            
            # Convert centroids to numpy arrays to avoid tuple subtraction errors
            cell_cent = np.array(cell_centroid, dtype=float)
            spot_cent = np.array(spot_centroid, dtype=float)
            
            # Distance between centroids
            polarity_vec = cell_cent - spot_cent
            adhesion_polarity = float(np.linalg.norm(polarity_vec))
        else:
            adhesion_polarity = 0.0
    else:
        adhesion_polarity = 0.0
    
    return {
        'bright_spot_count': float(bright_spot_count),
        'bright_spot_area': float(bright_spot_area),
        'adhesion_polarity': float(adhesion_polarity)
    }


def compute_extended_metrics(
    cell_images: Dict[str, np.ndarray],
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
    pixel_size_um: float = 1.0
) -> Dict[str, float]:
    """
    Compute all extended metrics for Modeling 3.
    
    This function:
    1. Calls Modeling 2's compute_all_metrics for base metrics
    2. Adds GLCM texture features from actin channel
    3. Adds alignment metrics from actin channel
    4. Adds adhesion proxies from phase channel
    5. Adds additional nucleus properties
    
    Parameters
    ----------
    cell_images : dict
        Dictionary with keys: 'phase', 'actin', 'mt', 'nuc'
        All images should be normalized [0, 1]
    cell_mask : np.ndarray
        Binary mask of the cell
    nucleus_mask : np.ndarray
        Binary mask of all nuclei
    cytoplasm_mask : np.ndarray
        Binary mask of the cytoplasm
    pixel_size_um : float
        Pixel size in microns per pixel
    
    Returns
    -------
    dict
        Dictionary with all metrics (base + extended)
    """
    # Extract channels
    actin_image = cell_images.get('actin')
    mt_image = cell_images.get('mt')
    phase_image = cell_images.get('phase')
    nuc_image = cell_images.get('nuc')
    
    # Step 1: Compute base metrics using Modeling 2's function
    base_metrics = compute_all_metrics(
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        cytoplasm_mask=cytoplasm_mask,
        actin_image=actin_image if actin_image is not None else np.zeros_like(cell_mask, dtype=float),
        microtubule_image=mt_image if mt_image is not None else np.zeros_like(cell_mask, dtype=float),
        pixel_size_um=pixel_size_um
    )
    
    # Step 2: Add GLCM features from actin channel (on cytoplasm)
    if actin_image is not None and np.any(cytoplasm_mask):
        glcm_features = compute_glcm_features(actin_image, cytoplasm_mask)
        base_metrics.update(glcm_features)
    else:
        # Default values if actin not available
        base_metrics.update({
            'glcm_contrast': 0.0,
            'glcm_homogeneity': 1.0,
            'glcm_energy': 0.0,
            'glcm_correlation': 0.0,
            'glcm_entropy': 0.0,
            'glcm_smoothness': 1.0,
            'glcm_skewness': 0.0
        })
    
    # Step 3: Add alignment metrics from actin channel (on cytoplasm)
    if actin_image is not None and np.any(cytoplasm_mask):
        alignment_metrics = compute_alignment_metrics(actin_image, cytoplasm_mask)
        base_metrics.update(alignment_metrics)
    else:
        base_metrics.update({
            'alignment_index': 0.0,
            'orientation_mean': 0.0,
            'orientation_std': 0.0
        })
    
    # Step 4: Add adhesion proxies from phase channel (on cell)
    if phase_image is not None and np.any(cell_mask):
        adhesion_proxies = compute_adhesion_proxies(phase_image, cell_mask)
        base_metrics.update(adhesion_proxies)
    else:
        base_metrics.update({
            'bright_spot_count': 0.0,
            'bright_spot_area': 0.0,
            'adhesion_polarity': 0.0
        })
    
    # Step 5: Add additional nucleus properties
    if nucleus_mask is not None and np.any(nucleus_mask):
        from skimage.measure import regionprops, label
        nuc_labeled = label(nucleus_mask)
        nuc_regions = regionprops(nuc_labeled)
        
        if len(nuc_regions) > 0:
            # Use largest nucleus
            largest_nuc = max(nuc_regions, key=lambda r: r.area)
            
            # Nucleus aspect ratio
            nuc_major = largest_nuc.major_axis_length
            nuc_minor = largest_nuc.minor_axis_length
            nuc_aspect_ratio = nuc_major / (nuc_minor + 1e-6)
            
            # Nucleus polarity (distance from cell centroid to nucleus centroid)
            from skimage.measure import regionprops as rp_cell
            cell_labeled = label(cell_mask)
            cell_regions = rp_cell(cell_labeled)
            
            if len(cell_regions) > 0:
                cell_centroid = cell_regions[0].centroid
                nuc_centroid = largest_nuc.centroid
                
                # Convert centroids to numpy arrays to avoid tuple subtraction errors
                cell_cent = np.array(cell_centroid, dtype=float)
                nuc_cent = np.array(nuc_centroid, dtype=float)
                
                # Distance between centroids
                polarity_vec = cell_cent - nuc_cent
                nuc_polarity = float(np.linalg.norm(polarity_vec))
            else:
                nuc_polarity = 0.0
        else:
            nuc_aspect_ratio = 1.0
            nuc_polarity = 0.0
        
        base_metrics['nucleus_aspect_ratio'] = float(nuc_aspect_ratio)
        base_metrics['nucleus_polarity'] = float(nuc_polarity)
    else:
        base_metrics['nucleus_aspect_ratio'] = 1.0
        base_metrics['nucleus_polarity'] = 0.0
    
    return base_metrics

