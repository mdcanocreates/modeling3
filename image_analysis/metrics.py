"""
Metrics computation for cell morphological and cytoskeletal analysis.

This module implements all seven required metrics:
1. Cell spread area
2. Cell circularity
3. Cell aspect ratio (elongation)
4. Nuclear count and total nuclear area
5. Nuclear to cytoplasmic area ratio (N:C ratio)
6. Mean cytoplasmic actin intensity and actin anisotropy
7. Mean cytoplasmic microtubule intensity
"""

from typing import Dict, Tuple
import numpy as np
from skimage import filters, measure
from skimage.measure import regionprops


def compute_cell_area(
    cell_mask: np.ndarray,
    pixel_size_um: float = 1.0
) -> float:
    """
    Compute cell spread area.
    
    Cell spread area reflects adhesion and spreading. Endothelial cells
    change area with migration, shear stress, and activation state.
    
    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell (True = cell, False = background)
    pixel_size_um : float
        Pixel size in microns per pixel. If 1.0, returns raw pixel count.
    
    Returns
    -------
    float
        Cell area in square microns (or pixels if pixel_size_um=1.0)
    """
    # Count pixels in mask
    pixel_area = np.sum(cell_mask)
    
    # Convert to square microns
    area_um2 = pixel_area * (pixel_size_um ** 2)
    
    return area_um2


def compute_circularity(
    cell_mask: np.ndarray,
    epsilon: float = 1e-6
) -> float:
    """
    Compute cell circularity.
    
    Circularity captures how round versus irregular or elongated the cell is.
    Endothelial cells often elongate along flow or matrix fibers.
    
    Definition: circularity = 4 * π * area / perimeter²
    A value of 1 represents a perfect circle.
    
    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell
    epsilon : float
        Small value to avoid division by zero
    
    Returns
    -------
    float
        Circularity value (0 to 1, where 1 is a perfect circle)
    """
    # Get region properties
    labeled = measure.label(cell_mask)
    if labeled.max() == 0:
        return 0.0
    
    regions = regionprops(labeled)
    if len(regions) == 0:
        return 0.0
    
    # Use the largest region
    region = max(regions, key=lambda r: r.area)
    
    area = region.area
    perimeter = region.perimeter
    
    # Compute circularity with epsilon to avoid division by zero
    circularity = (4.0 * np.pi * area) / ((perimeter ** 2) + epsilon)
    
    return circularity


def compute_aspect_ratio(
    cell_mask: np.ndarray,
    epsilon: float = 1e-6
) -> float:
    """
    Compute cell aspect ratio (elongation).
    
    Aspect ratio more directly captures elongation than circularity.
    High aspect ratio indicates an elongated cell.
    
    Definition: aspect_ratio = major_axis_length / minor_axis_length
    of an ellipse fit to the cell mask.
    
    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell
    epsilon : float
        Small value to avoid division by zero
    
    Returns
    -------
    float
        Aspect ratio (>= 1.0, where 1.0 is a circle)
    """
    # Get region properties
    labeled = measure.label(cell_mask)
    if labeled.max() == 0:
        return 1.0
    
    regions = regionprops(labeled)
    if len(regions) == 0:
        return 1.0
    
    # Use the largest region
    region = max(regions, key=lambda r: r.area)
    
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    
    # Compute aspect ratio with epsilon to avoid division by zero
    aspect_ratio = major_axis / (minor_axis + epsilon)
    
    return aspect_ratio


def compute_nuclear_metrics(
    nucleus_mask: np.ndarray,
    cell_mask: np.ndarray,
    pixel_size_um: float = 1.0
) -> Tuple[int, float]:
    """
    Compute nuclear count and total nuclear area.
    
    Nuclear count and total nuclear area relate to cell cycle, binucleation,
    or fusion events. The example images show a cell with two nuclei.
    
    Parameters
    ----------
    nucleus_mask : np.ndarray
        Binary mask of all nuclei (union of nuclear objects)
    cell_mask : np.ndarray
        Binary mask of the cell (to restrict nuclei to cell interior)
    pixel_size_um : float
        Pixel size in microns per pixel
    
    Returns
    -------
    tuple
        (nuclear_count, total_nuclear_area_um2)
        - nuclear_count: Number of disconnected nuclear objects within the cell
        - total_nuclear_area_um2: Sum of all nuclear object areas in square microns
    """
    # Restrict nuclei to cell interior
    nuclei_in_cell = nucleus_mask & cell_mask
    
    # Label connected components
    labeled = measure.label(nuclei_in_cell)
    
    if labeled.max() == 0:
        return 0, 0.0
    
    # Get region properties
    regions = regionprops(labeled)
    
    # Count nuclei and sum areas
    nuclear_count = len(regions)
    total_pixel_area = sum(r.area for r in regions)
    
    # Convert to square microns
    total_nuclear_area_um2 = total_pixel_area * (pixel_size_um ** 2)
    
    return nuclear_count, total_nuclear_area_um2


def compute_nc_ratio(
    cell_area_um2: float,
    total_nuclear_area_um2: float,
    epsilon: float = 1e-6
) -> float:
    """
    Compute nuclear to cytoplasmic area ratio (N:C ratio).
    
    N:C ratio is a classic morphological marker, often altered in activation
    or pathology.
    
    Definition: N:C = total_nuclear_area / cytoplasmic_area,
    where cytoplasmic_area = cell_area - total_nuclear_area
    
    Parameters
    ----------
    cell_area_um2 : float
        Total cell area in square microns
    total_nuclear_area_um2 : float
        Total nuclear area in square microns
    epsilon : float
        Small value to avoid division by zero
    
    Returns
    -------
    float
        Nuclear to cytoplasmic area ratio
    """
    cytoplasmic_area = cell_area_um2 - total_nuclear_area_um2
    
    # Protect against division by zero
    if cytoplasmic_area <= epsilon:
        return 0.0
    
    nc_ratio = total_nuclear_area_um2 / cytoplasmic_area
    
    return nc_ratio


def compute_orientation_order_parameter(
    image: np.ndarray,
    mask: np.ndarray,
    gradient_threshold: float = 0.1
) -> float:
    """
    Compute orientation order parameter (anisotropy) from image gradients.
    
    This measures how aligned the image features are within the mask.
    Values near 1 mean highly aligned fibers, values near 0 mean random orientations.
    
    Implementation uses the nematic order parameter:
    S = |mean(exp(2i θ))|, where θ is the gradient orientation.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image (e.g., Actin channel)
    mask : np.ndarray
        Binary mask defining the region of interest
    gradient_threshold : float
        Minimum gradient magnitude to include a pixel in the calculation
    
    Returns
    -------
    float
        Orientation order parameter (0 to 1, where 1 is perfectly aligned)
    """
    # Normalize image to 0-1 range if needed
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Compute image gradients using Sobel filters
    gx = filters.sobel_h(image)
    gy = filters.sobel_v(image)
    
    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)
    
    # Only consider pixels within mask and with sufficient gradient
    valid = mask & (magnitude > gradient_threshold)
    
    if not np.any(valid):
        return 0.0
    
    # Get orientations for valid pixels
    valid_orientations = orientation[valid]
    
    # Compute nematic order parameter: S = |mean(exp(2i θ))|
    # This is equivalent to the magnitude of the mean of exp(2iθ)
    complex_values = np.exp(2j * valid_orientations)
    mean_complex = np.mean(complex_values)
    S = np.abs(mean_complex)
    
    return S


def compute_actin_metrics(
    actin_image: np.ndarray,
    cytoplasm_mask: np.ndarray,
    cell_mask: np.ndarray
) -> Tuple[float, float]:
    """
    Compute mean cytoplasmic actin intensity and actin anisotropy.
    
    Actin amount and organization relate to contractility, stress fibers,
    and response to mechanical cues. We want both quantity and organization.
    
    This function now includes background normalization for accurate intensity measurements.
    
    Parameters
    ----------
    actin_image : np.ndarray
        Grayscale Actin channel image
    cytoplasm_mask : np.ndarray
        Binary mask of the cytoplasm (cell AND NOT nucleus)
    cell_mask : np.ndarray
        Binary mask of the cell (for background estimation)
    
    Returns
    -------
    tuple
        (actin_mean_intensity, actin_anisotropy)
        - actin_mean_intensity: Mean background-normalized intensity within cytoplasm (0 to 1)
        - actin_anisotropy: Orientation order parameter (0 to 1)
    """
    # Normalize image to 0-1 range if needed
    if actin_image.max() > 1.0:
        actin_image = actin_image.astype(np.float32) / 255.0
    else:
        actin_image = actin_image.astype(np.float32)
    
    # Background subtraction
    bg_mask = ~cell_mask  # Background is outside cell
    bg_pixels = actin_image[bg_mask]
    
    if len(bg_pixels) > 0:
        bg_median = np.median(bg_pixels)
        # Subtract background
        img_bs = np.clip(actin_image - bg_median, 0, None)
        
        # Normalize by 99th percentile inside cell
        cell_pixels = img_bs[cell_mask]
        if len(cell_pixels) > 0:
            scale = np.percentile(cell_pixels, 99)
            if scale > 0:
                img_norm = np.clip(img_bs / scale, 0, 1)
            else:
                img_norm = img_bs
        else:
            img_norm = img_bs
    else:
        # No background pixels, use original normalized image
        img_norm = actin_image
    
    # Compute mean intensity within cytoplasm (using normalized image)
    if not np.any(cytoplasm_mask):
        actin_mean_intensity = 0.0
    else:
        actin_mean_intensity = np.mean(img_norm[cytoplasm_mask])
    
    # Compute anisotropy using orientation order parameter (on normalized image)
    actin_anisotropy = compute_orientation_order_parameter(
        img_norm, cytoplasm_mask
    )
    
    return actin_mean_intensity, actin_anisotropy


def compute_microtubule_intensity(
    microtubule_image: np.ndarray,
    cytoplasm_mask: np.ndarray,
    cell_mask: np.ndarray
) -> float:
    """
    Compute mean cytoplasmic microtubule intensity.
    
    Microtubule density and distribution reflect polarity, trafficking,
    and organization of the cytoskeleton.
    
    This function now includes background normalization for accurate intensity measurements.
    
    Parameters
    ----------
    microtubule_image : np.ndarray
        Grayscale Microtubules channel image
    cytoplasm_mask : np.ndarray
        Binary mask of the cytoplasm
    cell_mask : np.ndarray
        Binary mask of the cell (for background estimation)
    
    Returns
    -------
    float
        Mean background-normalized intensity in Microtubules image within cytoplasm (0 to 1)
    """
    # Normalize image to 0-1 range if needed
    if microtubule_image.max() > 1.0:
        microtubule_image = microtubule_image.astype(np.float32) / 255.0
    else:
        microtubule_image = microtubule_image.astype(np.float32)
    
    # Background subtraction
    bg_mask = ~cell_mask  # Background is outside cell
    bg_pixels = microtubule_image[bg_mask]
    
    if len(bg_pixels) > 0:
        bg_median = np.median(bg_pixels)
        # Subtract background
        img_bs = np.clip(microtubule_image - bg_median, 0, None)
        
        # Normalize by 99th percentile inside cell
        cell_pixels = img_bs[cell_mask]
        if len(cell_pixels) > 0:
            scale = np.percentile(cell_pixels, 99)
            if scale > 0:
                img_norm = np.clip(img_bs / scale, 0, 1)
            else:
                img_norm = img_bs
        else:
            img_norm = img_bs
    else:
        # No background pixels, use original normalized image
        img_norm = microtubule_image
    
    # Compute mean intensity within cytoplasm (using normalized image)
    if not np.any(cytoplasm_mask):
        mtub_mean_intensity = 0.0
    else:
        mtub_mean_intensity = np.mean(img_norm[cytoplasm_mask])
    
    return mtub_mean_intensity


def compute_all_metrics(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
    actin_image: np.ndarray,
    microtubule_image: np.ndarray,
    pixel_size_um: float = 1.0
) -> Dict[str, float]:
    """
    Compute all seven required metrics for a cell.
    
    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell
    nucleus_mask : np.ndarray
        Binary mask of all nuclei
    cytoplasm_mask : np.ndarray
        Binary mask of the cytoplasm
    actin_image : np.ndarray
        Grayscale Actin channel image
    microtubule_image : np.ndarray
        Grayscale Microtubules channel image
    pixel_size_um : float
        Pixel size in microns per pixel
    
    Returns
    -------
    dict
        Dictionary with all computed metrics:
        - cell_area: Cell spread area (µm²)
        - circularity: Cell circularity (0 to 1)
        - aspect_ratio: Cell aspect ratio (>= 1.0)
        - nuclear_count: Number of nuclei
        - nuclear_area: Total nuclear area (µm²)
        - nc_ratio: Nuclear to cytoplasmic area ratio
        - actin_mean_intensity: Mean cytoplasmic actin intensity (0 to 1)
        - actin_anisotropy: Actin orientation order parameter (0 to 1)
        - mtub_mean_intensity: Mean cytoplasmic microtubule intensity (0 to 1)
    """
    # 1. Cell spread area
    cell_area = compute_cell_area(cell_mask, pixel_size_um)
    
    # 2. Cell circularity
    circularity = compute_circularity(cell_mask)
    
    # 3. Cell aspect ratio
    aspect_ratio = compute_aspect_ratio(cell_mask)
    
    # 4. Nuclear count and total nuclear area
    nuclear_count, nuclear_area = compute_nuclear_metrics(
        nucleus_mask, cell_mask, pixel_size_um
    )
    
    # 5. Nuclear to cytoplasmic area ratio
    nc_ratio = compute_nc_ratio(cell_area, nuclear_area)
    
    # 6. Mean cytoplasmic actin intensity and actin anisotropy
    actin_mean_intensity, actin_anisotropy = compute_actin_metrics(
        actin_image, cytoplasm_mask, cell_mask
    )
    
    # 7. Mean cytoplasmic microtubule intensity
    mtub_mean_intensity = compute_microtubule_intensity(
        microtubule_image, cytoplasm_mask, cell_mask
    )
    
    metrics = {
        'cell_area': cell_area,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'nuclear_count': nuclear_count,
        'nuclear_area': nuclear_area,
        'nc_ratio': nc_ratio,
        'actin_mean_intensity': actin_mean_intensity,
        'actin_anisotropy': actin_anisotropy,
        'mtub_mean_intensity': mtub_mean_intensity
    }
    
    return metrics

