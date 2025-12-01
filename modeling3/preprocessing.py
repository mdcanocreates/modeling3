"""
Preprocessing utilities for image normalization and channel extraction.
"""

import numpy as np
from typing import Dict, Optional
from skimage.transform import resize
import modeling3.config as config


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to float [0, 1] range.
    
    Assumes input is 8-bit (0-255) or already normalized.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (any dtype)
    
    Returns
    -------
    np.ndarray
        Normalized float image [0, 1]
    """
    if image.dtype == np.uint8:
        # 8-bit image: normalize to [0, 1]
        return image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        # Likely 16-bit or other: normalize by max
        return (image.astype(np.float32) / image.max()).clip(0, 1)
    else:
        # Already normalized
        return image.astype(np.float32)


def resize_to_target(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image to target size, preserving aspect ratio if needed.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D array)
    target_size : tuple
        Target (height, width)
    
    Returns
    -------
    np.ndarray
        Resized image (same dtype as input)
    """
    if image.shape[:2] == target_size:
        return image
    
    # Resize using skimage (preserves range)
    resized = resize(
        image,
        target_size,
        preserve_range=True,
        anti_aliasing=True
    )
    
    # Preserve dtype
    return resized.astype(image.dtype)


def extract_channels(cell_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract and normalize channels from cell data.
    
    Parameters
    ----------
    cell_data : dict
        Dictionary with keys: 'BF', 'Actin', 'Microtubules', 'Nuclei'
    
    Returns
    -------
    dict
        Dictionary with normalized channels:
        {
            'phase': np.ndarray,  # BF alias
            'actin': np.ndarray,
            'mt': np.ndarray,     # Microtubules alias
            'nuc': np.ndarray     # Nuclei alias
        }
    """
    # Map channels
    channels = {}
    
    # BF → phase
    if 'BF' in cell_data:
        channels['phase'] = normalize_to_float(cell_data['BF'])
    
    # Actin → actin
    if 'Actin' in cell_data:
        channels['actin'] = normalize_to_float(cell_data['Actin'])
    
    # Microtubules → mt
    if 'Microtubules' in cell_data:
        channels['mt'] = normalize_to_float(cell_data['Microtubules'])
    
    # Nuclei → nuc
    if 'Nuclei' in cell_data:
        channels['nuc'] = normalize_to_float(cell_data['Nuclei'])
    
    return channels


def preprocess_for_generation(
    cell_data: Dict[str, np.ndarray],
    target_size: Optional[tuple] = None
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline: extract channels, normalize, resize.
    
    Parameters
    ----------
    cell_data : dict
        Raw cell data with channels: 'BF', 'Actin', 'Microtubules', 'Nuclei'
    target_size : tuple, optional
        Target size (height, width). Defaults to config.IMAGE_SIZE.
    
    Returns
    -------
    dict
        Preprocessed channels: 'phase', 'actin', 'mt', 'nuc'
        All normalized to [0, 1] float and resized to target_size.
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE
    
    # Extract and normalize
    channels = extract_channels(cell_data)
    
    # Resize all channels to target size
    for key in channels:
        channels[key] = resize_to_target(channels[key], target_size)
    
    return channels

