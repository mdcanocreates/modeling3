"""
Algorithm 1: Classical Biological Augmentation

Generates plausible variants using:
- Rotation (±15°)
- Flip (LR/UD)
- Elastic deformation
- Gaussian noise
- Brightness/contrast jitter
- Zoom/crop

All channels are synchronized (same transformation applied to all).
"""

import numpy as np
from typing import Dict, List
from pathlib import Path
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from scipy.ndimage import map_coordinates
import modeling3.config as config
from modeling3.manifest import ImageRecord
from modeling3.preprocessing import normalize_to_float


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees."""
    return rotate(image, angle, mode='reflect', preserve_range=True).astype(image.dtype)


def apply_flip(image: np.ndarray, flip_lr: bool = False, flip_ud: bool = False) -> np.ndarray:
    """Flip image left-right and/or up-down."""
    if flip_lr:
        image = np.fliplr(image)
    if flip_ud:
        image = np.flipud(image)
    return image


def apply_elastic_deformation(
    image: np.ndarray,
    alpha: float = 50,
    sigma: float = 5
) -> np.ndarray:
    """
    Apply elastic deformation to image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    alpha : float
        Deformation strength
    sigma : float
        Smoothness of deformation
    """
    h, w = image.shape[:2]
    
    # Generate random displacement fields
    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha
    
    # Smooth the displacement fields
    from scipy.ndimage import gaussian_filter
    dx = gaussian_filter(dx, sigma=sigma)
    dy = gaussian_filter(dy, sigma=sigma)
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply displacement
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Interpolate
    if len(image.shape) == 2:
        deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(h, w)
    else:
        deformed = np.zeros_like(image)
        for c in range(image.shape[2]):
            deformed[:, :, c] = map_coordinates(
                image[:, :, c], indices, order=1, mode='reflect'
            ).reshape(h, w)
    
    return deformed.astype(image.dtype)


def apply_noise(image: np.ndarray, std: float = 0.02) -> np.ndarray:
    """Add Gaussian noise to image."""
    # Ensure image is in [0, 1] range
    img_norm = normalize_to_float(image)
    
    # Add noise
    noisy = random_noise(img_norm, mode='gaussian', var=std**2)
    
    # Clip to [0, 1] and convert back to original dtype
    noisy = np.clip(noisy, 0, 1)
    
    if image.dtype == np.uint8:
        return (noisy * 255).astype(np.uint8)
    else:
        return noisy.astype(image.dtype)


def apply_brightness_contrast(
    image: np.ndarray,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0
) -> np.ndarray:
    """
    Apply brightness and contrast adjustments.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (normalized to [0, 1])
    brightness_factor : float
        Brightness multiplier
    contrast_factor : float
        Contrast multiplier (around 0.5)
    """
    # Ensure normalized
    img_norm = normalize_to_float(image)
    
    # Apply brightness
    adjusted = img_norm * brightness_factor
    
    # Apply contrast (around 0.5)
    adjusted = (adjusted - 0.5) * contrast_factor + 0.5
    
    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0, 1)
    
    # Convert back to original dtype
    if image.dtype == np.uint8:
        return (adjusted * 255).astype(np.uint8)
    else:
        return adjusted.astype(image.dtype)


def apply_zoom_crop(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Apply zoom (crop and resize).
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    zoom_factor : float
        Zoom factor (>1 = zoom in, <1 = zoom out)
    """
    h, w = image.shape[:2]
    
    # Calculate crop size
    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)
    
    # Center crop
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    
    cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
    
    # Resize back to original size
    from skimage.transform import resize
    resized = resize(cropped, (h, w), preserve_range=True, anti_aliasing=True)
    
    return resized.astype(image.dtype)


def generate_classical_augs(
    cell: Dict[str, np.ndarray],
    n_per_cell: int,
    output_dir: Path,
    parent_id: str,
    config_obj=None
) -> List[ImageRecord]:
    """
    Generate classical augmentation variants for a cell.
    
    Parameters
    ----------
    cell : dict
        Dictionary with keys: 'phase', 'actin', 'mt', 'nuc'
        All images should be same size and normalized [0, 1]
    n_per_cell : int
        Number of variants to generate
    output_dir : Path
        Directory to save generated images
    parent_id : str
        Parent cell ID (CellA, CellB, CellC)
    config_obj : module, optional
        Config module (defaults to modeling3.config)
    
    Returns
    -------
    list
        List of ImageRecord objects
    """
    if config_obj is None:
        config_obj = config
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(config_obj.RANDOM_SEED)
    
    records = []
    channel_keys = ['phase', 'actin', 'mt', 'nuc']
    
    # Verify all channels exist and have same size
    shapes = {k: cell[k].shape for k in channel_keys if k in cell}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Channels have mismatched shapes: {shapes}")
    
    h, w = list(shapes.values())[0]
    
    for i in range(n_per_cell):
        sample_id = f"{parent_id}_alg1_{i+1:03d}"
        
        # Generate random transformation parameters
        angle = np.random.uniform(*config_obj.ALG1_ROTATION_RANGE)
        flip_lr = np.random.choice([True, False])
        flip_ud = np.random.choice([True, False])
        apply_elastic = np.random.choice([True, False], p=[0.7, 0.3])  # 70% chance
        noise_std = np.random.uniform(0, config_obj.ALG1_NOISE_STD)
        brightness = np.random.uniform(*config_obj.ALG1_BRIGHTNESS_RANGE)
        contrast = np.random.uniform(*config_obj.ALG1_CONTRAST_RANGE)
        zoom = np.random.uniform(*config_obj.ALG1_ZOOM_RANGE)
        
        # Apply same transformation to all channels
        transformed_channels = {}
        path_phase = None
        path_actin = None
        path_mt = None
        path_nuc = None
        
        for ch_key in channel_keys:
            if ch_key not in cell:
                continue
            
            img = cell[ch_key].copy()
            
            # Rotation
            img = apply_rotation(img, angle)
            
            # Flip
            img = apply_flip(img, flip_lr, flip_ud)
            
            # Elastic deformation (if selected)
            if apply_elastic:
                img = apply_elastic_deformation(
                    img,
                    alpha=config_obj.ALG1_ELASTIC_ALPHA,
                    sigma=config_obj.ALG1_ELASTIC_SIGMA
                )
            
            # Noise
            if noise_std > 0:
                img = apply_noise(img, noise_std)
            
            # Brightness/contrast
            img = apply_brightness_contrast(img, brightness, contrast)
            
            # Zoom/crop
            img = apply_zoom_crop(img, zoom)
            
            # Ensure still in [0, 1] range
            img = np.clip(img, 0, 1)
            
            # Save image
            channel_name = config_obj.CHANNEL_ALIASES.get(
                ch_key.replace('phase', 'BF').replace('mt', 'Microtubules').replace('nuc', 'Nuclei'),
                ch_key
            )
            filename = f"{sample_id}_{channel_name}.png"
            filepath = output_dir / filename
            
            # Convert to uint8 for saving
            img_uint8 = (img * 255).astype(np.uint8)
            from skimage.io import imsave
            imsave(str(filepath), img_uint8, check_contrast=False)
            
            # Store path based on channel
            if ch_key == 'phase':
                path_phase = str(filepath)
            elif ch_key == 'actin':
                path_actin = str(filepath)
            elif ch_key == 'mt':
                path_mt = str(filepath)
            elif ch_key == 'nuc':
                path_nuc = str(filepath)
            
            transformed_channels[ch_key] = img
        
        # Create record
        record = ImageRecord(
            sample_id=sample_id,
            parent_id=parent_id,
            algorithm="alg1",
            path_phase=path_phase,
            path_actin=path_actin,
            path_mt=path_mt,
            path_nuc=path_nuc
        )
        records.append(record)
    
    return records

