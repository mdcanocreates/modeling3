"""
Algorithm 2: Structural/Texture Variant Generation using Thin-Plate Spline (TPS) Warping

Generates plausible structural variants by applying TPS warping to all channels.
This creates biologically plausible deformations distinct from Algorithm 1.
"""

import numpy as np
from typing import Dict, List
from pathlib import Path
from skimage.transform import PiecewiseAffineTransform, warp
import modeling3.config as config
from modeling3.manifest import ImageRecord
from modeling3.preprocessing import normalize_to_float


def generate_tps_control_points(
    image_shape: tuple,
    n_points: int = 10,
    displacement_max: float = 20
) -> tuple:
    """
    Generate control points for Thin-Plate Spline warping.
    
    Parameters
    ----------
    image_shape : tuple
        (height, width) of image
    n_points : int
        Number of control points
    displacement_max : float
        Maximum displacement in pixels
    
    Returns
    -------
    tuple
        (src_points, dst_points) where each is (n_points, 2) array
    """
    h, w = image_shape[:2]
    
    # Generate source control points (regular grid)
    # Place points with some margin from edges
    margin = 20
    x_coords = np.linspace(margin, w - margin, int(np.sqrt(n_points)))
    y_coords = np.linspace(margin, h - margin, int(np.sqrt(n_points)))
    
    # Create grid
    xx, yy = np.meshgrid(x_coords, y_coords)
    src_points = np.column_stack([yy.ravel(), xx.ravel()])
    
    # Limit to n_points
    if len(src_points) > n_points:
        indices = np.random.choice(len(src_points), n_points, replace=False)
        src_points = src_points[indices]
    elif len(src_points) < n_points:
        # Add random points if needed
        n_add = n_points - len(src_points)
        x_add = np.random.uniform(margin, w - margin, n_add)
        y_add = np.random.uniform(margin, h - margin, n_add)
        src_points = np.vstack([src_points, np.column_stack([y_add, x_add])])
    
    # Generate random displacements
    displacements = np.random.uniform(
        -displacement_max,
        displacement_max,
        size=(len(src_points), 2)
    )
    
    # Destination points = source + displacement
    dst_points = src_points + displacements
    
    # Clip to image bounds
    dst_points[:, 0] = np.clip(dst_points[:, 0], 0, h - 1)
    dst_points[:, 1] = np.clip(dst_points[:, 1], 0, w - 1)
    
    return src_points, dst_points


def apply_tps_warp(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> np.ndarray:
    """
    Apply Thin-Plate Spline warping to image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    src_points : np.ndarray
        Source control points (n_points, 2)
    dst_points : np.ndarray
        Destination control points (n_points, 2)
    
    Returns
    -------
    np.ndarray
        Warped image
    """
    # Create TPS transform
    tform = PiecewiseAffineTransform()
    tform.estimate(src_points, dst_points)
    
    # Apply warp
    warped = warp(
        image,
        tform,
        output_shape=image.shape[:2],
        mode='reflect',
        preserve_range=True
    )
    
    return warped.astype(image.dtype)


def generate_structural_variants(
    cell: Dict[str, np.ndarray],
    n_per_cell: int,
    output_dir: Path,
    parent_id: str,
    config_obj=None
) -> List[ImageRecord]:
    """
    Generate structural variant images using TPS warping.
    
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
        sample_id = f"{parent_id}_alg2_{i+1:03d}"
        
        # Generate TPS control points (same for all channels)
        src_points, dst_points = generate_tps_control_points(
            (h, w),
            n_points=config_obj.ALG2_N_CONTROL_POINTS,
            displacement_max=config_obj.ALG2_DISPLACEMENT_MAX
        )
        
        # Apply same TPS warp to all channels
        path_phase = None
        path_actin = None
        path_mt = None
        path_nuc = None
        
        for ch_key in channel_keys:
            if ch_key not in cell:
                continue
            
            img = cell[ch_key].copy()
            
            # Apply TPS warp
            img = apply_tps_warp(img, src_points, dst_points)
            
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
        
        # Create record
        record = ImageRecord(
            sample_id=sample_id,
            parent_id=parent_id,
            algorithm="alg2",
            path_phase=path_phase,
            path_actin=path_actin,
            path_mt=path_mt,
            path_nuc=path_nuc
        )
        records.append(record)
    
    return records

