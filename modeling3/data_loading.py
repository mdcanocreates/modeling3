"""
Data loading utilities for original endothelial cell images.

Reuses Modeling 2's io_utils for loading images.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from image_analysis.io_utils import load_cell_images, normalize_image_sizes, CELL_ID_TO_PREFIX
import modeling3.config as config


def load_original_cells(
    data_root: Optional[Path] = None,
    cell_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all original cell images with 4 channels each.
    
    Parameters
    ----------
    data_root : Path, optional
        Root directory containing cell subdirectories. Defaults to config.DATA_ROOT.
    cell_ids : list, optional
        List of cell IDs to load. Defaults to ['CellA', 'CellB', 'CellC'].
    
    Returns
    -------
    dict
        Dictionary mapping cell_id to channel images:
        {
            'CellA': {
                'BF': np.ndarray,      # Phase channel
                'Actin': np.ndarray,
                'Microtubules': np.ndarray,
                'Nuclei': np.ndarray
            },
            ...
        }
    
    Raises
    ------
    FileNotFoundError
        If any cell directory or image file is missing.
    """
    if data_root is None:
        data_root = config.DATA_ROOT
    
    if cell_ids is None:
        cell_ids = ['CellA', 'CellB', 'CellC']
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    # Channel mapping: use existing channel names from Modeling 2
    channel_map = {
        'bf': 'BF',
        'actin': 'Actin',
        'microtubules': 'Microtubules',
        'nuclei': 'Nuclei'
    }
    
    all_cells = {}
    
    for cell_id in cell_ids:
        try:
            # Load images using Modeling 2's loader
            images = load_cell_images(cell_id, data_root, channel_map)
            
            # Normalize sizes
            normalize_image_sizes(images)
            
            # Map to our channel names
            cell_data = {
                'BF': images['bf'],
                'Actin': images['actin'],
                'Microtubules': images['microtubules'],
                'Nuclei': images['nuclei']
            }
            
            all_cells[cell_id] = cell_data
            
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load {cell_id}: {e}"
            )
    
    return all_cells


def verify_cell_data(cells: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, bool]:
    """
    Verify that all cells have all 4 channels with reasonable dimensions.
    
    Parameters
    ----------
    cells : dict
        Dictionary of cell data from load_original_cells()
    
    Returns
    -------
    dict
        Dictionary mapping cell_id to verification status (True = valid)
    """
    required_channels = config.CHANNEL_ORDER
    results = {}
    
    for cell_id, cell_data in cells.items():
        valid = True
        
        # Check all channels exist
        for channel in required_channels:
            if channel not in cell_data:
                print(f"  ✗ {cell_id}: Missing channel {channel}")
                valid = False
                continue
            
            img = cell_data[channel]
            
            # Check shape (should be 2D)
            if len(img.shape) != 2:
                print(f"  ✗ {cell_id}: Channel {channel} has wrong shape {img.shape} (expected 2D)")
                valid = False
                continue
            
            # Check reasonable size
            h, w = img.shape
            if h < 50 or w < 50 or h > 5000 or w > 5000:
                print(f"  ✗ {cell_id}: Channel {channel} has unreasonable size {img.shape}")
                valid = False
                continue
        
        # Check all channels have same size
        if valid:
            sizes = {ch: cell_data[ch].shape for ch in required_channels}
            if len(set(sizes.values())) > 1:
                print(f"  ✗ {cell_id}: Channels have mismatched sizes: {sizes}")
                valid = False
        
        results[cell_id] = valid
        if valid:
            print(f"  ✓ {cell_id}: All 4 channels loaded successfully")
    
    return results

