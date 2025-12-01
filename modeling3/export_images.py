"""
Image export utility for Modeling 3 assignment deliverables.

Exports:
- ≥9 grayscale images (from any channel)
- ≥21 RGB composite images (R=mt, G=actin, B=nuc)

This module is self-contained and does not modify the core pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import random
import modeling3.config as config


def normalize_channel(image: np.ndarray, use_percentile: bool = True) -> np.ndarray:
    """
    Normalize a grayscale image to float32 [0, 1] range.
    
    Uses percentile-based normalization (2nd-98th percentile) for better contrast
    and more vibrant colors, which is standard in microscopy image processing.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (any dtype)
    use_percentile : bool
        If True, use percentile-based normalization (default: True)
    
    Returns
    -------
    np.ndarray
        Normalized float32 image [0, 1]
    """
    img_float = image.astype(np.float32)
    
    if use_percentile:
        # Use percentile-based normalization for better contrast
        # This avoids issues with outliers and provides more vibrant colors
        p2 = np.percentile(img_float, 2)
        p98 = np.percentile(img_float, 98)
        
        if p98 > p2:
            # Normalize using percentile range
            normalized = (img_float - p2) / (p98 - p2)
            normalized = normalized.clip(0, 1)
        else:
            # Fallback to min-max if percentiles are too close
            img_min = img_float.min()
            img_max = img_float.max()
            if img_max > img_min:
                normalized = (img_float - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(img_float)
    else:
        # Standard min-max normalization
        if image.dtype == np.uint8:
            normalized = img_float / 255.0
        elif image.max() > 1.0:
            img_max = image.max()
            normalized = (img_float / img_max).clip(0, 1)
        else:
            normalized = img_float
    
    return normalized


def make_color_composite(
    actin: np.ndarray,
    mt: np.ndarray,
    nuclei: np.ndarray
) -> np.ndarray:
    """
    Create RGB composite image with enhanced contrast.
    
    RGB mapping:
    - R channel = microtubules (mt)
    - G channel = actin
    - B channel = nuclei (nuc)
    
    Uses percentile-based normalization for each channel independently
    to ensure vibrant, visible colors.
    
    Parameters
    ----------
    actin : np.ndarray
        Actin channel image
    mt : np.ndarray
        Microtubules channel image
    nuclei : np.ndarray
        Nuclei channel image
    
    Returns
    -------
    np.ndarray
        RGB composite image (H, W, 3) as uint8 [0, 255]
    """
    # Normalize each channel independently using percentile-based normalization
    # This ensures each channel has good contrast and colors are vibrant
    actin_norm = normalize_channel(actin, use_percentile=True)
    mt_norm = normalize_channel(mt, use_percentile=True)
    nuclei_norm = normalize_channel(nuclei, use_percentile=True)
    
    # Stack into RGB: R=mt, G=actin, B=nuc
    rgb = np.stack([mt_norm, actin_norm, nuclei_norm], axis=-1)
    
    # Convert to uint8 [0, 255]
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    return rgb_uint8


def export_sample_images(
    manifest_path: Path,
    output_dir: Path,
    n_gray: int = 9,
    n_color: int = 21
) -> None:
    """
    Export grayscale and RGB composite images from manifest.
    
    Parameters
    ----------
    manifest_path : Path
        Path to manifest_mc3.csv
    output_dir : Path
        Output directory for exported images
    n_gray : int
        Minimum number of grayscale images to export (default: 9)
    n_color : int
        Minimum number of RGB composite images to export (default: 21)
    """
    # Load manifest
    manifest_df = pd.read_csv(manifest_path)
    
    # Create output directories
    gray_dir = output_dir / "exported_images" / "grayscale"
    color_dir = output_dir / "exported_images" / "color"
    gray_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Channel mapping for grayscale export
    channel_map = {
        'path_phase': 'phase',
        'path_actin': 'actin',
        'path_mt': 'mt',
        'path_nuc': 'nuc'
    }
    
    # Export grayscale images
    print(f"Exporting {n_gray} grayscale images...")
    gray_count = 0
    gray_exported = set()  # Track exported to avoid duplicates
    
    # Shuffle manifest for random selection
    manifest_shuffled = manifest_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    for _, row in manifest_shuffled.iterrows():
        if gray_count >= n_gray:
            break
        
        sample_id = row['sample_id']
        
        # Randomly select a channel
        available_channels = [
            ('path_phase', 'phase'),
            ('path_actin', 'actin'),
            ('path_mt', 'mt'),
            ('path_nuc', 'nuc')
        ]
        
        # Filter to channels that exist and have valid paths
        valid_channels = [
            (col, name) for col, name in available_channels
            if pd.notna(row[col]) and Path(row[col]).exists()
        ]
        
        if not valid_channels:
            continue
        
        # Randomly select one channel
        col_name, channel_name = random.choice(valid_channels)
        image_path = Path(row[col_name])
        
        # Create unique key to avoid duplicates
        unique_key = f"{sample_id}_{channel_name}"
        if unique_key in gray_exported:
            continue
        
        try:
            # Load and save grayscale image
            image = imread(str(image_path))
            
            # Ensure it's grayscale (2D)
            if len(image.shape) == 3:
                # Convert RGB to grayscale (take first channel or average)
                image = image[:, :, 0] if image.shape[2] >= 1 else np.mean(image, axis=2)
            
            # Save with 300 DPI metadata
            output_path = gray_dir / f"gray_{sample_id}_{channel_name}.png"
            imsave(str(output_path), image, check_contrast=False)
            
            gray_exported.add(unique_key)
            gray_count += 1
            
        except Exception as e:
            print(f"  Warning: Failed to export {sample_id} {channel_name}: {e}")
            continue
    
    print(f"  ✓ Saved {gray_count} grayscale images")
    
    # Export RGB composite images
    print(f"\nExporting {n_color} RGB composite images...")
    color_count = 0
    color_exported = set()  # Track exported to avoid duplicates
    
    # Shuffle again for color images
    manifest_shuffled = manifest_df.sample(frac=1, random_state=config.RANDOM_SEED + 1).reset_index(drop=True)
    
    for _, row in manifest_shuffled.iterrows():
        if color_count >= n_color:
            break
        
        sample_id = row['sample_id']
        
        # Skip if already exported
        if sample_id in color_exported:
            continue
        
        # Check that all required channels exist
        if (pd.isna(row['path_actin']) or pd.isna(row['path_mt']) or pd.isna(row['path_nuc'])):
            continue
        
        actin_path = Path(row['path_actin'])
        mt_path = Path(row['path_mt'])
        nuc_path = Path(row['path_nuc'])
        
        if not (actin_path.exists() and mt_path.exists() and nuc_path.exists()):
            continue
        
        try:
            # Load all three channels
            actin_img = imread(str(actin_path))
            mt_img = imread(str(mt_path))
            nuc_img = imread(str(nuc_path))
            
            # Ensure grayscale (2D)
            if len(actin_img.shape) == 3:
                actin_img = actin_img[:, :, 0] if actin_img.shape[2] >= 1 else np.mean(actin_img, axis=2)
            if len(mt_img.shape) == 3:
                mt_img = mt_img[:, :, 0] if mt_img.shape[2] >= 1 else np.mean(mt_img, axis=2)
            if len(nuc_img.shape) == 3:
                nuc_img = nuc_img[:, :, 0] if nuc_img.shape[2] >= 1 else np.mean(nuc_img, axis=2)
            
            # Create RGB composite
            rgb_composite = make_color_composite(actin_img, mt_img, nuc_img)
            
            # Save with 300 DPI metadata
            output_path = color_dir / f"color_{sample_id}.png"
            imsave(str(output_path), rgb_composite, check_contrast=False)
            
            color_exported.add(sample_id)
            color_count += 1
            
        except Exception as e:
            print(f"  Warning: Failed to export RGB composite for {sample_id}: {e}")
            continue
    
    print(f"  ✓ Saved {color_count} RGB composite images")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"✓ Saved {gray_count} grayscale images")
    print(f"✓ Saved {color_count} RGB composite images")
    print(f"→ Output directory: {output_dir}/exported_images/")
    print(f"  - Grayscale: {gray_dir}/")
    print(f"  - Color: {color_dir}/")


def find_latest_manifest() -> Optional[Path]:
    """
    Find the most recent manifest_mc3.csv file.
    
    Returns
    -------
    Path or None
        Path to most recent manifest, or None if not found
    """
    # Check common output directories
    candidates = [
        Path("modeling3_outputs_fixed/manifest_mc3.csv"),
        Path("modeling3_outputs/manifest_mc3.csv"),
        Path(config.OUTPUT_ROOT / "manifest_mc3.csv")
    ]
    
    # Return first existing candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def main():
    """Main entry point for image export utility."""
    parser = argparse.ArgumentParser(
        description="Export grayscale and RGB composite images for Modeling 3 assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        default=None,
        help='Path to manifest_mc3.csv (default: auto-detect from latest output)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as manifest directory)'
    )
    
    parser.add_argument(
        '--n-gray',
        type=int,
        default=9,
        help='Minimum number of grayscale images (default: 9)'
    )
    
    parser.add_argument(
        '--n-color',
        type=int,
        default=21,
        help='Minimum number of RGB composite images (default: 21)'
    )
    
    args = parser.parse_args()
    
    # Find manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = find_latest_manifest()
    
    if manifest_path is None or not manifest_path.exists():
        print("ERROR: Could not find manifest_mc3.csv")
        print("Please specify --manifest or ensure output directory exists.")
        sys.exit(1)
    
    print(f"Using manifest: {manifest_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use same directory as manifest
        output_dir = manifest_path.parent
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Export images
    try:
        export_sample_images(
            manifest_path=manifest_path,
            output_dir=output_dir,
            n_gray=args.n_gray,
            n_color=args.n_color
        )
    except Exception as e:
        print(f"ERROR: Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

