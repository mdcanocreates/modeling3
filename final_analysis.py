"""
Final Production Pipeline for Cell Image Analysis

This is the clean, final version of the analysis pipeline that:
1. Uses SAM-first segmentation for cells (primary method)
2. Uses classical methods for nuclei segmentation inside SAM mask
3. Computes 6-7 robust metrics
4. Performs similarity analysis
5. Generates QC images

This script represents the "production" version - all experimental branches
and refinement attempts have been removed.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import os

from image_analysis.io_utils import load_cell_images, normalize_image_sizes, CELL_ID_TO_PREFIX
from image_analysis.segmentation import (
    segment_nuclei_robust, create_cytoplasm_mask
)
from image_analysis.metrics import compute_all_metrics
from image_analysis.similarity import analyze_similarity
from image_analysis.plotting import (
    plot_combo_with_masks, plot_actin_with_cell_mask, plot_nuclei_with_nuclear_mask
)
from image_analysis.sam_wrapper import sam_segment_cell
from image_analysis.gemini_qc import evaluate_segmentation_with_gemini


def process_cell_final(
    cell_id: str,
    data_root: Path,
    output_root: Path,
    pixel_size_um: float = 1.0
) -> Dict[str, float]:
    """
    Process a single cell using the final production pipeline.
    
    Steps:
    1. Load and normalize images
    2. Segment cell using SAM (primary method)
    3. Segment nuclei using classical methods (inside SAM mask)
    4. Compute metrics
    5. Generate QC images
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell (e.g., "CellA")
    data_root : Path
        Root directory containing cell subdirectories
    output_root : Path
        Output directory for QC images
    pixel_size_um : float
        Pixel size in microns per pixel
    
    Returns
    -------
    dict
        Dictionary of computed metrics for this cell
    """
    print(f"Processing {cell_id}...")
    
    # Load images
    images = load_cell_images(cell_id, data_root)
    # Normalize image sizes (resize if needed)
    normalize_image_sizes(images)
    
    # Extract individual channels
    actin_image = images['actin']
    microtubule_image = images['microtubules']
    nuclei_image = images['nuclei']
    combo_image = images['combo']
    
    # Get image paths for SAM
    filename_prefix = CELL_ID_TO_PREFIX.get(cell_id, cell_id)
    actin_path = data_root / cell_id / f"{filename_prefix}_Actin.jpg"
    if not actin_path.exists():
        actin_path = data_root / cell_id / f"{cell_id}_Actin.jpg"
    
    # Try Combo image if Actin not found
    if not actin_path.exists():
        combo_path = data_root / cell_id / f"{filename_prefix}_Combo.jpg"
        if not combo_path.exists():
            combo_path = data_root / cell_id / f"{cell_id}_Combo.jpg"
        if combo_path.exists():
            actin_path = combo_path
    
    # Get nuclei image path
    nuclei_path = data_root / cell_id / f"{filename_prefix}_Nuclei.jpg"
    if not nuclei_path.exists():
        nuclei_path = data_root / cell_id / f"{cell_id}_Nuclei.jpg"
    
    # Step 1: Segment cell using SAM (PRIMARY METHOD)
    # Check if manual mask exists (from UI refinement)
    manual_mask_path = output_root / f"{cell_id}_cell_mask_manual.npy"
    
    if manual_mask_path.exists():
        print(f"  Loading manually refined mask from {manual_mask_path.name}...")
        cell_mask = np.load(manual_mask_path)
        
        # Resize if needed
        if cell_mask.shape != actin_image.shape:
            from skimage.transform import resize
            cell_mask = resize(cell_mask, actin_image.shape, order=0, preserve_range=True).astype(bool)
        
        cell_area_pixels = np.sum(cell_mask)
        print(f"    Cell area: {cell_area_pixels:.0f} pixels (from manual mask)")
    else:
        print(f"  Segmenting cell using SAM...")
        try:
            cell_mask = sam_segment_cell(
                image_path=str(actin_path),
                nuclei_image_path=str(nuclei_path) if nuclei_path.exists() else None
            )
            
            # Resize SAM mask to match normalized image dimensions
            if cell_mask.shape != actin_image.shape:
                from skimage.transform import resize
                cell_mask = resize(cell_mask, actin_image.shape, order=0, preserve_range=True).astype(bool)
            
            cell_area_pixels = np.sum(cell_mask)
            print(f"    Cell area: {cell_area_pixels:.0f} pixels")
        except Exception as e:
            print(f"    ERROR: SAM segmentation failed: {e}")
            raise
    
    # Step 2: Segment nuclei from Nuclei channel using classical method (inside SAM mask)
    print(f"  Segmenting nuclei (classical method, inside SAM mask)...")
    nucleus_mask, nuclei_props = segment_nuclei_robust(
        nuclei_image, 
        cell_mask, 
        cell_area_pixels
    )
    
    # Log nuclear segmentation results
    print(f"    Nuclear count: {nuclei_props['count']}")
    print(f"    Nuclear areas: {nuclei_props['areas']}")
    if nuclei_props['warnings']:
        for warning in nuclei_props['warnings']:
            print(f"    WARNING: {warning}")
    
    # Create cytoplasm mask
    cytoplasm_mask = create_cytoplasm_mask(cell_mask, nucleus_mask)
    
    # Step 3: Compute metrics
    print(f"  Computing metrics...")
    metrics = compute_all_metrics(
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        cytoplasm_mask=cytoplasm_mask,
        actin_image=actin_image,
        microtubule_image=microtubule_image,
        pixel_size_um=pixel_size_um
    )
    
    # Add cell_id to metrics
    metrics['cell_id'] = cell_id
    
    # Step 4: Generate QC images
    print(f"  Generating QC images...")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Actin image with cell mask
    actin_path_qc = output_root / f"{cell_id}_actin_with_cell_mask.png"
    plot_actin_with_cell_mask(
        actin_image=actin_image,
        cell_mask=cell_mask,
        output_path=str(actin_path_qc)
    )
    
    # Nuclei image with nuclear mask and cell outline
    nuclei_path_qc = output_root / f"{cell_id}_nuclei_with_nuclear_mask.png"
    plot_nuclei_with_nuclear_mask(
        nuclei_image=nuclei_image,
        nucleus_mask=nucleus_mask,
        cell_mask=cell_mask,
        output_path=str(nuclei_path_qc)
    )
    
    # Combo image with both masks (optional, for comprehensive view)
    combo_path_qc = output_root / f"{cell_id}_combo_with_masks.png"
    plot_combo_with_masks(
        combo_image=combo_image,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        output_path=str(combo_path_qc)
    )
    
    # Step 5: Optional Gemini QC evaluation
    print(f"  Running Gemini QC evaluation (optional)...")
    try:
        # Get raw nuclei image path
        filename_prefix = CELL_ID_TO_PREFIX.get(cell_id, cell_id)
        raw_nuclei_path = data_root / cell_id / f"{filename_prefix}_Nuclei.jpg"
        if not raw_nuclei_path.exists():
            raw_nuclei_path = data_root / cell_id / f"{cell_id}_Nuclei.jpg"
        
        if raw_nuclei_path.exists():
            qc_result = evaluate_segmentation_with_gemini(
                cell_id=cell_id,
                raw_image_path=str(raw_nuclei_path),
                overlay_image_path=str(nuclei_path_qc),
                channel="nuclei"
            )
            
            # Log QC results
            cell_score = qc_result.get('cell_mask_score')
            nucleus_score = qc_result.get('nucleus_mask_score')
            
            if cell_score is not None:
                print(f"    Cell mask score: {cell_score:.2f}")
            if nucleus_score is not None:
                print(f"    Nuclear mask score: {nucleus_score:.2f}")
            
            # Store QC result in metrics for later saving
            metrics['_gemini_qc'] = qc_result
        else:
            print(f"    Warning: Raw nuclei image not found, skipping Gemini QC")
    except Exception as e:
        print(f"    Warning: Gemini QC failed: {e}")
        metrics['_gemini_qc'] = None
    
    print(f"  Completed {cell_id}")
    
    return metrics


def main():
    """Main entry point for final production pipeline."""
    parser = argparse.ArgumentParser(
        description="Final production pipeline for cell image analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the clean, final production version of the analysis pipeline.
It uses SAM-first segmentation for cells and classical methods for nuclei.
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='img_model',
        help='Root directory containing cell subdirectories (default: img_model)'
    )
    
    parser.add_argument(
        '--output-root',
        type=str,
        default='final_outputs',
        help='Output directory for results (default: final_outputs)'
    )
    
    parser.add_argument(
        '--cell-ids',
        nargs='+',
        default=['CellA', 'CellB', 'CellC'],
        help='List of cell IDs to process (default: CellA CellB CellC)'
    )
    
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=1.0,
        help='Pixel size in microns per pixel (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FINAL PRODUCTION PIPELINE - Cell Image Analysis")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Cell IDs: {', '.join(args.cell_ids)}")
    print()
    
    # Check SAM availability
    try:
        from image_analysis.sam_wrapper import SAM_AVAILABLE
        if not SAM_AVAILABLE:
            print("ERROR: SAM is not available. Please install segment-anything package.")
            sys.exit(1)
        
        checkpoint_path = os.getenv('SAM_CHECKPOINT_PATH')
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print("ERROR: SAM_CHECKPOINT_PATH not set or checkpoint not found.")
            print("Please set SAM_CHECKPOINT_PATH environment variable.")
            sys.exit(1)
        
        print(f"SAM checkpoint: {checkpoint_path}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to check SAM availability: {e}")
        sys.exit(1)
    
    # Process each cell
    all_metrics = []
    
    for cell_id in args.cell_ids:
        try:
            metrics = process_cell_final(
                cell_id=cell_id,
                data_root=data_root,
                output_root=output_root,
                pixel_size_um=args.pixel_size
            )
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"ERROR: Failed to process {cell_id}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Save metrics to CSV (exclude Gemini QC from CSV)
    if all_metrics:
        # Separate Gemini QC results
        all_gemini_qc = []
        metrics_clean = []
        
        for m in all_metrics:
            qc = m.pop('_gemini_qc', None)
            if qc:
                all_gemini_qc.append(qc)
            metrics_clean.append(m)
        
        df = pd.DataFrame(metrics_clean)
        metrics_csv = output_root / "final_metrics.csv"
        df.to_csv(metrics_csv, index=False)
        print(f"\nMetrics saved to {metrics_csv}")
        
        # Save Gemini QC results if available
        if all_gemini_qc:
            import json
            qc_json = output_root / "gemini_qc_results.json"
            with open(qc_json, 'w') as f:
                json.dump(all_gemini_qc, f, indent=2)
            print(f"Gemini QC results saved to {qc_json}")
            
            # Create summary CSV
            qc_summary = []
            for qc in all_gemini_qc:
                qc_summary.append({
                    'cell_id': qc.get('cell_id', ''),
                    'channel': qc.get('channel', ''),
                    'cell_mask_score': qc.get('cell_mask_score'),
                    'nucleus_mask_score': qc.get('nucleus_mask_score')
                })
            qc_df = pd.DataFrame(qc_summary)
            qc_csv = output_root / "gemini_qc_summary.csv"
            qc_df.to_csv(qc_csv, index=False)
            print(f"Gemini QC summary saved to {qc_csv}")
    
    # Perform similarity analysis
    if len(metrics_clean) >= 2:
        print("\n" + "="*60)
        print("SIMILARITY ANALYSIS")
        print("="*60)
        
        # Convert list of dicts to DataFrame (use cleaned metrics without Gemini QC)
        metrics_df = pd.DataFrame(metrics_clean)
        
        # Define metric columns (exclude non-metric columns)
        # We'll use the core 6-7 metrics: area, circularity, aspect_ratio, 
        # actin_mean_intensity, actin_anisotropy, mtub_mean_intensity
        # Optionally include nc_ratio if nuclear segmentation is reliable
        exclude_cols = {'cell_id'}
        metric_columns = [col for col in metrics_df.columns if col not in exclude_cols]
        
        # Get cell IDs
        if 'cell_id' in metrics_df.columns:
            cell_ids = metrics_df['cell_id'].tolist()
        else:
            cell_ids = args.cell_ids
        
        # Set cell_id as index for similarity analysis
        if 'cell_id' in metrics_df.columns:
            metrics_df = metrics_df.set_index('cell_id')
        
        similarity_results = analyze_similarity(
            metrics_df=metrics_df,
            metric_columns=metric_columns,
            cell_ids=cell_ids
        )
        
        # Print results
        print("\n" + "="*60)
        print("CELL METRICS")
        print("="*60)
        for cell_id in cell_ids:
            if cell_id in metrics_df.index:
                row = metrics_df.loc[cell_id]
                print(f"\n{cell_id}:")
                for col in metric_columns:
                    if col in row:
                        print(f"  {col}: {row[col]:.4f}")
        
        print("\n" + "="*60)
        print("NORMALIZED METRIC VECTORS")
        print("="*60)
        norm_df = similarity_results['normalized_metrics']
        for cell_id in cell_ids:
            if cell_id in norm_df.index:
                row = norm_df.loc[cell_id, metric_columns]
                print(f"\nm̃_{cell_id} = {row.values}")
        
        print("\n" + "="*60)
        print("PAIRWISE DISTANCES")
        print("="*60)
        distances = similarity_results['distances']
        n = len(cell_ids)
        for i in range(n):
            for j in range(i + 1, n):
                cell1 = cell_ids[i]
                cell2 = cell_ids[j]
                if (cell1, cell2) in distances:
                    print(f"d({cell1}, {cell2}) = {distances[(cell1, cell2)]:.4f}")
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        most_similar = similarity_results['most_similar_pair']
        min_dist = similarity_results['min_distance']
        print(f"\nCells {most_similar[0]} and {most_similar[1]} are most similar")
        print(f"(distance = {min_dist:.4f})")
        
        # Save similarity results
        similarity_csv = output_root / "similarity_results.csv"
        similarity_data = []
        for i in range(n):
            for j in range(i + 1, n):
                cell1 = cell_ids[i]
                cell2 = cell_ids[j]
                if (cell1, cell2) in distances:
                    similarity_data.append({
                        'cell1': cell1,
                        'cell2': cell2,
                        'distance': distances[(cell1, cell2)]
                    })
        similarity_df = pd.DataFrame(similarity_data)
        similarity_df.to_csv(similarity_csv, index=False)
        print(f"\nSimilarity results saved to {similarity_csv}")
    
    print("\n" + "="*60)
    print("✓ FINAL ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nResults saved to: {output_root}/")
    print(f"  - Metrics: {output_root}/final_metrics.csv")
    print(f"  - Similarity: {output_root}/similarity_results.csv")
    print(f"  - QC images: {output_root}/*.png")


if __name__ == "__main__":
    main()

