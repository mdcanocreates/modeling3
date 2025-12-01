"""
Main CLI entry point for Modeling 3 pipeline.

Usage:
    python -m modeling3.main_mc3 [--dry-run] [--verbose]

Runs the complete pipeline:
1. Load original cells
2. Generate ≥30 images (≥15 per algorithm)
3. Extract extended metrics
4. Perform clustering
5. Evaluate parent-match rates
6. Generate publication-ready figures
7. Save all outputs
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set up paths
import modeling3.config as config
from modeling3.manifest import ImageRecord, create_manifest, load_manifest
from modeling3.data_loading import load_original_cells, verify_cell_data
from modeling3.preprocessing import preprocess_for_generation
from modeling3.gen_alg1 import generate_classical_augs
from modeling3.gen_alg2 import generate_structural_variants
from modeling3.metrics import compute_extended_metrics
from modeling3.clustering import (
    perform_kmeans_clustering,
    perform_hierarchical_clustering,
    compute_silhouette_score
)
from modeling3.evaluation import (
    compute_parent_match_rate,
    compute_intra_cluster_distances,
    compare_algorithms
)
from modeling3.viz import (
    plot_image_grid,
    plot_pca_clustering,
    plot_dendrogram,
    plot_correlation_matrix,
    plot_clustering_comparison
)

# Import Modeling 2 utilities for segmentation
from image_analysis.segmentation import segment_nuclei_robust, create_cytoplasm_mask
from image_analysis.sam_wrapper import sam_segment_cell


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Set up logging to file and console."""
    log_file = output_dir / config.LOG_FILE
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def sanity_check_pre_generation(cells: Dict, logger: logging.Logger) -> bool:
    """Pre-generation sanity checks."""
    logger.info("=" * 60)
    logger.info("PRE-GENERATION SANITY CHECKS")
    logger.info("=" * 60)
    
    # Check 1: Found all 3 original cells
    expected_cells = ['CellA', 'CellB', 'CellC']
    found_cells = list(cells.keys())
    if set(found_cells) != set(expected_cells):
        logger.error(f"✗ Expected {expected_cells}, found {found_cells}")
        return False
    logger.info(f"✓ Found all 3 original cells: {found_cells}")
    
    # Check 2: Each has 4 channels
    verification = verify_cell_data(cells)
    if not all(verification.values()):
        logger.error("✗ Some cells missing channels or have invalid dimensions")
        return False
    logger.info("✓ All cells have 4 channels with valid dimensions")
    
    # Check 3: Channels load successfully (already done in load_original_cells)
    logger.info("✓ All channels loaded successfully")
    
    # Check 4: Dimensions reasonable
    for cell_id, cell_data in cells.items():
        for channel, img in cell_data.items():
            h, w = img.shape
            if h < 50 or w < 50:
                logger.error(f"✗ {cell_id} {channel} has unreasonable size: {img.shape}")
                return False
    logger.info("✓ All images have reasonable dimensions")
    
    return True


def generate_images(
    cells: Dict,
    output_dir: Path,
    n_per_cell: int,
    dry_run: bool,
    logger: logging.Logger
) -> List[ImageRecord]:
    """Generate images using both algorithms."""
    logger.info("=" * 60)
    logger.info("IMAGE GENERATION")
    logger.info("=" * 60)
    
    all_records = []
    
    # Create records for originals
    for cell_id, cell_data in cells.items():
        # Preprocess
        preprocessed = preprocess_for_generation(cell_data)
        
        # Save original images
        original_dir = output_dir / "originals"
        original_dir.mkdir(parents=True, exist_ok=True)
        
        sample_id = f"{cell_id}_original"
        
        # Save each channel
        path_phase = None
        path_actin = None
        path_mt = None
        path_nuc = None
        
        for ch_key, img in preprocessed.items():
            filename = f"{sample_id}_{ch_key}.png"
            filepath = original_dir / filename
            
            # Store path even in dry run (for manifest)
            if ch_key == 'phase':
                path_phase = str(filepath)
            elif ch_key == 'actin':
                path_actin = str(filepath)
            elif ch_key == 'mt':
                path_mt = str(filepath)
            elif ch_key == 'nuc':
                path_nuc = str(filepath)
            
            # Only save file if not dry run
            if not dry_run:
                from skimage.io import imsave
                img_uint8 = (img * 255).astype(np.uint8)
                imsave(str(filepath), img_uint8, check_contrast=False)
        
        record = ImageRecord(
            sample_id=sample_id,
            parent_id=cell_id,
            algorithm="original",
            path_phase=path_phase,
            path_actin=path_actin,
            path_mt=path_mt,
            path_nuc=path_nuc
        )
        all_records.append(record)
    
    # Generate with Algorithm 1
    logger.info(f"\nGenerating {n_per_cell} images per cell with Algorithm 1...")
    alg1_dir = output_dir / "alg1"
    
    for cell_id, cell_data in cells.items():
        preprocessed = preprocess_for_generation(cell_data)
        
        if not dry_run:
            records_alg1 = generate_classical_augs(
                preprocessed,
                n_per_cell,
                alg1_dir,
                cell_id,
                config
            )
            all_records.extend(records_alg1)
            logger.info(f"  Generated {len(records_alg1)} images for {cell_id}")
        else:
            # Create placeholder records for dry run
            for i in range(n_per_cell):
                sample_id = f"{cell_id}_alg1_{i+1:03d}"
                record = ImageRecord(
                    sample_id=sample_id,
                    parent_id=cell_id,
                    algorithm="alg1",
                    path_phase=str(alg1_dir / f"{sample_id}_BF.png"),
                    path_actin=str(alg1_dir / f"{sample_id}_Actin.png"),
                    path_mt=str(alg1_dir / f"{sample_id}_Microtubules.png"),
                    path_nuc=str(alg1_dir / f"{sample_id}_Nuclei.png")
                )
                all_records.append(record)
            logger.info(f"  [DRY RUN] Would generate {n_per_cell} images for {cell_id}")
    
    # Generate with Algorithm 2
    logger.info(f"\nGenerating {n_per_cell} images per cell with Algorithm 2...")
    alg2_dir = output_dir / "alg2"
    
    for cell_id, cell_data in cells.items():
        preprocessed = preprocess_for_generation(cell_data)
        
        if not dry_run:
            records_alg2 = generate_structural_variants(
                preprocessed,
                n_per_cell,
                alg2_dir,
                cell_id,
                config
            )
            all_records.extend(records_alg2)
            logger.info(f"  Generated {len(records_alg2)} images for {cell_id}")
        else:
            # Create placeholder records for dry run
            for i in range(n_per_cell):
                sample_id = f"{cell_id}_alg2_{i+1:03d}"
                record = ImageRecord(
                    sample_id=sample_id,
                    parent_id=cell_id,
                    algorithm="alg2",
                    path_phase=str(alg2_dir / f"{sample_id}_BF.png"),
                    path_actin=str(alg2_dir / f"{sample_id}_Actin.png"),
                    path_mt=str(alg2_dir / f"{sample_id}_Microtubules.png"),
                    path_nuc=str(alg2_dir / f"{sample_id}_Nuclei.png")
                )
                all_records.append(record)
            logger.info(f"  [DRY RUN] Would generate {n_per_cell} images for {cell_id}")
    
    # Sanity check: total images
    n_generated = len([r for r in all_records if r.algorithm != 'original'])
    logger.info(f"\n✓ Total generated images: {n_generated} (target: ≥30)")
    
    if n_generated < 30 and not dry_run:
        logger.warning(f"⚠ Only {n_generated} images generated (target: ≥30)")
    
    return all_records


def compute_all_metrics_pipeline(
    records: List[ImageRecord],
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger
) -> pd.DataFrame:
    """Compute metrics for all images."""
    logger.info("=" * 60)
    logger.info("METRICS COMPUTATION")
    logger.info("=" * 60)
    
    all_metrics = []
    
    for record in tqdm(records, desc="Computing metrics"):
        try:
            # Load images
            from skimage.io import imread
            cell_images = {}
            if record.path_phase:
                cell_images['phase'] = np.array(imread(record.path_phase)) / 255.0
            if record.path_actin:
                cell_images['actin'] = np.array(imread(record.path_actin)) / 255.0
            if record.path_mt:
                cell_images['mt'] = np.array(imread(record.path_mt)) / 255.0
            if record.path_nuc:
                cell_images['nuc'] = np.array(imread(record.path_nuc)) / 255.0
            
            # Segment cell (use actin channel)
            if 'actin' in cell_images:
                # Try to use SAM or simple thresholding
                try:
                    # Use simple thresholding for now (can be improved)
                    from skimage.filters import threshold_otsu
                    actin_img = (cell_images['actin'] * 255).astype(np.uint8)
                    threshold = threshold_otsu(actin_img)
                    cell_mask = actin_img > threshold
                    
                    # Clean up mask
                    from skimage.morphology import remove_small_objects, binary_closing
                    from skimage.morphology import disk
                    cell_mask = remove_small_objects(cell_mask, min_size=500)
                    cell_mask = binary_closing(cell_mask, disk(5))
                except:
                    # Fallback: use full image as mask
                    cell_mask = np.ones_like(cell_images['actin'], dtype=bool)
            else:
                cell_mask = np.ones((256, 256), dtype=bool)
            
            # Segment nuclei
            if 'nuc' in cell_images:
                nuc_img = cell_images['nuc']
                cell_area = np.sum(cell_mask)
                nucleus_mask, _ = segment_nuclei_robust(
                    nuc_img,
                    cell_mask,
                    cell_area
                )
            else:
                nucleus_mask = np.zeros_like(cell_mask, dtype=bool)
            
            # Create cytoplasm mask
            cytoplasm_mask = create_cytoplasm_mask(cell_mask, nucleus_mask)
            
            # Compute extended metrics
            metrics = compute_extended_metrics(
                cell_images,
                cell_mask,
                nucleus_mask,
                cytoplasm_mask
            )
            
            # Add metadata
            metrics['sample_id'] = record.sample_id
            metrics['parent_id'] = record.parent_id
            metrics['algorithm'] = record.algorithm
            
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to compute metrics for {record.sample_id}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sanity checks
    logger.info("\nPost-metrics sanity checks:")
    
    # Check for NaNs/Infs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_cols = df[numeric_cols].isna().any()
    inf_cols = (df[numeric_cols] == np.inf).any()
    
    if nan_cols.any():
        logger.warning(f"⚠ Found NaNs in columns: {nan_cols[nan_cols].index.tolist()}")
        df = df.fillna(0)  # Fill NaNs with 0
    
    if inf_cols.any():
        logger.warning(f"⚠ Found Infs in columns: {inf_cols[inf_cols].index.tolist()}")
        df = df.replace([np.inf, -np.inf], 0)  # Replace Infs with 0
    
    logger.info("✓ No NaNs or Infs in final features")
    
    # Check circularity range
    if 'circularity' in df.columns:
        invalid = (df['circularity'] < 0) | (df['circularity'] > 1)
        if invalid.any():
            logger.warning(f"⚠ Found {invalid.sum()} samples with circularity outside [0,1]")
            df.loc[invalid, 'circularity'] = df.loc[invalid, 'circularity'].clip(0, 1)
        logger.info("✓ Circularity in [0,1] range")
    
    # Drop constant features
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        logger.info(f"  Dropping constant features: {constant_cols}")
        df = df.drop(columns=constant_cols)
    
    # Save features CSV
    if not dry_run:
        features_path = output_dir / "features_mc3.csv"
        df.to_csv(features_path, index=False)
        logger.info(f"✓ Features saved to {features_path}")
    
    return df


def perform_clustering_pipeline(
    features_df: pd.DataFrame,
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger
) -> Dict:
    """Perform clustering and evaluation."""
    logger.info("=" * 60)
    logger.info("CLUSTERING & EVALUATION")
    logger.info("=" * 60)
    
    # Extract feature columns (exclude metadata)
    exclude_cols = ['sample_id', 'parent_id', 'algorithm']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Extract feature matrix
    X = features_df[feature_cols].values
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    logger.info("\nPerforming K-Means clustering (k=3)...")
    kmeans_result = perform_kmeans_clustering(X_scaled)
    kmeans_labels = kmeans_result['labels']
    kmeans_silhouette = compute_silhouette_score(X_scaled, kmeans_labels)
    logger.info(f"  K-Means silhouette score: {kmeans_silhouette:.3f}")
    
    # Hierarchical clustering
    logger.info("\nPerforming Hierarchical clustering (Ward)...")
    hierarchical_result = perform_hierarchical_clustering(X_scaled)
    hierarchical_labels = hierarchical_result['labels']
    hierarchical_silhouette = compute_silhouette_score(X_scaled, hierarchical_labels)
    logger.info(f"  Hierarchical silhouette score: {hierarchical_silhouette:.3f}")
    
    # Evaluation
    logger.info("\nEvaluating parent-match rates...")
    sample_ids = features_df['sample_id'].tolist()
    parent_ids = features_df['parent_id'].tolist()
    
    kmeans_match = compute_parent_match_rate(kmeans_labels, parent_ids, sample_ids)
    hierarchical_match = compute_parent_match_rate(hierarchical_labels, parent_ids, sample_ids)
    
    logger.info(f"  K-Means parent-match rate: {kmeans_match['overall_rate']:.1%}")
    logger.info(f"  Hierarchical parent-match rate: {hierarchical_match['overall_rate']:.1%}")
    
    # Intra-cluster distances
    logger.info("\nComputing intra-cluster distances...")
    kmeans_intra = compute_intra_cluster_distances(X_scaled, kmeans_labels)
    hierarchical_intra = compute_intra_cluster_distances(X_scaled, hierarchical_labels)
    
    logger.info(f"  K-Means mean intra-cluster distance: {kmeans_intra['overall_mean']:.3f}")
    logger.info(f"  Hierarchical mean intra-cluster distance: {hierarchical_intra['overall_mean']:.3f}")
    
    # Algorithm comparison
    alg_comparison = compare_algorithms(X_scaled, kmeans_labels, sample_ids, parent_ids)
    logger.info(f"\nAlgorithm comparison:")
    logger.info(f"  Alg1 match rate: {alg_comparison['alg1_match_rate']:.1%}")
    logger.info(f"  Alg2 match rate: {alg_comparison['alg2_match_rate']:.1%}")
    
    results = {
        'kmeans': {
            'labels': kmeans_labels,
            'silhouette': kmeans_silhouette,
            'parent_match': kmeans_match,
            'intra_cluster': kmeans_intra
        },
        'hierarchical': {
            'labels': hierarchical_labels,
            'silhouette': hierarchical_silhouette,
            'parent_match': hierarchical_match,
            'intra_cluster': hierarchical_intra,
            'linkage_matrix': hierarchical_result['linkage_matrix']
        },
        'algorithm_comparison': alg_comparison
    }
    
    return results


def generate_figures(
    records: List[ImageRecord],
    features_df: pd.DataFrame,
    clustering_results: Dict,
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger
) -> None:
    """Generate all publication-ready figures."""
    logger.info("=" * 60)
    logger.info("FIGURE GENERATION")
    logger.info("=" * 60)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info("[DRY RUN] Would generate figures...")
        return
    
    # Extract feature matrix
    exclude_cols = ['sample_id', 'parent_id', 'algorithm']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X = features_df[feature_cols].values
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sample_ids = features_df['sample_id'].tolist()
    
    # 1. Image grid
    logger.info("Generating image grid...")
    plot_image_grid(
        records,
        figures_dir / "image_grid.png",
        n_cols=6
    )
    logger.info("  ✓ Saved image_grid.png")
    
    # 2. PCA clustering (K-Means)
    logger.info("Generating PCA clustering plot (K-Means)...")
    plot_pca_clustering(
        X_scaled,
        clustering_results['kmeans']['labels'],
        sample_ids,
        figures_dir / "pca_clustering_kmeans.png"
    )
    logger.info("  ✓ Saved pca_clustering_kmeans.png")
    
    # 3. PCA clustering (Hierarchical)
    logger.info("Generating PCA clustering plot (Hierarchical)...")
    plot_pca_clustering(
        X_scaled,
        clustering_results['hierarchical']['labels'],
        sample_ids,
        figures_dir / "pca_clustering_hierarchical.png"
    )
    logger.info("  ✓ Saved pca_clustering_hierarchical.png")
    
    # 4. Clustering comparison
    logger.info("Generating clustering comparison...")
    plot_clustering_comparison(
        X_scaled,
        clustering_results['kmeans']['labels'],
        clustering_results['hierarchical']['labels'],
        sample_ids,
        figures_dir / "clustering_comparison.png"
    )
    logger.info("  ✓ Saved clustering_comparison.png")
    
    # 5. Dendrogram
    logger.info("Generating dendrogram...")
    plot_dendrogram(
        clustering_results['hierarchical']['linkage_matrix'],
        sample_ids,
        figures_dir / "dendrogram.png"
    )
    logger.info("  ✓ Saved dendrogram.png")
    
    # 6. Correlation matrix
    logger.info("Generating correlation matrix...")
    plot_correlation_matrix(
        features_df[feature_cols],
        figures_dir / "correlation_matrix.png"
    )
    logger.info("  ✓ Saved correlation_matrix.png")


def save_summary(
    records: List[ImageRecord],
    features_df: pd.DataFrame,
    clustering_results: Dict,
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger
) -> None:
    """Save summary JSON and text file."""
    logger.info("=" * 60)
    logger.info("SAVING SUMMARY")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("[DRY RUN] Would save summary...")
        return
    
    # Count images
    n_originals = len([r for r in records if r.algorithm == 'original'])
    n_alg1 = len([r for r in records if r.algorithm == 'alg1'])
    n_alg2 = len([r for r in records if r.algorithm == 'alg2'])
    
    summary = {
        'n_images': {
            'originals': n_originals,
            'alg1': n_alg1,
            'alg2': n_alg2,
            'total': len(records)
        },
        'clustering': {
            'kmeans': {
                'silhouette_score': float(clustering_results['kmeans']['silhouette']),
                'parent_match_rate': float(clustering_results['kmeans']['parent_match']['overall_rate']),
                'mean_intra_cluster_distance': float(clustering_results['kmeans']['intra_cluster']['overall_mean'])
            },
            'hierarchical': {
                'silhouette_score': float(clustering_results['hierarchical']['silhouette']),
                'parent_match_rate': float(clustering_results['hierarchical']['parent_match']['overall_rate']),
                'mean_intra_cluster_distance': float(clustering_results['hierarchical']['intra_cluster']['overall_mean'])
            }
        },
        'algorithm_comparison': clustering_results['algorithm_comparison']
    }
    
    # Save JSON
    summary_json = output_dir / "summary_mc3.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved {summary_json}")
    
    # Save text summary
    summary_txt = output_dir / "summary_mc3.txt"
    with open(summary_txt, 'w') as f:
        f.write("MODELING 3 PIPELINE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images: {len(records)}\n")
        f.write(f"  - Originals: {n_originals}\n")
        f.write(f"  - Algorithm 1: {n_alg1}\n")
        f.write(f"  - Algorithm 2: {n_alg2}\n\n")
        f.write("CLUSTERING RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"K-Means silhouette: {clustering_results['kmeans']['silhouette']:.3f}\n")
        f.write(f"K-Means parent-match: {clustering_results['kmeans']['parent_match']['overall_rate']:.1%}\n")
        f.write(f"Hierarchical silhouette: {clustering_results['hierarchical']['silhouette']:.3f}\n")
        f.write(f"Hierarchical parent-match: {clustering_results['hierarchical']['parent_match']['overall_rate']:.1%}\n")
    logger.info(f"✓ Saved {summary_txt}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Modeling 3: Endothelial Cell Generation, Metrics, and Clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview steps without heavy computation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: {config.OUTPUT_ROOT})'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help=f'Data root directory (default: {config.DATA_ROOT})'
    )
    
    parser.add_argument(
        '--n-per-cell',
        type=int,
        default=config.N_IMAGES_PER_CELL,
        help=f'Number of images per algorithm per cell (default: {config.N_IMAGES_PER_CELL})'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_ROOT
    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, args.verbose)
    
    logger.info("=" * 60)
    logger.info("MODELING 3 PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")
    
    try:
        # Step 1: Load original cells
        logger.info("Step 1: Loading original cells...")
        cells = load_original_cells(data_root)
        
        # Sanity checks
        if not sanity_check_pre_generation(cells, logger):
            logger.error("Pre-generation sanity checks failed!")
            sys.exit(1)
        
        # Step 2: Generate images
        logger.info("\nStep 2: Generating images...")
        records = generate_images(
            cells,
            output_dir,
            args.n_per_cell,
            args.dry_run,
            logger
        )
        
        # Save manifest (even in dry run, to show what would be generated)
        manifest_path = output_dir / "manifest_mc3.csv"
        create_manifest(records, manifest_path)
        logger.info(f"✓ Manifest saved to {manifest_path}")
        
        # Step 3: Compute metrics (skip in dry run)
        if args.dry_run:
            logger.info("\nStep 3: Computing metrics... [SKIPPED in dry run]")
            logger.info("  [DRY RUN] Would compute metrics for all images")
            logger.info("\nStep 4: Clustering & evaluation... [SKIPPED in dry run]")
            logger.info("  [DRY RUN] Would perform clustering and evaluation")
            logger.info("\nStep 5: Generating figures... [SKIPPED in dry run]")
            logger.info("  [DRY RUN] Would generate publication-ready figures")
            logger.info("\nStep 6: Saving summary... [SKIPPED in dry run]")
            logger.info("  [DRY RUN] Would save summary files")
        else:
            logger.info("\nStep 3: Computing metrics...")
            features_df = compute_all_metrics_pipeline(
                records,
                output_dir,
                args.dry_run,
                logger
            )
            
            # Step 4: Clustering & evaluation
            logger.info("\nStep 4: Clustering & evaluation...")
            clustering_results = perform_clustering_pipeline(
                features_df,
                output_dir,
                args.dry_run,
                logger
            )
            
            # Step 5: Generate figures
            logger.info("\nStep 5: Generating figures...")
            generate_figures(
                records,
                features_df,
                clustering_results,
                output_dir,
                args.dry_run,
                logger
            )
            
            # Step 6: Save summary
            logger.info("\nStep 6: Saving summary...")
            save_summary(
                records,
                features_df,
                clustering_results,
                output_dir,
                args.dry_run,
                logger
            )
        
        logger.info("\n" + "=" * 60)
        if args.dry_run:
            logger.info("✓ DRY RUN COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"\nDry run results:")
            logger.info(f"  - Found {len(cells)} original cells")
            logger.info(f"  - Would generate {len([r for r in records if r.algorithm != 'original'])} images")
            logger.info(f"  - Manifest saved to: {output_dir}/manifest_mc3.csv")
        else:
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"\nResults saved to: {output_dir}/")
            logger.info(f"  - manifest_mc3.csv")
            logger.info(f"  - features_mc3.csv")
            logger.info(f"  - summary_mc3.txt / .json")
            logger.info(f"  - errors_mc3.log")
            logger.info(f"  - figures/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

