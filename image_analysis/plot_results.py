"""
Results visualization module for generating print-ready dashboard figures.

This module creates Matplotlib figures showing:
1. Segmentation overlays (Actin + SAM masks) for each cell
2. Quantitative metrics bar plots
3. Pairwise distance heatmap (optional)

All data is loaded from actual pipeline outputs - no hardcoded values.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io


def generate_results_dashboard(
    metrics_csv_path: str = "final_outputs/final_metrics.csv",
    overlay_dir: str = "final_outputs",
    output_path: str = "final_outputs/final_results_dashboard.png",
    selected_metrics: Optional[List[str]] = None
) -> None:
    """
    Load real metrics and overlay images for CellA/B/C and create a print-ready
    Matplotlib figure showing:
      - Left: actin + SAM cell mask overlays
      - Right: bar plots of quantitative metrics for each cell.
    
    All data is loaded from actual pipeline outputs - no hardcoded values.
    
    Parameters
    ----------
    metrics_csv_path : str
        Path to the metrics CSV file (e.g., final_metrics.csv)
    overlay_dir : str
        Directory containing overlay images
    output_path : str
        Path to save the output figure
    selected_metrics : list, optional
        List of metric names to plot. If None, uses default set.
    """
    # Load metrics from CSV
    metrics_path = Path(metrics_csv_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")
    
    df = pd.read_csv(metrics_path)
    
    # Verify required columns
    if 'cell_id' not in df.columns:
        raise ValueError("Metrics CSV must contain 'cell_id' column")
    
    # Get cell IDs from CSV
    cell_ids = df['cell_id'].tolist()
    
    # Default metrics to plot (if they exist in CSV)
    if selected_metrics is None:
        default_metrics = [
            'cell_area',
            'circularity',
            'aspect_ratio',
            'actin_mean_intensity',
            'actin_anisotropy',
            'mtub_mean_intensity'
        ]
        # Only include metrics that exist in the CSV
        selected_metrics = [m for m in default_metrics if m in df.columns]
    
    # Filter to only metrics that exist
    available_metrics = [m for m in selected_metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        raise ValueError("No valid metrics found in CSV. Available columns: " + ", ".join(df.columns))
    
    print(f"Plotting metrics: {', '.join(available_metrics)}")
    
    # Set up figure: 3 rows (cells) × 4 columns (image, cell_area, other metrics)
    n_cells = len(cell_ids)
    fig = plt.figure(figsize=(18, 5 * n_cells))
    
    overlay_dir_path = Path(overlay_dir)
    
    # Separate cell_area from other metrics
    if 'cell_area' in available_metrics:
        area_metrics = ['cell_area']
        other_metrics = [m for m in available_metrics if m != 'cell_area']
    else:
        area_metrics = []
        other_metrics = available_metrics
    
    # Process each cell
    for i, cell_id in enumerate(cell_ids):
        # Left column: Load and display overlay image
        ax_img = plt.subplot2grid((n_cells, 4), (i, 0), colspan=1)
        
        # Build overlay image path
        overlay_path = overlay_dir_path / f"{cell_id}_actin_with_cell_mask.png"
        
        if not overlay_path.exists():
            # Try alternative naming
            overlay_path = overlay_dir_path / f"{cell_id}_actin_with_cell_mask_sam.png"
        
        if overlay_path.exists():
            img = io.imread(overlay_path)
            ax_img.imshow(img)
            ax_img.set_title(f"{cell_id} (Actin + SAM mask)", fontsize=12, fontweight='bold')
        else:
            ax_img.text(0.5, 0.5, f"Overlay not found:\n{overlay_path.name}", 
                       ha='center', va='center', transform=ax_img.transAxes,
                       fontsize=10, color='red')
            ax_img.set_title(f"{cell_id} (Overlay missing)", fontsize=12)
        
        ax_img.axis('off')
        
        # Get metrics for this cell
        cell_row = df[df['cell_id'] == cell_id].iloc[0]
        
        # Column 2: Cell area (if present)
        if area_metrics:
            ax_area = plt.subplot2grid((n_cells, 4), (i, 1), colspan=1)
            area_values = [cell_row[m] / 1000.0 for m in area_metrics]  # Convert to ×10³
            x_pos_area = np.arange(len(area_metrics))
            bars_area = ax_area.bar(x_pos_area, area_values, alpha=0.7, color='steelblue')
            
            ax_area.set_xticks(x_pos_area)
            ax_area.set_xticklabels(['Cell area\n(×10³ pixels)'], rotation=0, ha='center', fontsize=9)
            ax_area.set_ylabel('Value', fontsize=9)
            if i == 0:
                ax_area.set_title("Cell Area", fontsize=11, fontweight='bold')
            ax_area.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, val in zip(bars_area, area_values):
                height = bar.get_height()
                label = f'{val:.1f}'
                ax_area.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=8)
            
            # Set y-axis limits
            y_max_area = max(area_values) * 1.2 if area_values else 1
            ax_area.set_ylim(0, y_max_area)
        
        # Columns 3-4: Other metrics
        if other_metrics:
            ax_other = plt.subplot2grid((n_cells, 4), (i, 2), colspan=2)
            other_values = [cell_row[m] for m in other_metrics]
            x_pos_other = np.arange(len(other_metrics))
            bars_other = ax_other.bar(x_pos_other, other_values, alpha=0.7, color='steelblue')
            
            ax_other.set_xticks(x_pos_other)
            ax_other.set_xticklabels(other_metrics, rotation=45, ha='right', fontsize=9)
            ax_other.set_ylabel('Value', fontsize=9)
            if i == 0:
                ax_other.set_title("Other Metrics", fontsize=11, fontweight='bold')
            ax_other.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for j, (bar, val) in enumerate(zip(bars_other, other_values)):
                height = bar.get_height()
                # Format value based on magnitude
                if abs(val) < 0.01:
                    label = f'{val:.4f}'
                elif abs(val) < 1:
                    label = f'{val:.3f}'
                elif abs(val) < 1000:
                    label = f'{val:.2f}'
                else:
                    label = f'{val:.0f}'
                
                ax_other.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=7)
            
            # Set y-axis limits
            y_max_other = max(other_values) * 1.2 if other_values else 1
            y_min_other = min(0, min(other_values) * 1.1) if other_values else 0
            ax_other.set_ylim(y_min_other, y_max_other)
        
        # If no metrics, show message
        if not area_metrics and not other_metrics:
            ax_other = plt.subplot2grid((n_cells, 4), (i, 1), colspan=3)
            ax_other.text(0.5, 0.5, 'No metrics available', 
                         ha='center', va='center', transform=ax_other.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to: {output_path}")
    
    plt.close()


def generate_distance_heatmap(
    distances_csv_path: str = "final_outputs/similarity_results.csv",
    output_path: str = "final_outputs/pairwise_distances_heatmap.png"
) -> None:
    """
    Load the REAL pairwise distances between cells from a CSV and plot
    a heatmap for A/B/C.
    
    All data is loaded from actual pipeline outputs - no hardcoded values.
    
    Parameters
    ----------
    distances_csv_path : str
        Path to the distances CSV file (e.g., similarity_results.csv)
    output_path : str
        Path to save the output figure
    """
    distances_path = Path(distances_csv_path)
    
    if not distances_path.exists():
        print(f"Warning: Distances CSV not found: {distances_csv_path}")
        print("Skipping distance heatmap generation.")
        return
    
    # Load distances CSV
    df = pd.read_csv(distances_path)
    
    # Check required columns
    required_cols = ['cell1', 'cell2', 'distance']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Distances CSV must contain columns: {required_cols}")
    
    # Get unique cell IDs
    all_cells = sorted(set(df['cell1'].tolist() + df['cell2'].tolist()))
    
    # Create distance matrix
    n = len(all_cells)
    distance_matrix = np.full((n, n), np.nan)
    
    # Fill matrix from CSV data
    for _, row in df.iterrows():
        cell1 = row['cell1']
        cell2 = row['cell2']
        dist = row['distance']
        
        i = all_cells.index(cell1)
        j = all_cells.index(cell2)
        
        # Distance matrix is symmetric
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist
    
    # Diagonal should be 0 (distance to self)
    np.fill_diagonal(distance_matrix, 0.0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(all_cells)
    ax.set_yticklabels(all_cells)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Euclidean Distance', fontsize=12)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(distance_matrix[i, j]):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Pairwise Cell Similarity (Distance Matrix)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cell', fontsize=12)
    ax.set_ylabel('Cell', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distance heatmap saved to: {output_path}")
    
    plt.close()


def main():
    """CLI entry point for generating visualization dashboards."""
    parser = argparse.ArgumentParser(
        description="Generate print-ready results dashboard from pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate main dashboard
  python -m image_analysis.plot_results --metrics-csv final_outputs/final_metrics.csv
  
  # Generate dashboard and heatmap
  python -m image_analysis.plot_results --metrics-csv final_outputs/final_metrics.csv --heatmap
  
  # Custom output path
  python -m image_analysis.plot_results --output results_dashboard.png
        """
    )
    
    parser.add_argument(
        '--metrics-csv',
        type=str,
        default='final_outputs/final_metrics.csv',
        help='Path to metrics CSV file (default: final_outputs/final_metrics.csv)'
    )
    
    parser.add_argument(
        '--overlay-dir',
        type=str,
        default='final_outputs',
        help='Directory containing overlay images (default: final_outputs)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='final_outputs/final_results_dashboard.png',
        help='Output path for dashboard figure (default: final_outputs/final_results_dashboard.png)'
    )
    
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Also generate pairwise distance heatmap'
    )
    
    parser.add_argument(
        '--distances-csv',
        type=str,
        default='final_outputs/similarity_results.csv',
        help='Path to distances CSV file for heatmap (default: final_outputs/similarity_results.csv)'
    )
    
    parser.add_argument(
        '--heatmap-output',
        type=str,
        default='final_outputs/pairwise_distances_heatmap.png',
        help='Output path for heatmap figure (default: final_outputs/pairwise_distances_heatmap.png)'
    )
    
    args = parser.parse_args()
    
    # Generate main dashboard
    print("="*60)
    print("Generating Results Dashboard")
    print("="*60)
    print(f"Metrics CSV: {args.metrics_csv}")
    print(f"Overlay directory: {args.overlay_dir}")
    print(f"Output: {args.output}")
    print()
    
    try:
        generate_results_dashboard(
            metrics_csv_path=args.metrics_csv,
            overlay_dir=args.overlay_dir,
            output_path=args.output
        )
    except Exception as e:
        print(f"ERROR: Failed to generate dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate heatmap if requested
    if args.heatmap:
        print()
        print("="*60)
        print("Generating Distance Heatmap")
        print("="*60)
        print(f"Distances CSV: {args.distances_csv}")
        print(f"Output: {args.heatmap_output}")
        print()
        
        try:
            generate_distance_heatmap(
                distances_csv_path=args.distances_csv,
                output_path=args.heatmap_output
            )
        except Exception as e:
            print(f"ERROR: Failed to generate heatmap: {e}")
            import traceback
            traceback.print_exc()
            # Don't exit - heatmap is optional
    
    print()
    print("="*60)
    print("✓ Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()

