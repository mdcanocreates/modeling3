"""
Streamlit UI for SAM mask refinement and analysis summary.

This app provides:
1. Interactive mask refinement with tunable parameters
2. Analysis summary with real metrics and similarity results
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io, color

from image_analysis.sam_parametric import segment_cell_with_sam
from image_analysis.io_utils import CELL_ID_TO_PREFIX
from image_analysis.metrics import (
    compute_cell_area, compute_circularity, compute_aspect_ratio
)


# Page config
st.set_page_config(
    layout="wide",
    page_title="Cell SAM Refinement & Analysis",
    page_icon="🔬"
)

# Initialize session state
if 'masks' not in st.session_state:
    st.session_state.masks = {}
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'images' not in st.session_state:
    st.session_state.images = {}


def load_cell_paths(cell_id: str, data_root: Path) -> dict:
    """Load image paths for a cell."""
    filename_prefix = CELL_ID_TO_PREFIX.get(cell_id, cell_id)
    
    actin_path = data_root / cell_id / f"{filename_prefix}_Actin.jpg"
    if not actin_path.exists():
        actin_path = data_root / cell_id / f"{cell_id}_Actin.jpg"
    
    nuclei_path = data_root / cell_id / f"{filename_prefix}_Nuclei.jpg"
    if not nuclei_path.exists():
        nuclei_path = data_root / cell_id / f"{cell_id}_Nuclei.jpg"
    
    return {
        'actin': str(actin_path) if actin_path.exists() else None,
        'nuclei': str(nuclei_path) if nuclei_path.exists() else None
    }


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Overlay mask on image with transparency."""
    if len(image.shape) == 2:
        image_rgb = color.gray2rgb(image)
    else:
        image_rgb = image.copy()
    
    # Normalize image to 0-1
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb.astype(np.float32) / 255.0
    
    # Create overlay
    overlay = image_rgb.copy()
    color_array = np.array([0, 1, 0])  # Green
    
    # Apply mask overlay
    mask_3d = np.stack([mask] * 3, axis=-1)
    mask_pixels = overlay[mask]
    overlay[mask] = (
        alpha * color_array + (1 - alpha) * mask_pixels
    )
    
    return overlay


# Main app
st.title("🔬 Cell SAM Refinement & Analysis")

# Create tabs
tab1, tab2 = st.tabs(["Mask Refinement", "Analysis Summary"])

# ========================================================================
# Tab 1: Mask Refinement
# ========================================================================
with tab1:
    st.header("Interactive SAM Mask Refinement")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Cell Selection")
        cell_id = st.selectbox("Cell", ["CellA", "CellB", "CellC"])
        
        st.subheader("Segmentation Parameters")
        margin = st.slider("ROI Margin", 0, 300, 150, help="Expansion margin around nuclei for ROI")
        dilate_radius = st.slider("Dilation Radius", 0, 30, 8, help="Radius for including actin-rich periphery")
        close_radius = st.slider("Closing Radius", 0, 15, 3, help="Radius for morphological smoothing")
        actin_percentile = st.slider("Actin Percentile", 0.1, 0.7, 0.3, 0.05, help="Percentile for high-actin region threshold")
        band_frac = st.slider("Band Fraction", 0.0, 0.1, 0.03, 0.01, help="Fraction of image height for band removal")
        band_coverage = st.slider("Band Coverage Threshold", 0.0, 0.8, 0.4, 0.05, help="Threshold for band removal")
        
        col1, col2 = st.columns(2)
        with col1:
            recompute_btn = st.button("🔄 Recompute Mask", type="primary")
        with col2:
            save_btn = st.button("💾 Accept & Save")
    
    # Data root
    data_root = Path("img_model")
    output_root = Path("final_outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Load cell paths
    paths = load_cell_paths(cell_id, data_root)
    
    if not paths['actin']:
        st.error(f"Actin image not found for {cell_id}")
        st.stop()
    
    # Recompute mask button
    if recompute_btn:
        with st.spinner(f"Computing mask for {cell_id}..."):
            try:
                result = segment_cell_with_sam(
                    cell_id=cell_id,
                    actin_path=paths['actin'],
                    nuclei_path=paths['nuclei'],
                    margin=margin,
                    dilate_radius=dilate_radius,
                    close_radius=close_radius,
                    actin_percentile=actin_percentile,
                    band_frac=band_frac,
                    band_coverage=band_coverage
                )
                
                # Store in session state
                st.session_state.masks[cell_id] = result['cell_mask']
                st.session_state.images[cell_id] = result['actin_img']
                st.session_state.params[cell_id] = {
                    'margin': margin,
                    'dilate_radius': dilate_radius,
                    'close_radius': close_radius,
                    'actin_percentile': actin_percentile,
                    'band_frac': band_frac,
                    'band_coverage': band_coverage
                }
                
                st.success(f"Mask computed for {cell_id}!")
            except Exception as e:
                st.error(f"Error computing mask: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display mask if available
    if cell_id in st.session_state.masks:
        mask = st.session_state.masks[cell_id]
        actin_img = st.session_state.images[cell_id]
        
        # Create overlay
        overlay = overlay_mask_on_image(actin_img, mask, alpha=0.3)
        
        # Display image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{cell_id} - Actin + Cell Mask")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(overlay)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.subheader("Metrics")
            
            # Compute metrics from mask
            cell_area_px = np.sum(mask)
            cell_area_k = cell_area_px / 1000.0
            circularity = compute_circularity(mask)
            aspect_ratio = compute_aspect_ratio(mask)
            
            metrics_df = pd.DataFrame({
                'Metric': ['Cell Area (pixels)', 'Cell Area (×10³)', 'Circularity', 'Aspect Ratio'],
                'Value': [f"{cell_area_px:.0f}", f"{cell_area_k:.1f}", f"{circularity:.3f}", f"{aspect_ratio:.2f}"]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Show current parameters
            st.subheader("Current Parameters")
            params = st.session_state.params[cell_id]
            params_df = pd.DataFrame({
                'Parameter': list(params.keys()),
                'Value': [f"{v:.2f}" if isinstance(v, float) else str(v) for v in params.values()]
            })
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Save button
        if save_btn:
            # Save mask
            mask_path = output_root / f"{cell_id}_cell_mask_manual.npy"
            np.save(mask_path, mask)
            
            # Save parameters
            params_path = output_root / f"{cell_id}_cell_mask_params.json"
            with open(params_path, 'w') as f:
                json.dump(st.session_state.params[cell_id], f, indent=2)
            
            st.success(f"✅ Mask and parameters saved!")
            st.info(f"Mask: `{mask_path}`\nParameters: `{params_path}`")
    else:
        st.info("👈 Click 'Recompute Mask' to generate a mask for this cell")


# ========================================================================
# Tab 2: Analysis Summary
# ========================================================================
with tab2:
    st.header("Analysis Summary")
    st.markdown("**Answers to the modeling questions based on real pipeline outputs**")
    
    # Load metrics
    metrics_path = Path("final_outputs/final_metrics.csv")
    distances_path = Path("final_outputs/similarity_results.csv")
    
    if not metrics_path.exists():
        st.error(f"Metrics CSV not found: {metrics_path}")
        st.info("Run the analysis pipeline first: `python3 final_analysis.py`")
        st.stop()
    
    # Load metrics CSV
    df_metrics = pd.read_csv(metrics_path)
    
    if 'cell_id' not in df_metrics.columns:
        st.error("Metrics CSV must contain 'cell_id' column")
        st.stop()
    
    # Get cell IDs
    cell_ids = df_metrics['cell_id'].tolist()
    
    # Section 1: Metrics Used
    st.markdown("---")
    st.markdown("## 1. Metrics Used")
    
    # Identify metric columns (exclude non-metric columns)
    exclude_cols = {'cell_id', '_gemini_qc'}
    metric_columns = [col for col in df_metrics.columns if col not in exclude_cols]
    
    # Metric definitions
    metric_definitions = {
        'cell_area': 'Cell area (pixels): Number of pixels in the cell mask',
        'circularity': 'Circularity: (4πA / P²), where A is area and P is perimeter',
        'aspect_ratio': 'Aspect ratio: Major axis / minor axis of best-fit ellipse',
        'nuclear_count': 'Nuclear count: Number of nuclei detected within the cell',
        'nuclear_area': 'Nuclear area (pixels): Total area of all nuclei',
        'nc_ratio': 'N:C ratio: Nuclear area / cytoplasmic area',
        'actin_mean_intensity': 'Actin mean intensity: Background-normalized mean actin fluorescence in cytoplasm (0-1)',
        'actin_anisotropy': 'Actin anisotropy: Orientation order parameter of actin filaments (0-1)',
        'mtub_mean_intensity': 'Microtubule mean intensity: Background-normalized mean microtubule fluorescence in cytoplasm (0-1)'
    }
    
    st.markdown("The following metrics were computed for each cell:")
    
    metrics_list = []
    for col in metric_columns:
        definition = metric_definitions.get(col, f"{col}: Computed metric")
        metrics_list.append(f"- **{col}**: {definition}")
    
    st.markdown("\n".join(metrics_list))
    
    # Show actual metrics table
    st.markdown("### Actual Metrics (from pipeline output)")
    display_df = df_metrics.set_index('cell_id')[metric_columns]
    st.dataframe(display_df, use_container_width=True)
    
    # Section 2: Quantitative Similarity
    st.markdown("---")
    st.markdown("## 2. Quantitative Similarity Between Cells")
    
    # Compute or load distances
    if distances_path.exists():
        df_distances = pd.read_csv(distances_path)
        st.info(f"Loaded distances from: {distances_path}")
    else:
        st.info("Computing distances from metrics...")
        # Normalize metrics
        metrics_df = df_metrics.set_index('cell_id')
        normalized_df = metrics_df[metric_columns].copy()
        
        for col in metric_columns:
            values = normalized_df[col].values
            mean = np.mean(values)
            std = np.std(values)
            if std > 1e-10:
                normalized_df[col] = (values - mean) / std
            else:
                normalized_df[col] = 0.0
        
        # Compute pairwise distances
        distances = {}
        n = len(cell_ids)
        for i in range(n):
            for j in range(i + 1, n):
                cell1 = cell_ids[i]
                cell2 = cell_ids[j]
                
                vec1 = normalized_df.loc[cell1, metric_columns].values
                vec2 = normalized_df.loc[cell2, metric_columns].values
                
                distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
                distances[(cell1, cell2)] = distance
        
        # Create DataFrame
        distance_list = []
        for (cell1, cell2), dist in distances.items():
            distance_list.append({'cell1': cell1, 'cell2': cell2, 'distance': dist})
        df_distances = pd.DataFrame(distance_list)
    
    # Normalize metrics for display
    metrics_df = df_metrics.set_index('cell_id')
    normalized_df = metrics_df[metric_columns].copy()
    
    for col in metric_columns:
        values = normalized_df[col].values
        mean = np.mean(values)
        std = np.std(values)
        if std > 1e-10:
            normalized_df[col] = (values - mean) / std
        else:
            normalized_df[col] = 0.0
    
    # Show normalized vectors
    st.markdown("### Normalized Metric Vectors (z-scores)")
    st.markdown("Each metric is normalized using z-score: $\\tilde{m}_k = \\frac{m_k - \\mu_k}{\\sigma_k}$")
    
    norm_display = normalized_df.round(3)
    st.dataframe(norm_display, use_container_width=True)
    
    # Show pairwise distances
    st.markdown("### Pairwise Distances")
    st.markdown("Euclidean distance in normalized metric space:")
    st.markdown("$$d(i,j) = \\sqrt{\\sum_k (\\tilde{m}_{ik} - \\tilde{m}_{jk})^2}$$")
    
    # Format distances table
    distances_display = df_distances.copy()
    distances_display['Pair'] = distances_display['cell1'] + '–' + distances_display['cell2']
    distances_display = distances_display[['Pair', 'distance']].rename(columns={'distance': 'Distance'})
    distances_display['Distance'] = distances_display['Distance'].round(4)
    st.dataframe(distances_display[['Pair', 'Distance']], use_container_width=True, hide_index=True)
    
    # Find most similar pair
    min_distance = df_distances['distance'].min()
    most_similar = df_distances.loc[df_distances['distance'].idxmin()]
    cell1 = most_similar['cell1']
    cell2 = most_similar['cell2']
    
    # Conclusion
    st.markdown("### Conclusion")
    st.success(
        f"**Based on Euclidean distance in z-scored metric space, "
        f"{cell1} and {cell2} are the most similar** "
        f"(distance = {min_distance:.4f})."
    )
    
    # Show which cell is the outlier
    # Find the cell that is NOT in the most similar pair
    remaining_cell = [c for c in cell_ids if c not in [cell1, cell2]]
    if remaining_cell:
        st.info(f"The remaining cell ({remaining_cell[0]}) is the outlier based on the chosen metrics.")
    
    # Display heatmap if available
    heatmap_path = Path("final_outputs/pairwise_distances_heatmap.png")
    if heatmap_path.exists():
        st.markdown("### Distance Heatmap")
        st.image(str(heatmap_path), use_container_width=True)
    
    # Data source note
    st.markdown("---")
    st.caption("📊 All metrics and distances computed from actual pipeline outputs. No hardcoded values.")


if __name__ == "__main__":
    pass

