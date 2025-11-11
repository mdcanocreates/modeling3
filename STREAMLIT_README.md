# Streamlit UI for SAM Mask Refinement

This Streamlit app provides an interactive interface for:
1. **Mask Refinement**: Tune SAM segmentation parameters with sliders and save refined masks
2. **Analysis Summary**: View auto-generated answers to modeling questions based on real pipeline outputs

## Running the App

```bash
# Activate the virtual environment
source venv_sam/bin/activate

# (Optional) Set Gemini API key for QC feature
export GEMINI_API_KEY="your-api-key-here"

# Run the Streamlit app
streamlit run sam_refine_ui.py
```

The app will:
- **Automatically download** the cell images from Dropbox if `img_model/` doesn't exist
- Open in your browser at `http://localhost:8501`

No manual download needed! The dataset will be fetched automatically on first run.

### Setting API Keys

See `SETUP_API_KEYS.md` for detailed instructions on setting the `GEMINI_API_KEY` environment variable.

## Features

### Tab 1: Mask Refinement
- Select a cell (CellA, CellB, or CellC)
- Adjust segmentation parameters via sliders:
  - ROI Margin: Expansion around nuclei for ROI
  - Dilation Radius: For including actin-rich periphery
  - Closing Radius: For morphological smoothing
  - Actin Percentile: Threshold for high-actin region
  - Band Fraction: Fraction of image height for band removal
  - Band Coverage Threshold: Threshold for band removal
- Click "Recompute Mask" to see the result
- Click "Accept & Save" to save the refined mask

Saved masks are automatically used by the main pipeline (`final_analysis.py`) if they exist.

### Tab 2: Analysis Summary
- Displays all metrics used in the analysis with definitions
- Shows normalized metric vectors (z-scores)
- Computes and displays pairwise distances
- Identifies the most similar cell pair
- All values are computed from actual CSV outputs (no hardcoded numbers)

## Saved Files

When you save a mask via the UI:
- Mask: `final_outputs/{cell_id}_cell_mask_manual.npy`
- Parameters: `final_outputs/{cell_id}_cell_mask_params.json`

The main pipeline will automatically use these masks if they exist.
