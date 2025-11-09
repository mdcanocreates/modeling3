"""
Google Gemini QC integration for evaluating segmentation masks.

This module uses Google's Gemini API to evaluate the quality of segmentation
masks by analyzing raw images and their overlay visualizations.

IMPORTANT: Gemini NEVER modifies masks directly; it only evaluates overlays
and suggests parameter tweaks. The classical skimage-based pipeline owns segmentation.
"""

import os
import json
import re
from typing import Dict, Optional
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")


# Gemini prompt for segmentation evaluation
GEMINI_PROMPT = """You are an expert in fluorescence microscopy and image segmentation.

IMAGE_1: raw microscope image for a single endothelial cell (either Actin or Nuclei channel).
IMAGE_2: same field, with GREEN contour = cell boundary mask and MAGENTA contour = nuclear mask(s).

Evaluate the accuracy of these masks.

1. Score the cell mask accuracy from 0.0 to 1.0, where:
   - 1.0 = perfectly follows the true cell boundary;
   - 0.0 = mostly incorrect (missing large parts or including large non-cell regions).

2. Score the nuclear mask accuracy from 0.0 to 1.0, where:
   - 1.0 = all nuclei of this cell are captured with tight boundaries and no large extra regions;
   - 0.0 = nuclei are missing or mostly wrong.
   If there is no nuclear mask visible (e.g. this is an Actin-only overlay), set nucleus_mask_score to null.

3. Briefly list the main issues you see, e.g.:
   - "cell mask includes large background region in the top-left corner"
   - "one bright nucleus is completely missed"
   - "nuclear mask spills outside the actual bright nuclear bodies"

4. Suggest concrete image-processing operations (in words) that could improve the masks.
   Focus on simple things like:
   - "crop to a tighter ROI around the nuclei"
   - "increase min_size for remove_small_objects"
   - "use adaptive thresholding instead of global Otsu"
   - "dilate nucleus mask by 1–2 pixels to include full bodies"

Respond ONLY with a compact JSON object, no extra text:

{
  "cell_mask_score": <float between 0 and 1>,
  "nucleus_mask_score": <float between 0 and 1 or null>,
  "issues": [<string>, ...],
  "suggested_ops": [<string>, ...]
}
"""


def load_image_file(image_path: Path) -> bytes:
    """
    Load image file as bytes for Gemini API.
    
    Parameters
    ----------
    image_path : Path
        Path to image file
    
    Returns
    -------
    bytes
        Image file contents as bytes
    """
    with open(image_path, 'rb') as f:
        return f.read()


def parse_gemini_json_response(text: str) -> Dict:
    """
    Parse JSON response from Gemini, handling code fences if present.
    
    Parameters
    ----------
    text : str
        Raw text response from Gemini
    
    Returns
    -------
    dict
        Parsed JSON as dictionary
    """
    # Remove markdown code fences if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    # Parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # If parsing fails, return default structure
        print(f"Warning: Failed to parse Gemini JSON response: {e}")
        print(f"Response text: {text[:200]}...")
        return {
            "cell_mask_score": None,
            "nucleus_mask_score": None,
            "issues": ["Failed to parse Gemini response"],
            "suggested_ops": []
        }


def evaluate_segmentation_with_gemini(
    cell_id: str,
    raw_image_path: str,
    overlay_image_path: str,
    channel: str,
    model_name: str = "gemini-2.5-flash"
) -> Dict:
    """
    Send raw + overlay images to Gemini, get QC scores and suggestions.
    
    This function evaluates segmentation quality by sending both the raw
    channel image and the overlay visualization to Gemini for analysis.
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell (e.g., "CellA")
    raw_image_path : str
        Path to raw channel image (Actin or Nuclei)
    overlay_image_path : str
        Path to overlay image with masks (green=cell, magenta=nuclei)
    channel : str
        Channel name ("actin" or "nuclei")
    model_name : str
        Gemini model name (default: "gemini-2.5-flash")
    
    Returns
    -------
    dict
        Dictionary with QC evaluation results:
        {
            "cell_id": str,
            "channel": str,
            "cell_mask_score": float (0-1) or None,
            "nucleus_mask_score": float (0-1) or None,
            "issues": list[str],
            "suggested_ops": list[str]
        }
    """
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        return {
            "cell_id": cell_id,
            "channel": channel,
            "cell_mask_score": None,
            "nucleus_mask_score": None,
            "issues": ["Google Generative AI SDK not installed"],
            "suggested_ops": []
        }
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Skipping Gemini QC.")
        return {
            "cell_id": cell_id,
            "channel": channel,
            "cell_mask_score": None,
            "nucleus_mask_score": None,
            "issues": ["GEMINI_API_KEY not set"],
            "suggested_ops": []
        }
    
    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
        
        # Try the specified model, or fall back to alternatives
        # Updated model names: gemini-2.5-flash, gemini-1.5-flash, gemini-1.5-pro, etc.
        model = None
        models_to_try = [
            model_name,
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-pro-vision"
        ]
        
        last_error = None
        selected_model_name = None
        for m in models_to_try:
            try:
                # Test model creation
                model = genai.GenerativeModel(m)
                selected_model_name = m
                # Verify model name is set correctly
                if m != model_name:
                    print(f"    Using model: {m} (requested {model_name} not available)")
                else:
                    print(f"    Using model: {m}")
                break
            except Exception as e:
                last_error = e
                if "404" in str(e) or "not found" in str(e).lower():
                    # Model doesn't exist, try next
                    continue
                else:
                    # Other error, log and try next
                    print(f"    Warning: Model {m} failed: {e}")
                    continue
        
        if model is None:
            error_msg = f"No available Gemini model found. Last error: {last_error}"
            print(f"Warning: {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        print(f"Warning: Failed to configure Gemini: {e}")
        return {
            "cell_id": cell_id,
            "channel": channel,
            "cell_mask_score": None,
            "nucleus_mask_score": None,
            "issues": [f"Gemini configuration failed: {str(e)}"],
            "suggested_ops": []
        }
    
    # Load images
    try:
        raw_image_path = Path(raw_image_path)
        overlay_image_path = Path(overlay_image_path)
        
        if not raw_image_path.exists():
            raise FileNotFoundError(f"Raw image not found: {raw_image_path}")
        if not overlay_image_path.exists():
            raise FileNotFoundError(f"Overlay image not found: {overlay_image_path}")
        
        raw_image_bytes = load_image_file(raw_image_path)
        overlay_image_bytes = load_image_file(overlay_image_path)
        
    except Exception as e:
        print(f"Warning: Failed to load images for Gemini QC: {e}")
        return {
            "cell_id": cell_id,
            "channel": channel,
            "cell_mask_score": None,
            "nucleus_mask_score": None,
            "issues": [f"Image loading failed: {str(e)}"],
            "suggested_ops": []
        }
    
    # Prepare content for Gemini
    try:
        # Determine MIME types
        raw_mime = "image/png" if raw_image_path.suffix.lower() == '.png' else "image/jpeg"
        overlay_mime = "image/png" if overlay_image_path.suffix.lower() == '.png' else "image/jpeg"
        
        # Create content parts: prompt + images
        # For google-generativeai, images should be passed using genai.types.Part
        # or as PIL Images, or as base64-encoded strings
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Convert bytes to PIL Images
        raw_image = Image.open(BytesIO(raw_image_bytes))
        overlay_image = Image.open(BytesIO(overlay_image_bytes))
        
        # Create content parts
        content_parts = [
            GEMINI_PROMPT,
            raw_image,
            overlay_image
        ]
        
        # Generate content with error handling
        try:
            # Log which model we're using for debugging
            if selected_model_name:
                print(f"    Calling Gemini API with model: {selected_model_name}")
            response = model.generate_content(content_parts)
            
            # Check if response has text
            if not hasattr(response, 'text') or not response.text:
                raise Exception("Empty response from Gemini API")
            
            # Parse response
            response_text = response.text
            parsed = parse_gemini_json_response(response_text)
            
            # Add metadata
            parsed["cell_id"] = cell_id
            parsed["channel"] = channel
            
            return parsed
            
        except Exception as api_error:
            # If API call fails, try to get more details
            error_msg = str(api_error)
            if "404" in error_msg:
                error_msg = f"Model not found (404). Try a different model name. Original: {error_msg}"
            elif "403" in error_msg:
                error_msg = f"API key invalid or quota exceeded (403). Original: {error_msg}"
            elif "429" in error_msg:
                error_msg = f"Rate limit exceeded (429). Original: {error_msg}"
            
            print(f"Warning: Gemini API call failed for {cell_id} {channel}: {error_msg}")
            return {
                "cell_id": cell_id,
                "channel": channel,
                "cell_mask_score": None,
                "nucleus_mask_score": None,
                "issues": [f"API call failed: {error_msg}"],
                "suggested_ops": []
            }
        
    except Exception as e:
        print(f"Warning: Gemini processing failed for {cell_id} {channel}: {e}")
        return {
            "cell_id": cell_id,
            "channel": channel,
            "cell_mask_score": None,
            "nucleus_mask_score": None,
            "issues": [f"Processing failed: {str(e)}"],
            "suggested_ops": []
        }

