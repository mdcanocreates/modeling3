"""
Manifest management for generated images.

Tracks all images (originals + generated) with metadata.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd
from pathlib import Path


@dataclass
class ImageRecord:
    """
    Record for a single image (original or generated).
    
    For originals, parent_id == sample_id.
    For generated, parent_id is the original cell ID (CellA, CellB, CellC).
    """
    sample_id: str  # Unique identifier (e.g., "CellA_alg1_001", "CellA_original")
    parent_id: str  # Parent cell ID (CellA, CellB, CellC)
    algorithm: str  # "original", "alg1", or "alg2"
    path_phase: Optional[str] = None  # Path to phase channel (BF)
    path_actin: Optional[str] = None  # Path to actin channel
    path_mt: Optional[str] = None  # Path to microtubules channel
    path_nuc: Optional[str] = None  # Path to nuclei channel
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ImageRecord':
        """Create from dictionary."""
        return cls(**d)


def create_manifest(records: List[ImageRecord], output_path: Path) -> pd.DataFrame:
    """
    Create manifest CSV from list of ImageRecord objects.
    
    Parameters
    ----------
    records : list
        List of ImageRecord objects
    output_path : Path
        Path to save manifest_mc3.csv
    
    Returns
    -------
    pd.DataFrame
        Manifest DataFrame
    """
    data = [r.to_dict() for r in records]
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df


def load_manifest(manifest_path: Path) -> List[ImageRecord]:
    """
    Load manifest from CSV.
    
    Parameters
    ----------
    manifest_path : Path
        Path to manifest_mc3.csv
    
    Returns
    -------
    list
        List of ImageRecord objects
    """
    df = pd.read_csv(manifest_path)
    records = [ImageRecord.from_dict(row.to_dict()) for _, row in df.iterrows()]
    return records

