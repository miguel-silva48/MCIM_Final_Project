"""
Data path configuration based on execution environment.
"""

from pathlib import Path
from typing import Dict
from .environment import is_kaggle


def get_data_paths() -> Dict[str, Path]:
    """
    Get data paths based on execution environment (Kaggle vs. Local).
    
    On Kaggle, data is automatically mounted at /kaggle/input/chest-xrays-indiana-university/
    Locally, data should be in the data/ directory at project root.
    
    Returns:
        Dict[str, Path]: Dictionary containing:
            - data_root: Root directory of the dataset
            - images_dir: Directory containing image files
            - projections_csv: Path to indiana_projections.csv
            - reports_csv: Path to indiana_reports.csv
    """
    if is_kaggle():
        # Kaggle environment - data is pre-loaded
        data_root = Path('/kaggle/input/chest-xrays-indiana-university')
        images_dir = data_root / 'images' / 'images_normalized'
        projections_csv = data_root / 'indiana_projections.csv'
        reports_csv = data_root / 'indiana_reports.csv'
    else:
        # Local environment - data in project's data/ folder
        # Assumes script is run from project root or notebooks/
        data_root = Path(__file__).parents[2] / 'data'
        images_dir = data_root / 'images' / 'images_normalized'
        projections_csv = data_root / 'indiana_projections.csv'
        reports_csv = data_root / 'indiana_reports.csv'
    
    return {
        'data_root': data_root,
        'images_dir': images_dir,
        'projections_csv': projections_csv,
        'reports_csv': reports_csv
    }


if __name__ == '__main__':
    # Quick test
    paths = get_data_paths()
    print("Data paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
        print(f"    Exists: {path.exists()}")
