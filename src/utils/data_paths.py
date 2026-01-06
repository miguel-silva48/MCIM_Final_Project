"""
Data path configuration based on execution environment.
"""

from pathlib import Path
from typing import Dict, Optional
from .environment import is_kaggle


def get_data_paths(
    preprocessing_variant: str = 'first_frontal_impression',
    project_root: Optional[Path] = None,
    kaggle_dataset_name: str = 'chest-xrays-indiana-university'
) -> Dict[str, Path]:
    """
    Get data paths based on execution environment (Kaggle vs. Local).
    
    On Kaggle, data is automatically mounted at /kaggle/input/{kaggle_dataset_name}/
    Locally, data should be in the data/ directory at project root.
    
    Args:
        preprocessing_variant: Name of preprocessing variant (e.g., 'first_frontal_impression')
        project_root: Project root directory (required for local execution, ignored on Kaggle)
        kaggle_dataset_name: Name of Kaggle dataset (default: 'chest-xrays-indiana-university')
    
    Returns:
        Dict[str, Path]: Dictionary containing:
            - data_root: Root directory of the dataset
            - images_dir: Directory containing image files
            - processed_dir: Directory containing processed data splits
            - vocab_file: Path to vocabulary.txt
            - train_csv: Path to train.csv
            - val_csv: Path to val.csv
            - test_csv: Path to test.csv
            - projections_csv: Path to indiana_projections.csv (raw data)
            - reports_csv: Path to indiana_reports.csv (raw data)
    """
    if is_kaggle():
        # Kaggle environment - data is pre-loaded
        data_root = Path('/kaggle/input') / kaggle_dataset_name
        images_dir = data_root / 'images' / 'images_normalized'
        processed_dir = data_root / 'processed' / preprocessing_variant
        projections_csv = data_root / 'indiana_projections.csv'
        reports_csv = data_root / 'indiana_reports.csv'
    else:
        # Local environment - data in project's data/ folder
        if project_root is None:
            # Fallback: try to determine from file location
            project_root = Path(__file__).parents[2]
        
        data_root = project_root / 'data'
        images_dir = data_root / 'images' / 'images_normalized'
        processed_dir = data_root / 'processed' / preprocessing_variant
        projections_csv = data_root / 'indiana_projections.csv'
        reports_csv = data_root / 'indiana_reports.csv'
    
    return {
        'data_root': data_root,
        'image_dir': images_dir,
        'processed_dir': processed_dir,
        'vocab_file': processed_dir / 'vocabulary.txt',
        'train_csv': processed_dir / 'train.csv',
        'val_csv': processed_dir / 'val.csv',
        'test_csv': processed_dir / 'test.csv',
        'projections_csv': projections_csv,
        'reports_csv': reports_csv,
    }


if __name__ == '__main__':
    # Quick test
    paths = get_data_paths()
    print("Data paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
        print(f"    Exists: {path.exists()}")
