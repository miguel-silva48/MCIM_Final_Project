"""
Data loading and patient-level analysis functions.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def load_raw_data(data_paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV files (projections and reports).
    
    Args:
        data_paths: Dictionary containing paths to CSV files
        
    Returns:
        Tuple of (projections_df, reports_df)
    """
    projections_df = pd.read_csv(data_paths['projections_csv'])
    reports_df = pd.read_csv(data_paths['reports_csv'])
    
    return projections_df, reports_df


def merge_projections_reports(
    projections_df: pd.DataFrame,
    reports_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge projections and reports DataFrames on patient uid.
    
    Args:
        projections_df: DataFrame with image projections
        reports_df: DataFrame with radiology reports
        
    Returns:
        Merged DataFrame with both image and report information
    """
    # Merge on uid
    merged_df = projections_df.merge(reports_df, on='uid', how='left')
    
    return merged_df


def analyze_patient_image_counts(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how many images each patient has (frontal vs. lateral).
    
    Args:
        merged_df: Merged DataFrame with projections and reports
        
    Returns:
        DataFrame with columns: uid, n_frontal, n_lateral, total_images
    """
    # Group by patient uid and projection type
    projection_counts = merged_df.groupby(['uid', 'projection']).size().unstack(fill_value=0)
    
    # Ensure both Frontal and Lateral columns exist
    if 'Frontal' not in projection_counts.columns:
        projection_counts['Frontal'] = 0
    if 'Lateral' not in projection_counts.columns:
        projection_counts['Lateral'] = 0
    
    # Create summary DataFrame
    patient_counts = pd.DataFrame({
        'uid': projection_counts.index,
        'n_frontal': projection_counts['Frontal'].values,
        'n_lateral': projection_counts['Lateral'].values,
    })
    
    patient_counts['total_images'] = patient_counts['n_frontal'] + patient_counts['n_lateral']
    
    return patient_counts


def get_projection_statistics(patient_counts_df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate statistics about patient projection patterns.
    
    Args:
        patient_counts_df: DataFrame from analyze_patient_image_counts
        
    Returns:
        Dictionary with statistics:
            - total_patients: Total number of unique patients
            - ideal_pairs: Patients with exactly 1 frontal + 1 lateral
            - multiple_frontals: Patients with >1 frontal image
            - multiple_laterals: Patients with >1 lateral image
            - frontal_only: Patients with only frontal images
            - lateral_only: Patients with only lateral images
            - single_image_only: Patients with exactly 1 image (any projection)
    """
    stats = {
        'total_patients': len(patient_counts_df),
        'ideal_pairs': len(patient_counts_df[
            (patient_counts_df['n_frontal'] == 1) & 
            (patient_counts_df['n_lateral'] == 1)
        ]),
        'multiple_frontals': len(patient_counts_df[patient_counts_df['n_frontal'] > 1]),
        'multiple_laterals': len(patient_counts_df[patient_counts_df['n_lateral'] > 1]),
        'frontal_only': len(patient_counts_df[
            (patient_counts_df['n_frontal'] > 0) & 
            (patient_counts_df['n_lateral'] == 0)
        ]),
        'lateral_only': len(patient_counts_df[
            (patient_counts_df['n_frontal'] == 0) & 
            (patient_counts_df['n_lateral'] > 0)
        ]),
        'single_image_only': len(patient_counts_df[patient_counts_df['total_images'] == 1]),
    }
    
    return stats


if __name__ == '__main__':
    # Quick test with local paths
    from src.utils import get_data_paths
    
    paths = get_data_paths()
    projections_df, reports_df = load_raw_data(paths)
    
    print(f"Projections shape: {projections_df.shape}")
    print(f"Reports shape: {reports_df.shape}")
    
    merged_df = merge_projections_reports(projections_df, reports_df)
    print(f"Merged shape: {merged_df.shape}")
    
    patient_counts = analyze_patient_image_counts(merged_df)
    print(f"\nPatient counts shape: {patient_counts.shape}")
    print(patient_counts.head(10))
    
    stats = get_projection_statistics(patient_counts)
    print("\nProjection statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
