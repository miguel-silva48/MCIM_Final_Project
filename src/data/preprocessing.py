"""
Data preprocessing and splitting for Medical Image Captioning.

This module handles:
- Filtering censored reports
- Patient-level data splitting (train/val/test)
- Vocabulary building on training data only
- Generating preprocessing manifest
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
import yaml
from collections import Counter

from .text_preprocessing import (
    calculate_censoring_ratio,
    remove_censoring_tokens,
    filter_reports_by_censoring,
    build_vocabulary,
    save_vocabulary,
    tokenize_text
)
from .data_loader import load_raw_data, merge_projections_reports
from ..utils.data_paths import get_data_paths


def apply_first_frontal_strategy(
    merged_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Keep only the first frontal image per patient.
    
    Args:
        merged_df: DataFrame with merged projections and reports
        
    Returns:
        Filtered DataFrame with first frontal only per patient
    """
    # Filter for frontal images only
    frontal_df = merged_df[merged_df['projection'] == 'Frontal'].copy()
    
    # Sort by uid and filename to ensure consistent "first" selection
    frontal_df = frontal_df.sort_values(['uid', 'filename'])
    
    # Keep first frontal per patient
    first_frontal_df = frontal_df.groupby('uid').first().reset_index()
    
    return first_frontal_df


def split_data_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by patient ID to prevent data leakage.
    
    Args:
        df: DataFrame with patient data
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get unique patient IDs
    unique_patients = df['uid'].unique()
    n_patients = len(unique_patients)
    
    # Shuffle patients
    np.random.seed(random_seed)
    shuffled_patients = np.random.permutation(unique_patients)
    
    # Calculate split indices
    train_size = int(n_patients * train_ratio)
    val_size = int(n_patients * val_ratio)
    
    # Split patient IDs
    train_patients = set(shuffled_patients[:train_size])
    val_patients = set(shuffled_patients[train_size:train_size + val_size])
    test_patients = set(shuffled_patients[train_size + val_size:])
    
    # Split dataframe
    train_df = df[df['uid'].isin(train_patients)].copy()
    val_df = df[df['uid'].isin(val_patients)].copy()
    test_df = df[df['uid'].isin(test_patients)].copy()
    
    # Add split column for tracking
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    return train_df, val_df, test_df


def preprocess_dataset(
    config_path: Path,
    output_dir: Path
) -> Dict:
    """
    Run complete preprocessing pipeline.
    
    Args:
        config_path: Path to data_config.yaml
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary with preprocessing statistics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("MEDICAL IMAGE CAPTIONING - DATA PREPROCESSING")
    print("="*60)
    
    # Step 1: Load raw data
    print("\n[1/7] Loading raw data...")
    data_paths = get_data_paths()
    projections_df, reports_df = load_raw_data(data_paths)
    merged_df = merge_projections_reports(projections_df, reports_df)
    print(f"  ✓ Loaded {len(merged_df)} image-report pairs")
    print(f"  ✓ Unique patients: {merged_df['uid'].nunique()}")
    
    # Step 2: Apply projection strategy
    print(f"\n[2/7] Applying projection strategy: {config['projection_strategy']}")
    if config['projection_strategy'] == 'first_frontal':
        processed_df = apply_first_frontal_strategy(merged_df)
    else:
        raise NotImplementedError(f"Strategy '{config['projection_strategy']}' not implemented yet")
    print(f"  ✓ Selected {len(processed_df)} images (one per patient)")
    
    # Step 3: Calculate censoring ratios
    print(f"\n[3/7] Analyzing text censoring...")
    text_source = config['text_source']
    processed_df['censoring_ratio'] = processed_df[text_source].apply(calculate_censoring_ratio)
    print(f"  ✓ Mean censoring ratio: {processed_df['censoring_ratio'].mean():.3f}")
    print(f"  ✓ Median censoring ratio: {processed_df['censoring_ratio'].median():.3f}")
    
    # Step 4: Filter by censoring threshold
    print(f"\n[4/7] Filtering reports (max_censoring_ratio={config['max_censoring_ratio']})...")
    before_count = len(processed_df)
    processed_df = processed_df[processed_df['censoring_ratio'] <= config['max_censoring_ratio']].copy()
    removed_count = before_count - len(processed_df)
    print(f"  ✓ Removed {removed_count} heavily censored reports ({removed_count/before_count*100:.1f}%)")
    print(f"  ✓ Remaining: {len(processed_df)} patients")
    
    # Step 5: Patient-level data splitting
    print(f"\n[5/7] Splitting data (patient-level)...")
    splits = config['splits']
    train_df, val_df, test_df = split_data_by_patient(
        processed_df,
        train_ratio=splits['train'],
        val_ratio=splits['val'],
        test_ratio=splits['test'],
        random_seed=config['random_seed']
    )
    print(f"  ✓ Train: {len(train_df)} patients")
    print(f"  ✓ Val:   {len(val_df)} patients")
    print(f"  ✓ Test:  {len(test_df)} patients")
    
    # Step 6: Build vocabulary on TRAINING DATA ONLY
    print(f"\n[6/7] Building vocabulary (training data only, min_freq={config['min_vocab_freq']})...")
    train_texts_raw = train_df[text_source].dropna().tolist()
    
    # Remove censoring tokens (XXXX, x-XXXX, standalone x) before building vocabulary
    print(f"  • Removing censoring tokens (XXXX, x-XXXX, etc.)...")
    train_texts_cleaned = [remove_censoring_tokens(text) for text in train_texts_raw]
    
    # Verify censoring token removal
    xxxx_in_cleaned = sum(1 for text in train_texts_cleaned if 'xxxx' in text.lower())
    if xxxx_in_cleaned > 0:
        print(f"  ⚠ Warning: {xxxx_in_cleaned} texts still contain 'xxxx' after cleaning")
    else:
        print(f"  ✓ All censoring tokens removed from {len(train_texts_cleaned)} training texts")
    
    preprocessing_config = config['text_preprocessing']
    token_counter, vocab = build_vocabulary(
        train_texts_cleaned,
        min_freq=config['min_vocab_freq'],
        lowercase=preprocessing_config['lowercase'],
        remove_punctuation=preprocessing_config['remove_punctuation']
    )
    
    print(f"  ✓ Vocabulary size: {len(vocab)} tokens")
    print(f"  ✓ Total unique tokens in training: {len(token_counter)}")
    
    # Check for 'xxxx' token (should not be present after cleaning)
    if 'xxxx' in vocab:
        xxxx_count = token_counter.get('xxxx', 0)
        print(f"  ⚠ ERROR: 'xxxx' token found {xxxx_count} times in vocabulary!")
        print(f"    This should not happen after censoring token removal.")
    else:
        print(f"  ✓ Confirmed: No 'xxxx' token in vocabulary")
    
    # Step 7: Save outputs
    print(f"\n[7/7] Saving processed data...")
    
    # Create variant-specific subdirectory
    # Format: data/processed/{projection_strategy}_{text_source}/
    variant_name = f"{config['projection_strategy']}_{text_source}"
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  • Preprocessing variant: {variant_name}")
    print(f"  • Output directory: {variant_dir}")
    
    # Save splits
    train_path = variant_dir / 'train.csv'
    val_path = variant_dir / 'val.csv'
    test_path = variant_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  ✓ Saved train.csv")
    print(f"  ✓ Saved val.csv")
    print(f"  ✓ Saved test.csv")
    
    # Save vocabulary
    vocab_path = variant_dir / 'vocabulary.txt'
    save_vocabulary(vocab, token_counter, vocab_path)
    print(f"  ✓ Saved vocabulary.txt")
    
    # Generate manifest
    manifest = {
        'preprocessing_info': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'variant': variant_name,
            'projection_strategy': config['projection_strategy'],
            'text_source': text_source,
            'config_used': config
        },
        'dataset_statistics': {
            'raw_data': {
                'total_image_report_pairs': len(merged_df),
                'unique_patients': merged_df['uid'].nunique()
            },
            'after_projection_strategy': {
                'selected_images': before_count,
                'unique_patients': before_count
            },
            'after_censoring_filter': {
                'remaining_patients': len(processed_df),
                'removed_patients': removed_count,
                'removal_rate': f"{removed_count/before_count*100:.2f}%"
            },
            'splits': {
                'train': {
                    'patients': len(train_df),
                    'percentage': f"{len(train_df)/len(processed_df)*100:.2f}%",
                    'patient_uids': sorted(train_df['uid'].unique().tolist())
                },
                'val': {
                    'patients': len(val_df),
                    'percentage': f"{len(val_df)/len(processed_df)*100:.2f}%",
                    'patient_uids': sorted(val_df['uid'].unique().tolist())
                },
                'test': {
                    'patients': len(test_df),
                    'percentage': f"{len(test_df)/len(processed_df)*100:.2f}%",
                    'patient_uids': sorted(test_df['uid'].unique().tolist())
                }
            }
        },
        'vocabulary_statistics': {
            'vocab_size': len(vocab),
            'total_unique_tokens_in_training': len(token_counter),
            'min_freq_threshold': config['min_vocab_freq'],
            'special_tokens': ['<PAD>', '<START>', '<END>', '<UNK>'],
            'coverage': f"{sum(count for token, count in token_counter.items() if token in vocab) / sum(token_counter.values()) * 100:.2f}%",
            'contains_xxxx_token': 'xxxx' in vocab
        },
        'text_statistics': {
            'train': {
                'avg_text_length_words': float(train_df[text_source].dropna().apply(lambda x: len(str(x).split())).mean()),
                'avg_censoring_ratio': float(train_df['censoring_ratio'].mean())
            },
            'val': {
                'avg_text_length_words': float(val_df[text_source].dropna().apply(lambda x: len(str(x).split())).mean()),
                'avg_censoring_ratio': float(val_df['censoring_ratio'].mean())
            },
            'test': {
                'avg_text_length_words': float(test_df[text_source].dropna().apply(lambda x: len(str(x).split())).mean()),
                'avg_censoring_ratio': float(test_df['censoring_ratio'].mean())
            }
        },
        'output_files': {
            'train_csv': str(train_path.name),
            'val_csv': str(val_path.name),
            'test_csv': str(test_path.name),
            'vocabulary': str(vocab_path.name),
            'manifest': 'preprocessing_manifest.json'
        }
    }
    
    manifest_path = variant_dir / 'preprocessing_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Saved preprocessing_manifest.json")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\n📁 All outputs saved to: {variant_dir}")
    print(f"\n📊 Summary:")
    print(f"  • Training patients: {len(train_df)}")
    print(f"  • Validation patients: {len(val_df)}")
    print(f"  • Test patients: {len(test_df)}")
    print(f"  • Vocabulary size: {len(vocab)}")
    print(f"  • No data leakage: ✓ (vocab built on training only)")
    
    return manifest


if __name__ == '__main__':
    # For direct execution
    from pathlib import Path
    
    project_root = Path(__file__).parents[2]
    config_path = project_root / 'configs' / 'data_config.yaml'
    output_dir = project_root / 'data' / 'processed'
    
    manifest = preprocess_dataset(config_path, output_dir)
    print(f"\n✓ Preprocessing manifest saved")
