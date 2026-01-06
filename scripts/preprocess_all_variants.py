#!/usr/bin/env python3
"""
Preprocess all combinations of preprocessing variants.

Generates 8 preprocessing variants:
- Projection strategies: first_frontal, pairs
- Text sources: impression, findings
- Min vocab frequencies: 3, 5

Output folders: {projection}_{text_source}_{min_freq}/
"""

import sys
from pathlib import Path
import yaml
import itertools

# Add src to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from src.data.preprocessing import preprocess_dataset


def preprocess_all_variants():
    """Run preprocessing for all combinations of parameters."""
    
    # Define all variants
    projection_strategies = ['first_frontal', 'pairs']
    text_sources = ['impression', 'findings']
    min_vocab_freqs = [3, 5]
    
    # Base config path
    config_path = project_root / 'configs' / 'data_config.yaml'
    output_dir = project_root / 'data' / 'processed'
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate all combinations
    variants = list(itertools.product(projection_strategies, text_sources, min_vocab_freqs))
    total = len(variants)
    
    print("=" * 80)
    print("PREPROCESSING ALL VARIANTS")
    print("=" * 80)
    print(f"Total variants to process: {total}")
    print()
    
    results = []
    
    for i, (proj_strategy, text_source, min_freq) in enumerate(variants, 1):
        variant_name = f"{proj_strategy}_{text_source}_{min_freq}"
        
        print(f"\n[{i}/{total}] Processing: {variant_name}")
        print("-" * 80)
        
        # Update config
        config = base_config.copy()
        config['projection_strategy'] = proj_strategy
        config['text_source'] = text_source
        config['min_vocab_freq'] = min_freq
        
        # Create temporary config file
        temp_config_path = project_root / 'configs' / f'temp_config_{variant_name}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run preprocessing
            manifest = preprocess_dataset(str(temp_config_path), str(output_dir))
            
            # Store results
            results.append({
                'variant': variant_name,
                'projection_strategy': proj_strategy,
                'text_source': text_source,
                'min_vocab_freq': min_freq,
                'patients': manifest['dataset_statistics']['splits']['train']['patients'] + 
                           manifest['dataset_statistics']['splits']['val']['patients'] + 
                           manifest['dataset_statistics']['splits']['test']['patients'],
                'vocab_size': manifest['vocabulary_statistics']['vocab_size'],
                'status': 'SUCCESS'
            })
            
            print(f"✓ {variant_name} completed successfully")
            print(f"  Patients: {results[-1]['patients']}")
            print(f"  Vocab size: {results[-1]['vocab_size']}")
            
        except Exception as e:
            print(f"✗ {variant_name} failed: {str(e)}")
            results.append({
                'variant': variant_name,
                'projection_strategy': proj_strategy,
                'text_source': text_source,
                'min_vocab_freq': min_freq,
                'status': f'FAILED: {str(e)}'
            })
        
        finally:
            # Clean up temp config
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    print(f"\nTotal variants: {total}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successful variants:")
        for r in successful:
            print(f"  • {r['variant']:30s} | Patients: {r['patients']:4d} | Vocab: {r['vocab_size']:4d}")
    
    if failed:
        print("\n✗ Failed variants:")
        for r in failed:
            print(f"  • {r['variant']:30s} | {r['status']}")
    
    print("\n" + "=" * 80)
    print("All preprocessing variants completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    preprocess_all_variants()
