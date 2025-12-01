"""
Quick test script to verify the project structure and imports.
"""
import sys
from pathlib import Path

print("=" * 60)
print("TESTING PROJECT SETUP")
print("=" * 60)

# Test 1: Import utilities
print("\n1. Testing utility imports...")
try:
    from src.utils.environment import get_execution_env
    from src.utils.data_paths import get_data_paths
    env = get_execution_env()
    paths = get_data_paths()
    print(f"   ✓ Environment: {env}")
    print(f"   ✓ Data root: {paths['data_root']}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import data modules
print("\n2. Testing data module imports...")
try:
    from src.data.data_loader import load_raw_data, merge_projections_reports
    from src.data.text_preprocessing import (
        extract_report_text, 
        calculate_censoring_ratio,
        tokenize_text
    )
    from src.data.ngram_analysis import extract_ngrams
    print("   ✓ All data modules imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Import visualization
print("\n3. Testing visualization imports...")
try:
    from src.visualization.eda_plots import (
        plot_patient_image_distribution,
        plot_text_length_distributions
    )
    print("   ✓ Visualization modules imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Check config file
print("\n4. Testing configuration loading...")
try:
    import yaml
    config_path = Path(__file__).parent / "configs" / "data_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Config loaded: projection_strategy={config['projection_strategy']}")
    print(f"   ✓ Text source: {config['text_source']}")
    print(f"   ✓ Min vocab freq: {config['min_vocab_freq']}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 5: Check data directory
print("\n5. Checking data directories...")
try:
    from src.utils.data_paths import get_data_paths
    paths = get_data_paths()
    
    print(f"   • Data directory exists: {paths['data_root'].exists()}")
    print(f"   • Images directory exists: {paths['images_dir'].exists()}")
    print(f"   • Projections CSV exists: {paths['projections_csv'].exists()}")
    print(f"   • Reports CSV exists: {paths['reports_csv'].exists()}")
    
    if paths['projections_csv'].exists() and paths['reports_csv'].exists():
        print("   ✓ Dataset files found - ready for EDA!")
        # Count images
        if paths['images_dir'].exists():
            image_count = len(list(paths['images_dir'].glob('*.png')))
            print(f"   ✓ Found {image_count} images")
    else:
        print("   ⚠ Dataset files not found - please ensure data is in data/ directory")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Test basic text preprocessing
print("\n6. Testing text preprocessing functions...")
try:
    from src.data.text_preprocessing import clean_metadata_text, calculate_censoring_ratio
    
    test_text = "INDICATION: xxxx year old with cough. FINDINGS: Clear lungs."
    cleaned = clean_metadata_text(test_text)
    censoring = calculate_censoring_ratio(test_text)
    
    print(f"   ✓ Cleaned text: {cleaned[:50]}...")
    print(f"   ✓ Censoring ratio: {censoring:.2%}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Setup complete!")
print("=" * 60)
print("\nNext steps:")
print("  1. Ensure dataset is in data/ directory")
print("  2. Download NLTK data if needed: python -c \"import nltk; nltk.download('punkt_tab')\"")
print("  3. Run: jupyter notebook notebooks/01_eda.ipynb")
print("  4. Execute all cells to generate analysis")
