# Medical Image Captioning - Setup Guide

## Initial Setup Complete ✅

This commit includes the complete initial structure for the Medical Image Captioning project with a focus on the Exploratory Data Analysis (EDA) phase.

## Project Status

**Phase 1: EDA Infrastructure** ✅ COMPLETE - NEEDS REVIEW
- Modular code structure implemented
- Environment-aware data loading (Kaggle/Local)
- Comprehensive text preprocessing utilities
- N-gram analysis tools
- Visualization functions
- Configuration management

**Phase 2: Model Development** 🔄 PENDING
- PyTorch Dataset implementation
- CheXNet encoder integration
- LSTM/Transformer decoder
- Training pipeline

## Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install EDA requirements (no PyTorch needed yet)
pip install -r requirements_eda.txt

# Download NLTK data (including punkt_tab for newer NLTK versions)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Verify Setup

```bash
# Run verification tests
python test_setup.py
```

Expected output:
- ✓ Environment detection working
- ✓ All modules import successfully
- ✓ Configuration loads correctly
- ✓ Dataset files found (7466 images)
- ✓ Text preprocessing functional

### 3. Run EDA

```bash
# Option A: Open in VS Code
code notebooks/01_eda.ipynb

# Option B: Launch Jupyter
jupyter notebook notebooks/01_eda.ipynb
```

The notebook includes 10 sections:
1. **Environment Setup** - Auto-detects Kaggle vs Local, clones repo if needed
2. **Data Loading** - Loads and merges projections + reports CSVs
3. **Patient-Image Analysis** - Analyzes distribution patterns (frontal/lateral counts)
4. **Image Visualization** - Displays sample X-rays with reports
5. **Text Analysis** - Compares findings vs impression lengths
6. **Censoring Analysis** - Detects and filters heavily censored reports (>30% XXXX)
7. **N-gram Analysis** - Extracts unigrams, bigrams, trigrams
8. **Vocabulary Building** - Tests min_freq thresholds [3, 4, 5]
9. **Summary & Recommendations** - Data quality insights
10. **Export Summary** - Saves findings to JSON

## Project Structure

```
MCIM_Final_Project/
├── src/
│   ├── utils/
│   │   ├── environment.py          # Kaggle/Local detection
│   │   └── data_paths.py           # Environment-aware paths
│   ├── data/
│   │   ├── data_loader.py          # CSV loading, patient analysis
│   │   ├── text_preprocessing.py   # Tokenization, vocab building
│   │   └── ngram_analysis.py       # N-gram extraction
│   └── visualization/
│       └── eda_plots.py            # All EDA plots
├── configs/
│   └── data_config.yaml            # Centralized configuration
├── notebooks/
│   └── 01_eda.ipynb                # Comprehensive EDA notebook
├── data/
│   ├── indiana_projections.csv     # Image metadata (uid, filename, projection)
│   ├── indiana_reports.csv         # Reports (uid, findings, impression)
│   ├── images/images_normalized/   # 7466 PNG X-ray images
│   ├── processed/                  # Generated: vocabulary.txt
│   └── reports/                    # Generated: plots, n-gram CSVs, summary.json
├── requirements_eda.txt            # EDA phase dependencies
├── requirements_training.txt       # Training phase dependencies (TBD later)
└── test_setup.py                   # Verification script
```

## Configuration

All data processing decisions are in `configs/data_config.yaml`:

```yaml
projection_strategy: 'first_frontal'  # Prevents data leakage
text_source: 'impression'             # Target caption field
max_censoring_ratio: 0.3              # Filter threshold
min_vocab_freq: 5                     # Vocabulary size control
splits:                               # Data splits (holdout methodology)
  train: 0.80
  val: 0.10
  test: 0.10
```

## Design Decisions

### 1. **First Frontal Only** (`projection_strategy='first_frontal'`)
- **Problem**: Some patients have multiple frontal views (e.g., uid 3549: 2 frontals + 1 lateral)
- **Solution**: Use only the first frontal image per patient
- **Benefit**: Enables patient-level train/val/test split without data leakage

### 2. **Impression as Target** (`text_source='impression'`)
- **Findings**: Longer, more detailed observations (~100-150 words)
- **Impression**: Concise clinical conclusion (~20-50 words)
- **Choice**: Impression has less censoring and is more suitable for caption generation

### 3. **Censoring Threshold** (`max_censoring_ratio=0.3`)
- Indiana dataset uses "XXXX" to anonymize patient identifiers
- Reports with >30% XXXX tokens are filtered out
- Balances data retention with caption quality

### 4. **Environment-Aware Loading**
- Code automatically detects Kaggle vs Local execution
- No manual path changes needed for collaboration
- Enables seamless Kaggle notebook sharing

## Data Statistics (Expected)

From the dataset:
- **Images**: 7,466 PNG files (frontal and lateral chest X-rays)
- **Patients**: 3,851 unique patients
- **Patterns**:
  - Ideal pairs (1 frontal + 1 lateral): ~2,500 patients
  - Multiple frontals: ~300 patients
  - Frontal only: ~800 patients
  - Lateral only: ~200 patients

## EDA Outputs

Running the notebook generates:
- `data/reports/unigrams_top100.csv` - Most frequent single words
- `data/reports/bigrams_top100.csv` - Most frequent word pairs
- `data/reports/trigrams_top100.csv` - Most frequent 3-word sequences
- `data/reports/*.png` - Various analysis plots
- `data/processed/vocabulary.txt` - Built vocabulary with special tokens
- `data/reports/eda_summary.json` - Complete analysis summary

## Next Steps After EDA

1. **Review EDA outputs** to understand data quality
2. **Commit initial structure** to GitHub
3. **Implement data splitting** (train/val/test by patient uid)
4. **Create PyTorch Dataset** for image loading + tokenization
5. **Build model architecture**:
   - CheXNet encoder (DenseNet-121 pretrained)
   - LSTM/Transformer decoder with attention
6. **Set up training pipeline** on Kaggle GPU
7. **Implement evaluation metrics** (BLEU, METEOR, ROUGE, CIDEr)

## Testing

Verify everything works:
```bash
python test_setup.py
```

All 6 tests should pass:
- ✅ Utility imports
- ✅ Data module imports  
- ✅ Visualization imports
- ✅ Configuration loading
- ✅ Dataset verification
- ✅ Text preprocessing

## Troubleshooting

### Import Errors
```bash
# Ensure you're in virtual environment
source .venv/bin/activate
pip install -r requirements_eda.txt
```

### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Jupyter Kernel Issues
```bash
# Install ipykernel in your venv
python -m ipykernel install --user --name=mcim-venv
# Then select "mcim-venv" as kernel in notebook
```

### Dataset Not Found
Ensure your data structure matches:
```
data/
├── indiana_projections.csv
├── indiana_reports.csv
└── images/
    └── images_normalized/
        ├── 1.png
        ├── 2.png
        └── ...
```

## Resources - WIP

- **Dataset**: Indiana University Chest X-Ray Dataset (Kaggle)
- **CheXNet Paper**: Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection"
- **Architecture**: Encoder-Decoder with Attention
- **Frameworks**: PyTorch, NLTK, Pandas
