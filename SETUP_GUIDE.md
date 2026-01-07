# Setup Guide - Medical Image Captioning

Complete installation and configuration guide for the Medical Image Captioning project.

**Last Updated**: December 27, 2025
**Maintained By**: Maria Linhares & Miguel Silva

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Project Structure](#project-structure)
5. [Data Setup](#data-setup)
6. [Troubleshooting](#troubleshooting)
7. [Testing](#testing)
8. [Development Workflow](#development-workflow)

---

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or later
- **RAM**: 8GB (16GB recommended for training)
- **Storage**: 5GB for dataset + 2GB for checkpoints

### GPU Requirements (Optional but Recommended)
- **NVIDIA**: GTX 1650 (4GB) or better, CUDA 11.8+
- **AMD**: RX 5000 series or better, ROCm 6.3+
- **Apple**: M1/M2/M3/M4 chips with macOS 12.3+

### Software Dependencies
- Git
- Python 3.10+ with pip
- CUDA Toolkit (for NVIDIA GPUs)
- ROCm (for AMD GPUs)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/miguel-silva48/MCIM_Final_Project.git
cd MCIM_Final_Project
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate      # Linux/macOS
# OR
.venv\Scripts\activate          # Windows
```

### 3. Install PyTorch

#### Option A: Interactive Script (Recommended)

```bash
# Make script executable
chmod +x setup_pytorch.sh

# Run interactive installer
./setup_pytorch.sh
```

The script will:
1. Auto-detect your hardware (NVIDIA/AMD/Apple Silicon/CPU)
2. Install the appropriate PyTorch build
3. Install all training dependencies
4. Verify the installation

#### Option B: Manual Installation

##### NVIDIA GPU (CUDA 12.x/13.x)
```bash
pip install --upgrade torch torchvision
```

##### NVIDIA GPU (CUDA 11.x)
```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

##### AMD GPU (ROCm 6.4)
```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

##### Apple Silicon (M1/M2/M3/M4)
```bash
pip install --upgrade torch torchvision
```

##### CPU Only
```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Python Dependencies

```bash
# EDA phase dependencies
pip install -r requirements_eda.txt

# Or install manually:
pip install pandas numpy matplotlib seaborn pillow nltk pyyaml tqdm jupyter ipython ipykernel

# Additional training dependencies (if not using setup_pytorch.sh)
pip install rouge-score scikit-learn
```

### 5. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 6. Verify Installation

```bash
# Quick verification
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Full test suite
python3 test_data_pipeline.py
```

---

## Configuration

### Configuration Files

All settings are in `configs/` directory:

#### `configs/data_config.yaml` - Data Processing
```yaml
# Data source configuration
projection_strategy: 'first_frontal'  # Prevents patient duplicates
text_source: 'impression'             # Use impression (not findings)
max_censoring_ratio: 0.3              # Filter reports with >30% XXXX tokens
min_vocab_freq: 5                     # Vocabulary size control → 514 tokens

# Data splits (patient-level)
splits:
  train: 0.80
  val: 0.10
  test: 0.10

# Preprocessing
max_caption_length: 50               # Truncate longer captions
normalization:
  mean: [0.485, 0.456, 0.406]        # ImageNet statistics
  std: [0.229, 0.224, 0.225]
```

#### `configs/model_config.yaml` - Model & Training
```yaml
# Encoder configuration
encoder:
  architecture: 'densenet121'         # DenseNet-121 backbone
  pretrained: true                    # Use ImageNet weights
  freeze_backbone: true               # Freeze all except last layer
  feature_dim: 1024                   # Output feature dimension

# Decoder configuration
decoder:
  embedding_dim: 512                  # Word embedding size
  hidden_dim: 1024                    # LSTM hidden size
  num_layers: 1                       # LSTM layers
  dropout: 0.5                        # Dropout rate

# Attention mechanism
attention:
  type: 'bahdanau'                    # Additive attention
  attention_dim: 512                  # Attention layer dimension

# Training configuration
training:
  batch_size: auto                    # Auto-adjust based on GPU memory
  learning_rate: 1e-4                 # Adam learning rate
  num_epochs: 30                      # Maximum epochs
  gradient_clip: 5.0                  # Gradient clipping threshold
  teacher_forcing_ratio: 0.5          # Use ground truth 50% of time
  
  # Optimizer
  optimizer: 'adam'
  optimizer_params:
    betas: [0.9, 0.999]
    eps: 1e-8
  
  # Learning rate scheduler
  scheduler: 'reduce_on_plateau'
  scheduler_params:
    mode: 'min'
    factor: 0.5
    patience: 3
    min_lr: 1e-6
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    metric: 'val_loss'
    mode: 'min'
  
  # Mixed precision training
  mixed_precision: true               # Auto-disabled on MPS/CPU
  
  # Checkpointing
  checkpoint:
    save_every: 1                     # Save every N epochs
    keep_last: 3                      # Keep last 3 checkpoints
    save_best: true                   # Save best model

# Inference configuration
inference:
  method: 'beam_search'               # greedy | beam_search
  beam_size: 3                        # Beam width (if beam_search)
  max_length: 50                      # Max caption length
  length_penalty: 0.0                 # Length penalty factor
```

### Customizing Configuration

To modify settings:

1. **Edit YAML files directly** (recommended for permanent changes)
2. **Override in notebooks** (for experiments)
   ```python
   from src.utils.config import load_config
   config = load_config('configs/model_config.yaml')
   config['training']['learning_rate'] = 5e-5
   ```

---

## Project Structure

```
MCIM_Final_Project/
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── data/                       # Data loading and processing
│   │   ├── __init__.py
│   │   ├── collate.py              # Custom collate function for batching variable-length captions
│   │   ├── data_loader.py          # CSV loading and merging (indiana_projections + indiana_reports)
│   │   ├── dataset.py              # PyTorch Dataset for X-ray images and captions
│   │   ├── ngram_analysis.py       # N-gram extraction and analysis
│   │   ├── preprocessing.py        # Main preprocessing pipeline (filtering, splitting, vocab building)
│   │   ├── text_preprocessing.py   # Text cleaning, normalization, and censoring detection
│   │   ├── transforms.py           # Image augmentations (random rotations, crops, flips)
│   │   └── vocabulary.py           # Tokenization, vocabulary building with special tokens
│   │
│   ├── models/                     # Model architecture
│   │   ├── __init__.py
│   │   ├── attention.py            # Bahdanau additive attention mechanism
│   │   ├── caption_model.py        # Complete encoder-decoder model with attention
│   │   ├── decoder.py              # LSTM decoder with attention
│   │   └── encoder.py              # DenseNet-121 CNN encoder (pretrained on ImageNet)
│   │
│   ├── training/                   # Training infrastructure
│   │   ├── __init__.py
│   │   ├── loss.py                 # Cross-entropy loss with padding ignore
│   │   ├── metrics.py              # BLEU, METEOR, ROUGE-L metrics computation
│   │   └── trainer.py              # Training loop with validation, early stopping, checkpointing
│   │
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── checkpoint.py           # Model checkpointing and loading (PyTorch 2.6+ compatible)
│   │   ├── config_loader.py        # YAML configuration loader with type conversion
│   │   ├── data_paths.py           # Environment-aware path management (Kaggle vs Local)
│   │   ├── device_check.py         # GPU detection and device selection (CUDA/MPS/CPU)
│   │   ├── environment.py          # Kaggle/Local/Colab environment detection
│   │   └── logging_utils.py        # Logging utilities and manifest generation
│   │
│   └── visualization/              # Plotting functions
│       ├── __init__.py
│       ├── eda_plots.py            # EDA visualizations (distributions, word clouds, projections)
│       └── training_viz.py         # Training visualizations (loss curves, attention maps)
│
├── scripts/                        # Utility scripts
│   └── preprocess_all_variants.py  # Batch preprocessing for all 8 variant combinations
│
├── configs/                        # Configuration files
│   ├── data_config.yaml            # Data processing config (projection, text_source, vocab)
│   └── model_config.yaml           # Model & training config (architecture, hyperparameters)
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Data preprocessing walkthrough
│   ├── 03_training.ipynb           # Main training notebook (first_frontal + min_freq=5)
│   ├── 03_training_frontal3.ipynb  # Training with first_frontal + min_freq=3
│   ├── 03_training_pairs3.ipynb    # Training with pairs strategy + min_freq=3
│   └── 03_training_pairs5.ipynb    # Training with pairs strategy + min_freq=5
│
├── data/                           # Dataset (images not in github)
│   ├── indiana_projections.csv     # Image metadata
│   ├── indiana_reports.csv         # Medical reports
│   ├── images/
│   │   └── images_normalized/      # 7,466 X-ray images (PNG format)
│   └── processed/                  # Generated preprocessing outputs
│       ├── first_frontal_impression_3/   # variant folder
│       ├── first_frontal_impression_5/   # variant folder
│       ├── first_frontal_findings_3/     # variant folder
│       ├── first_frontal_findings_5/     # variant folder
│       ├── pairs_impression_3/           # variant folder
│       ├── pairs_impression_5/           # variant folder
│       ├── pairs_findings_3/             # variant folder
│       └── pairs_findings_5/             # variant folder
│           ├── train.csv                 # Training split with image paths and captions
│           ├── val.csv                   # Validation split
│           ├── test.csv                  # Test split
│           ├── vocabulary.txt            # Vocabulary tokens (one per line)
│           └── preprocessing_manifest.json  # Metadata about preprocessing run
│
├── outputs/                        # Training outputs (not in git)
│   └── training_runs/
│       └── {variant}_{timestamp}/
│           ├── training_manifest.json    # Training configuration and results
│           ├── metrics.csv               # Epoch-by-epoch metrics
│           ├── sample_outputs/           # Sample generated captions during training
│           └── checkpoints/              # Model checkpoints (.pth files)
│
├── docs/                           # Documentation (empty or archived)
│
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── README.md                       # Main project documentation
├── SETUP_GUIDE.md                  # This file - complete installation guide
├── setup_pytorch.sh                # Interactive PyTorch installer script
├── requirements_eda.txt            # EDA-only dependencies (pandas, matplotlib, seaborn, etc.)
└── requirements_kaggle.txt         # Kaggle-compatible dependencies (excludes pre-installed packages)
```

---

## Data Setup

### Expected Data Structure

Place the Indiana dataset in this structure:

```
data/
├── indiana_projections.csv
├── indiana_reports.csv
└── images/
    └── images_normalized/
        ├── 1.png
        ├── 2.png
        └── ... (7,466 total)
```

### Download Dataset

#### Local Environment

**Option 1: Kaggle CLI** (recommended)
```bash
# Install Kaggle CLI if not already installed
pip install kaggle

# Download dataset (downloads as archive.zip)
kaggle datasets download -d raddar/chest-xrays-indiana-university

# Extract to data/ directory
unzip archive.zip -d data/
```

**Option 2: Manual Download**
1. Visit https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
2. Click "Download" (downloads as `archive.zip`)
3. Extract to project's `data/` directory:
   ```bash
   unzip ~/Downloads/archive.zip -d data/
   ```

**Option 3: KaggleHub** (requires API authentication)
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
print("Path to dataset files:", path)
```

#### Kaggle Notebook Environment

In Kaggle notebooks, **do not download** - instead link the dataset:

1. Open your Kaggle notebook
2. Click "+ Add Data" in the right sidebar
3. Search for "chest-xrays-indiana-university"
4. Click "Add" to link the dataset
5. Dataset will be available at: `/kaggle/input/chest-xrays-indiana-university/`

The training notebook (`03_training.ipynb`) automatically handles Kaggle paths.

### Preprocessing

Run preprocessing notebook to generate splits:

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

This creates:
- `data/processed/first_frontal_impression/train.csv`
- `data/processed/first_frontal_impression/val.csv`
- `data/processed/first_frontal_impression/test.csv`
- `data/processed/first_frontal_impression/vocabulary.txt`

---

## Troubleshooting

### PyTorch Installation Issues

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi  # Should show GPU info

# Check PyTorch CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# If False, reinstall PyTorch
pip uninstall torch torchvision
./setup_pytorch.sh
```

#### Apple Silicon MPS Not Available
```bash
# Check macOS version (needs 12.3+)
sw_vers

# Check PyTorch MPS
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Update PyTorch if needed
pip install --upgrade torch torchvision
```

#### ROCm Issues (AMD)
```bash
# Check ROCm installation
rocm-smi  # Should show GPU info

# Verify PyTorch ROCm build
python3 -c "import torch; print(torch.version.hip)"

# Reinstall if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

### NLTK Data Issues

#### LookupError: punkt/punkt_tab not found
```bash
# Download all required NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Or download interactively
python3
>>> import nltk
>>> nltk.download()  # Opens download GUI
```

#### NLTK Download Behind Proxy
```bash
# Set proxy environment variables
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080

# Then download
python3 -c "import nltk; nltk.download('punkt')"
```

### Module Import Errors

#### ModuleNotFoundError: No module named 'src'
```bash
# Ensure you're in project root
pwd  # Should end in MCIM_Final_Project

# Activate virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements_eda.txt
```

#### RuntimeWarning: Module found in sys.modules
This is harmless. Occurs when running test code in modules:
```bash
python3 -m src.data.dataset  # Warning appears but tests run fine
```

### Dataset Issues

#### FileNotFoundError: indiana_projections.csv
```bash
# Check data directory structure
ls -R data/

# Should show:
# data/indiana_projections.csv
# data/indiana_reports.csv
# data/images/images_normalized/*.png
```

#### Images not found
```bash
# Check image path
ls data/images/images_normalized/ | head

# Should list: 1.png, 2.png, etc.
```

### Jupyter Kernel Issues

#### Kernel not found
```bash
# Install ipykernel in your venv
python -m ipykernel install --user --name=mcim-venv --display-name="Python (MCIM)"

# Restart Jupyter
jupyter notebook
# Select "Python (MCIM)" kernel
```

#### Kernel dies during training
- **Cause**: Out of memory
- **Solution**: Reduce batch size in `configs/model_config.yaml`
  ```yaml
  training:
    batch_size: 4  # Reduce if OOM
  ```

### Memory Issues

#### CUDA Out of Memory
```python
# In notebook or config, reduce batch size
batch_size = 4  # For 4GB GPUs
batch_size = 8  # For 8GB GPUs
batch_size = 16 # For 16GB GPUs

# Enable gradient accumulation
gradient_accumulation_steps = 4  # Effective batch = 4 × 4 = 16
```

#### RAM Issues (CPU)
```bash
# Monitor memory usage
htop  # Linux
top   # macOS

# Reduce workers in DataLoader
num_workers = 0  # No multiprocessing
```

---

## Testing

### Quick Tests

```bash
# Test PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Test CUDA/MPS
python3 -m src.utils.device_check

# Test data loading
python3 -c "from src.data.data_loader import load_data; df = load_data(); print(f'Loaded {len(df)} samples')"
```

---

## Development Workflow

### Local Development

1. **Activate environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Make changes** to source code

3. **Test changes**
   ```bash
   python3 -m src.data.dataset  # Test specific module
   python3 test_data_pipeline.py  # Test all
   ```

4. **Commit to Git**
   ```bash
   git add .
   git commit -m "Add feature X"
   git push
   ```

### Kaggle Development

1. **Upload notebook** to Kaggle

2. **Add dataset**
   - Add "Indiana Chest X-Rays" dataset

3. **Clone repo in notebook**
   ```python
   !git clone https://github.com/miguel-silva48/MCIM_Final_Project.git
   %cd MCIM_Final_Project/notebooks
   ```

4. **Install dependencies**
   ```python
   !pip install -r kaggle_requirements.txt
   ```

5. **Run training**
  > Start running the cells—you're ready to train!

---

## Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

### Papers
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) - Attention mechanism
- [Neural Machine Translation](https://arxiv.org/abs/1409.0473) - Bahdanau attention
- [DenseNet](https://arxiv.org/abs/1608.06993) - Encoder architecture
