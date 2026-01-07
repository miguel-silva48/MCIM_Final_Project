# Medical Image Captioning with Deep Learning

Automated chest X-ray captioning system combining DenseNet-121 encoder with LSTM decoder and attention mechanism. Generates radiological impressions from X-ray images.

**Dataset**: Indiana University Chest X-Ray Collection (7,466 images, 3,851 patients)  
**Model**: 23.2M params (16.2M trainable) | DenseNet-121 + LSTM + Bahdanau Attention  
**Status**: Training infrastructure complete, ready for full training  
**Team**: Maria Linhares & Miguel Silva

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/miguel-silva48/MCIM_Final_Project.git
cd MCIM_Final_Project
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch (auto-detects your hardware)
chmod +x setup_pytorch.sh && ./setup_pytorch.sh

# 3. Install dependencies
pip install -r requirements_eda.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Download dataset (see SETUP_GUIDE.md for details)
kaggle datasets download -d raddar/chest-xrays-indiana-university
unzip archive.zip -d data/

# 5. Run notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_training.ipynb
```

**Having issues?** See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions and troubleshooting.

---

## What This Does

```
Input: Chest X-Ray Image (224×224)
    ↓
DenseNet-121 Encoder (ImageNet pretrained)
    ↓ 1024-dim features
Bahdanau Attention Mechanism
    ↓
LSTM Decoder (512 embed, 1024 hidden)
    ↓
Output: Caption (e.g. "No acute cardiopulmonary abnormality.")
```

### Key Features
- ✅ **Medical-Appropriate**: Patient-level splits, no data leakage, smart augmentation (no horizontal flip!)
- ✅ **Production-Ready**: Multi-device (CUDA/ROCm/MPS/CPU), mixed precision, checkpointing, logging
- ✅ **Comprehensive Metrics**: BLEU-1/2/3/4, METEOR, ROUGE-L
- ✅ **Robust Testing**: Every module has standalone tests

---

## Project Structure

```
MCIM_Final_Project/
├── src/               # All source code (data, models, training, utils)
├── configs/           # YAML configuration files
├── notebooks/         # Jupyter notebooks (EDA, preprocessing, training)
├── data/              # Dataset and processed outputs
│   ├── processed/     # Train/val/test splits, vocabulary
│   └── images/        # 7,466 X-ray images
├── outputs/           # Training runs (checkpoints, metrics, logs)
└── setup_pytorch.sh   # Interactive PyTorch installer
```

**Full structure:** See [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## Model Specifications

- **Encoder**: DenseNet-121 (7.9M params, frozen)
- **Decoder**: LSTM + Attention (15.2M params, trainable)
- **Vocabulary**: 584 tokens (min_freq=5) or 782 tokens (min_freq=3)
- **Training**: Adam (lr=1e-4), teacher forcing, early stopping
- **Inference**: Beam search (beam_size=3)

**Full configuration:** See `configs/model_config.yaml`

---

## Testing

```bash
# Test individual modules 
# (This applies to the vast majority of standalone python scripts)
# (Exceptions are those that agregate functionality from other modules)
# (Below we list only some examples; run tests for all modules as needed)
python3 -m src.data.dataset
python3 -m src.models.caption_model
python3 -m src.training.metrics
```

---

## Other Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup, troubleshooting, configuration
- **[notebooks/](notebooks/)** - Interactive Jupyter notebooks with explanations

---

## Resources

- **Dataset**: [Indiana University Chest X-Ray Collection](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **Architecture**: [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) (Xu et al., 2015)
- **Attention**: [Neural Machine Translation](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)
- **Encoder**: [DenseNet](https://arxiv.org/abs/1608.06993) (Huang et al., 2017)

---

## License

MIT License - See [LICENSE](LICENSE) file for details

---

## Team

**Maria Linhares & Miguel Silva**  
Medical Informatics Course Final Project

Questions? Open an issue or check the [setup guide](SETUP_GUIDE.md).