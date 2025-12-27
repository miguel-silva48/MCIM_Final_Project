# Medical Image Captioning with Deep Learning

Automated chest X-ray captioning system combining DenseNet-121 encoder with LSTM decoder and attention mechanism. Generates radiological impressions from X-ray images.

**Dataset**: Indiana University Chest X-Ray Collection (~7,470 images, ~3,855 patients)  
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

# 4. Run notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
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
Output: "No acute cardiopulmonary abnormality."
```

### Key Features
- ✅ **Medical-Appropriate**: Patient-level splits, no data leakage, smart augmentation (no horizontal flip!)
- ✅ **Production-Ready**: Multi-device (CUDA/ROCm/MPS/CPU), mixed precision, checkpointing, logging
- ✅ **Comprehensive Metrics**: BLEU-1/2/3/4, METEOR, ROUGE-L
- ✅ **Robust Testing**: Every module has standalone tests

---

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Data Pipeline | ✅ Complete | Dataset, transforms, patient-level splits |
| Model Architecture | ✅ Complete | Encoder-decoder with attention |
| Training Infrastructure | ✅ Complete | Metrics, checkpointing, logging |
| Training Execution | 🔄 In Progress | Testing on GTX 1650Ti, then Kaggle |
| Evaluation | ⏳ Next | Performance analysis, visualization |

**See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed progress tracking.**

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
- **Vocabulary**: 514 tokens (min_freq=5)
- **Training**: Adam (lr=1e-4), teacher forcing, early stopping
- **Inference**: Beam search (beam_size=3)

**Full configuration:** See `configs/model_config.yaml`

---

## Testing

```bash
# Test all components
python3 test_data_pipeline.py

# Test individual modules
python3 -m src.data.dataset
python3 -m src.models.caption_model
python3 -m src.training.metrics
```

---

## Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup, troubleshooting, configuration
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Development progress, what's next
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