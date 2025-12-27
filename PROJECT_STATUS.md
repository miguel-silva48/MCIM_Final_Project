# Project Status - Medical Image Captioning

Detailed development tracking for the Medical Image Captioning project.

**Last Updated**: December 27, 2025
**Current Phase**: Training Execution  
**Team**: Maria Linhares & Miguel Silva

---

## Overview

```
Timeline: Nov 2025 - Jan 2026
Progress: ████████████████░░░░ 80%
Status: Training infrastructure complete, ready for full training
```

---

## Phase Breakdown

### ✅ Phase 1: EDA & Preprocessing (COMPLETE)

**Status**: 100% complete, reviewed

#### Deliverables
- [x] Exploratory data analysis notebook
- [x] Patient-level train/val/test splits
- [x] Censoring detection and removal
- [x] Vocabulary building (514 tokens)
- [x] First-frontal projection strategy
- [x] Preprocessing manifest generation

#### Key Findings
- **Dataset**: 7,470 images from 3,855 patients
- **Ideal pairs** (1 frontal + 1 lateral): ~2,500 patients
- **Multiple frontals**: ~300 patients (handled by first-frontal strategy)
- **Censoring**: 3 reports have >30% XXXX tokens (filtered)
- **Vocabulary**: 514 tokens (min_freq=5) includes medical terms

#### Decisions Made
1. **First-frontal only** - Prevents patient data leakage
2. **Impression as target** - More concise than findings
3. **30% censoring threshold** - Balances quality vs quantity
4. **Patient-level splits** - 80/10/10 train/val/test

#### Files Created
- `notebooks/01_eda.ipynb` - Full EDA
- `notebooks/02_preprocessing.ipynb` - Data splitting
- `data/processed/first_frontal_impression/` - Splits + vocab
- `src/data/data_loader.py` - Data loading utilities
- `src/data/text_preprocessing.py` - Text cleaning
- `src/visualization/eda_plots.py` - Plotting functions

---

### ✅ Phase 2: Data Pipeline (COMPLETE)

**Status**: 100% complete, tested

#### Deliverables
- [x] PyTorch Dataset implementation
- [x] Medical-appropriate image transforms
- [x] Vocabulary encoding/decoding
- [x] Custom collate function for variable-length captions
- [x] Multi-device support (CUDA/ROCm/MPS/CPU)
- [x] Comprehensive unit tests

#### Technical Details
- **Image size**: 224×224 (DenseNet-121 input)
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation**: RandomRotation(10°), ColorJitter - NO horizontal flip
- **Tokenization**: <PAD>=0, <START>=1, <END>=2, <UNK>=3, vocab tokens 4-517
- **Batch collation**: Pads captions to max length in batch

#### Files Created
- `src/data/dataset.py` - ChestXrayDataset
- `src/data/transforms.py` - get_train_transforms(), get_val_transforms()
- `src/data/vocabulary.py` - Vocabulary class
- `test_data_pipeline.py` - Test suite

#### Testing Results
```
✅ Dataset loads 3,684 samples
✅ Images normalized correctly
✅ Augmentation works (no horizontal flip)
✅ Vocabulary encodes/decodes correctly
✅ Collate function handles variable lengths
✅ Works on CUDA, MPS, and CPU
```

---

### ✅ Phase 3: Model Architecture (COMPLETE)

**Status**: 100% complete, tested

#### Deliverables
- [x] DenseNet-121 encoder (ImageNet pretrained)
- [x] LSTM decoder with embeddings
- [x] Bahdanau attention mechanism
- [x] Complete encoder-decoder model
- [x] Loss functions (CrossEntropy, Perplexity)
- [x] Evaluation metrics (BLEU-1/2/3/4, METEOR, ROUGE-L)
- [x] Unit tests for all components

#### Model Specifications

**Encoder (DenseNet-121)**
- Input: 224×224×3 RGB images
- Output: 1024-dim features (7×7 spatial)
- Parameters: 7.9M (1.0M trainable when frozen)
- Pretrained: ImageNet weights
- Frozen: All layers except last feature layer

**Attention (Bahdanau)**
- Type: Additive attention
- Attention dim: 512
- Input: Encoder features (7×7×1024) + Decoder hidden (1024)
- Output: Context vector (1024) + Attention weights (7×7)
- Parameters: 1.6M

**Decoder (LSTM)**
- Embedding dim: 512
- Hidden dim: 1024
- Num layers: 1
- Dropout: 0.5
- Parameters: 13.6M
- Input: <START> token or previous word + context vector
- Output: Next word probabilities (514 vocab)

**Total Model**
- Total parameters: 23.2M
- Trainable parameters: 16.2M (encoder frozen)
- FP32 size: ~93 MB
- FP16 size: ~47 MB (with mixed precision)

#### Files Created
- `src/models/encoder.py` - EncoderCNN
- `src/models/attention.py` - BahdanauAttention
- `src/models/decoder.py` - DecoderRNN
- `src/models/caption_model.py` - EncoderDecoderModel
- `src/training/loss.py` - CaptionLoss, PerplexityMetric
- `src/training/metrics.py` - calculate_bleu(), calculate_meteor(), calculate_rouge()

#### Testing Results
```
✅ Encoder outputs 1024-dim features
✅ Attention computes context vectors
✅ Decoder generates captions
✅ Full model forward pass works
✅ Loss computes correctly
✅ Metrics (BLEU/METEOR/ROUGE) work
✅ Model saves/loads checkpoints
```

---

### ✅ Phase 4: Training Infrastructure (COMPLETE)

**Status**: 100% complete, tested

#### Deliverables
- [x] Training loop with teacher forcing
- [x] Checkpointing system (save/load/resume)
- [x] Logging utilities (manifests, metrics CSV)
- [x] Early stopping mechanism
- [x] Learning rate scheduling
- [x] Mixed precision training (FP16 on CUDA)
- [x] Device management (CUDA/ROCm/MPS/CPU)
- [x] Automatic batch size adjustment

#### Training Features

**Training Loop**
- Teacher forcing ratio: 0.5 (use ground truth 50% of time)
- Gradient clipping: 5.0 (prevent exploding gradients)
- Optimizer: Adam (lr=1e-4, betas=[0.9, 0.999])
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)

**Checkpointing**
- Saves every N epochs (default: 1)
- Keeps last K checkpoints (default: 3)
- Saves best model based on metric (default: val_loss)
- Can resume from checkpoint

**Logging**
- Training manifest (JSON): Config, hardware, dataset info
- Metrics CSV: Per-epoch loss, BLEU, METEOR, ROUGE
- Sample outputs: Generated captions per epoch
- Training curves: Loss, metrics over time

**Early Stopping**
- Metric: val_loss (or val_bleu)
- Patience: 10 epochs
- Mode: minimize (for loss) or maximize (for BLEU)

**Hardware Support**
- CUDA: Mixed precision (FP16), batch_size=32 (16GB GPU)
- MPS: FP32 only, batch_size=16 (Apple Silicon)
- CPU: FP32 only, batch_size=4-8 (slow)
- Auto batch sizing: 4GB→4, 8GB→16, 16GB→32

#### Files Created
- `src/training/trainer.py` - Trainer class
- `src/utils/checkpoint.py` - CheckpointManager
- `src/utils/logging_utils.py` - TrainingLogger, MetricsTracker
- `src/utils/device_check.py` - Device detection utilities
- `setup_pytorch.sh` - Multi-device PyTorch installer

#### Testing Results
```
✅ Training loop runs without errors
✅ Checkpoints save/load correctly
✅ Logging creates manifests and CSVs
✅ Early stopping triggers correctly
✅ Mixed precision works on CUDA
✅ Works on all device types
✅ Batch size auto-adjusts correctly
```

---

### 🔄 Phase 5: Training Execution (IN PROGRESS)

**Status**: 20% complete

#### Current Tasks
- [x] Setup training notebook (`03_training.ipynb`)
- [x] Configure training on GTX 1650Ti (4GB)
- [ ] **NEXT**: Run 1-2 epoch test locally (validate setup)
- [ ] Upload to Kaggle, configure for P100/T4 (16GB)
- [ ] Run full training (~20-30 epochs)
- [ ] Monitor metrics, adjust hyperparameters if needed
- [ ] Save best model

#### Expected Timeline
- **Week 1**: Local testing (1-2 epochs on GTX 1650Ti)
- **Week 2**: Kaggle setup, initial training (10 epochs)
- **Week 3**: Continued training, hyperparameter tuning

#### Hardware Plan
1. **Local (GTX 1650Ti, 4GB)**: Initial testing, debugging
   - Batch size: 4-8
   - Expected: ~10-15 min/epoch
   - Purpose: Validate setup works

2. **Kaggle (P100, 16GB)**: Full training
   - Batch size: 32-64
   - Expected: ~3-5 min/epoch
   - Purpose: Train to convergence (~20-30 epochs)

#### Expected Results
- **Training loss**: Should decrease to ~2.0-2.5
- **Validation loss**: Should follow training loss
- **BLEU-4**: Target >0.15 (decent for medical captions)
- **Perplexity**: Target <15 (indicates model confidence)

#### Known Issues
- None currently

---

### ⏳ Phase 6: Evaluation & Analysis (PENDING)

**Status**: 0% complete

#### Planned Tasks
- [ ] Load best model checkpoint
- [ ] Evaluate on test set (368 samples)
- [ ] Calculate comprehensive metrics:
  - [ ] BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - [ ] METEOR
  - [ ] ROUGE-L
  - [ ] Perplexity
- [ ] Generate sample outputs (qualitative analysis)
- [ ] Visualize attention weights
- [ ] Analyze failure cases
- [ ] Compare with baseline/previous work

#### Deliverables
- Evaluation notebook
- Test set metrics (JSON)
- Sample outputs with attention visualization
- Error analysis document

---

### ⏳ Phase 7: Final Presentation (PENDING)

**Status**: 0% complete

#### Planned Tasks
- [ ] Create presentation slides
- [ ] Record demo video (optional)
- [ ] Prepare GitHub README for publication
- [ ] Archive final models

#### Deliverables
- Presentation slides (PPT/PDF)
- Demo video (optional)
- Public GitHub repository

---

## Resources

### Papers Considered
- [x] Show, Attend and Tell (Xu et al., 2015)
- [x] Neural Machine Translation (Bahdanau et al., 2014)
- [x] DenseNet (Huang et al., 2017)
- [ ] CheXNet (Rajpurkar et al., 2017) - for reference

### Datasets
- [x] Indiana University Chest X-Ray Collection

### Tools Used
- PyTorch 2.x
- Jupyter Notebook
- VS Code
- Git/GitHub
- Kaggle (for GPU)

---

**Next Action**: Run 1-2 epoch test to validate training setup and create training notebook.