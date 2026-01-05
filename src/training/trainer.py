"""Training loop and trainer class for caption generation."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm

from ..models.caption_model import EncoderDecoderModel
from ..data.vocabulary import Vocabulary
from .loss import CaptionLoss, PerplexityLoss
from .metrics import CaptionMetrics
from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import TrainingLogger, MetricsTracker


class CaptionTrainer:
    """Trainer for image captioning model.
    
    Handles:
    - Training loop with teacher forcing
    - Validation with metric evaluation
    - Checkpointing and early stopping
    - Sample caption generation
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: EncoderDecoderModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocabulary: Vocabulary,
        config: Dict[str, Any],
        device: str = 'cuda',
        output_dir: str = 'outputs/training_runs',
        resume_checkpoint: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Caption model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            vocabulary: Vocabulary for decoding
            config: Training configuration dictionary
            device: Device to train on
            output_dir: Directory for outputs
            resume_checkpoint: Path to checkpoint to resume from
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        self.device = device
        
        # Create output directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        variant = config['paths']['preprocessing_variant']
        self.output_dir = Path(output_dir) / f"{variant}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_loss()
        self._initialize_metrics()
        
        # Checkpointing and logging
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir / 'checkpoints'),
            max_checkpoints=5
        )
        
        self.logger = TrainingLogger(
            output_dir=str(self.output_dir),
            config=config,
            resume=(resume_checkpoint is not None)
        )
        
        # Early stopping tracker
        early_stop_config = config['training']['early_stopping']
        self.metrics_tracker = MetricsTracker(
            metric_name=early_stop_config['metric'],
            mode=early_stop_config['mode'],
            patience=early_stop_config['patience']
        )
        
        # Mixed precision training
        self.use_amp = (
            config['training'].get('mixed_precision', False) and 
            device.startswith('cuda')
        )
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if specified
        if resume_checkpoint:
            self._resume_from_checkpoint(resume_checkpoint)
    
    def _initialize_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['type'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler."""
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'] == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config['mode'],
                patience=sched_config['patience'],
                factor=sched_config['factor'],
                min_lr=sched_config.get('min_lr', 0)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['type']}")
    
    def _initialize_loss(self):
        """Initialize loss functions."""
        self.criterion = CaptionLoss(
            vocab_size=len(self.vocabulary),
            pad_idx=self.vocabulary.PAD_IDX,
            label_smoothing=self.config['training'].get('label_smoothing', 0.0)
        )
        self.perplexity_fn = PerplexityLoss(pad_idx=self.vocabulary.PAD_IDX)
    
    def _initialize_metrics(self):
        """Initialize metrics calculator."""
        self.metrics_calculator = CaptionMetrics()
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config['training']['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Batch size: {self.train_loader.batch_size}")
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            val_metrics = self._validate_epoch(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.log_epoch_metrics(
                epoch=epoch + 1,
                step=self.global_step,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"  Val BLEU-4: {val_metrics.get('bleu_4', 0):.4f}, METEOR: {val_metrics.get('meteor', 0):.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
            
            # Check if best model
            combined_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            is_best = self.metrics_tracker.update(epoch + 1, combined_metrics)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                step=self.global_step,
                metrics=combined_metrics,
                is_best=is_best
            )
            
            # Check early stopping
            if self.config['training']['early_stopping']['enabled'] and self.metrics_tracker.should_stop:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best {self.metrics_tracker.metric_name}: {self.metrics_tracker.best_value:.4f} at epoch {self.metrics_tracker.best_epoch}")
                break
        
        # Finalize training
        best_info = self.metrics_tracker.get_best_info()
        self.logger.finalize(
            final_epoch=epoch + 1,
            final_step=self.global_step,
            best_epoch=best_info['best_epoch'],
            best_metric_name=best_info['metric_name'],
            best_metric_value=best_info['best_value'],
            final_metrics=val_metrics
        )
        
        print(f"\nTraining complete!")
        print(f"Best model: epoch {best_info['best_epoch']}, {best_info['metric_name']}={best_info['best_value']:.4f}")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for images, captions, caption_lengths, _, _ in progress_bar:
            # Move to device
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lengths = caption_lengths.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Model outputs: predictions, attention_weights, sorted_captions, sorted_lengths
                # Teacher forcing: use ground truth tokens as decoder input
                predictions, attention_weights, sorted_captions, sorted_lengths = self.model(
                    images, captions[:, :-1], caption_lengths - 1
                )
                
                # Calculate loss
                # Decoder returns predictions for actual max length (not padded max)
                # Truncate targets to match predictions length
                # predictions shape: [batch_size, actual_max_len, vocab_size]
                # targets shape: [batch_size, actual_max_len]
                seq_len = predictions.size(1)
                targets = sorted_captions[:, 1:1+seq_len]  # Skip <START>, match predictions length
                loss = self.criterion(predictions, targets)
                
                # Calculate perplexity
                perplexity = self.perplexity_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('gradient_clip_norm', 5.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('gradient_clip_norm', 5.0)
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{perplexity.item():.2f}'
            })
        
        # Return average metrics
        return {
            'loss': total_loss / num_batches,
            'perplexity': total_perplexity / num_batches
        }
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Collect references and hypotheses for metrics
        all_references = []
        all_hypotheses = []
        sample_outputs = []
        
        num_samples_to_generate = self.config['training'].get('num_val_samples_to_generate', 10)
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for batch_idx, (images, captions, caption_lengths, image_paths, uids) in enumerate(progress_bar):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lengths = caption_lengths.to(self.device)
            
            # Calculate loss (teacher forcing)
            # Model returns: predictions, attention_weights, sorted_captions, sorted_lengths
            predictions, attention_weights, sorted_captions, sorted_lengths = self.model(
                images, captions[:, :-1], caption_lengths - 1
            )
            # Truncate targets to match predictions length
            seq_len = predictions.size(1)
            targets = sorted_captions[:, 1:1+seq_len]
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Generate captions (beam search)
            for i in range(len(images)):
                # Generate caption
                # Note: method is implicit - beam_size=1 for greedy, >1 for beam search
                generated_ids, _ = self.model.generate_caption(
                    images[i:i+1],
                    max_length=self.config['inference']['decoding']['max_length'],
                    beam_size=self.config['inference']['decoding']['beam_size']
                )
                
                # Decode reference
                reference_tokens = [
                    self.vocabulary.idx_to_token.get(idx, '<UNK>')
                    for idx in captions[i].cpu().tolist()
                    if idx not in [self.vocabulary.PAD_IDX, self.vocabulary.START_IDX]
                ]
                if self.vocabulary.END_TOKEN in reference_tokens:
                    reference_tokens = reference_tokens[:reference_tokens.index(self.vocabulary.END_TOKEN)]
                
                # Decode hypothesis
                # generated_ids is already a list of ints, not nested
                hypothesis_tokens = [
                    self.vocabulary.idx_to_token.get(idx, '<UNK>')
                    for idx in generated_ids
                    if idx not in [self.vocabulary.PAD_IDX, self.vocabulary.START_IDX]
                ]
                if self.vocabulary.END_TOKEN in hypothesis_tokens:
                    hypothesis_tokens = hypothesis_tokens[:hypothesis_tokens.index(self.vocabulary.END_TOKEN)]
                
                all_references.append([reference_tokens])  # List of lists for multiple references
                all_hypotheses.append(hypothesis_tokens)
                
                # Save samples for logging
                if len(sample_outputs) < num_samples_to_generate:
                    sample_outputs.append({
                        'uid': int(uids[i]),
                        'reference': ' '.join(reference_tokens),
                        'generated': ' '.join(hypothesis_tokens)
                    })
        
        # Calculate metrics
        metrics = self.metrics_calculator.compute_all_metrics(all_references, all_hypotheses)
        metrics['loss'] = total_loss / num_batches
        
        # Save sample outputs
        self.logger.save_sample_outputs(epoch + 1, sample_outputs)
        
        return metrics
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.start_epoch = checkpoint_info['epoch']
        self.global_step = checkpoint_info['step']
        
        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")


if __name__ == '__main__':
    print("Trainer module created successfully!")
    print("\nTo use the trainer:")
    print("1. Load config from model_config.yaml")
    print("2. Create model, dataloaders, vocabulary")
    print("3. Initialize CaptionTrainer")
    print("4. Call trainer.train()")
