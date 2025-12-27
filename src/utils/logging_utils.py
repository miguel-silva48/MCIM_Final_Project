"""Logging utilities for training runs."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd


class TrainingLogger:
    """Logger for training runs with manifest generation.
    
    Maintains same structure as preprocessing phase:
    - Training manifest (JSON) with config and metadata
    - Metrics CSV with per-epoch results
    - Sample outputs during training
    """
    
    def __init__(
        self,
        output_dir: str,
        config: Dict[str, Any],
        resume: bool = False
    ):
        """Initialize training logger.
        
        Args:
            output_dir: Directory for logging outputs
            config: Training configuration dictionary
            resume: Whether resuming from checkpoint
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.resume = resume
        
        # File paths
        self.manifest_path = self.output_dir / 'training_manifest.json'
        self.metrics_csv_path = self.output_dir / 'metrics.csv'
        self.samples_dir = self.output_dir / 'sample_outputs'
        self.samples_dir.mkdir(exist_ok=True)
        
        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Initialize manifest
        if not resume:
            self._initialize_manifest()
    
    def _initialize_manifest(self):
        """Create initial training manifest."""
        manifest = {
            'run_info': {
                'start_time': datetime.now().isoformat(),
                'output_directory': str(self.output_dir),
                'resume': self.resume
            },
            'config': self.config,
            'training_summary': {
                'total_epochs': 0,
                'total_steps': 0,
                'best_epoch': None,
                'best_metric': None,
                'final_metrics': {}
            },
            'hardware': {
                'device': 'unknown',  # Will be updated
                'gpu_name': 'unknown',
                'gpu_memory_gb': 0
            }
        }
        
        # Save initial manifest
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def update_hardware_info(self, device_info: Dict[str, Any]):
        """Update hardware information in manifest.
        
        Args:
            device_info: Dictionary with device information
        """
        manifest = self._load_manifest()
        manifest['hardware'] = device_info
        self._save_manifest(manifest)
    
    def log_epoch_metrics(
        self,
        epoch: int,
        step: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ):
        """Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            step: Global training step
            train_metrics: Training metrics (loss, perplexity, etc.)
            val_metrics: Validation metrics (BLEU, METEOR, etc.)
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch (seconds)
        """
        # Combine metrics
        epoch_data = {
            'epoch': epoch,
            'step': step,
            'learning_rate': learning_rate,
            'epoch_time_seconds': epoch_time,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        
        # Append to history
        self.metrics_history.append(epoch_data)
        
        # Save to CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.metrics_csv_path, index=False)
    
    def save_sample_outputs(
        self,
        epoch: int,
        samples: List[Dict[str, str]]
    ):
        """Save sample captions from model.
        
        Args:
            epoch: Current epoch
            samples: List of dicts with 'uid', 'reference', 'generated' keys
        """
        sample_file = self.samples_dir / f'epoch_{epoch:03d}_samples.json'
        
        with open(sample_file, 'w') as f:
            json.dump(samples, f, indent=2)
    
    def finalize(
        self,
        final_epoch: int,
        final_step: int,
        best_epoch: int,
        best_metric_name: str,
        best_metric_value: float,
        final_metrics: Dict[str, float]
    ):
        """Finalize training manifest with summary.
        
        Args:
            final_epoch: Final epoch reached
            final_step: Final training step
            best_epoch: Epoch with best validation metric
            best_metric_name: Name of metric used for model selection
            best_metric_value: Best metric value achieved
            final_metrics: Final evaluation metrics
        """
        manifest = self._load_manifest()
        
        # Update training summary
        manifest['training_summary'] = {
            'total_epochs': final_epoch,
            'total_steps': final_step,
            'best_epoch': best_epoch,
            'best_metric': {
                'name': best_metric_name,
                'value': best_metric_value
            },
            'final_metrics': final_metrics
        }
        
        # Add completion time
        manifest['run_info']['end_time'] = datetime.now().isoformat()
        
        # Calculate training duration
        start_time = datetime.fromisoformat(manifest['run_info']['start_time'])
        end_time = datetime.fromisoformat(manifest['run_info']['end_time'])
        duration = (end_time - start_time).total_seconds()
        manifest['run_info']['duration_seconds'] = duration
        manifest['run_info']['duration_hours'] = duration / 3600
        
        self._save_manifest(manifest)
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load training manifest."""
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
    
    def _save_manifest(self, manifest: Dict[str, Any]):
        """Save training manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def __repr__(self) -> str:
        return f"TrainingLogger(output_dir={self.output_dir})"


class MetricsTracker:
    """Track metrics during training for early stopping and checkpointing."""
    
    def __init__(
        self,
        metric_name: str = 'val_bleu_4',
        mode: str = 'max',
        patience: int = 5,
        min_delta: float = 0.0001
    ):
        """Initialize metrics tracker.
        
        Args:
            metric_name: Name of metric to track
            mode: 'max' for metrics where higher is better, 'min' for lower is better
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update tracker with new metrics.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        
        Returns:
            True if this is the best model so far, False otherwise
        """
        if self.metric_name not in metrics:
            raise ValueError(f"Metric '{self.metric_name}' not found in metrics")
        
        current_value = metrics[self.metric_name]
        
        # Check if improved
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model."""
        return {
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'metric_name': self.metric_name
        }
    
    def __repr__(self) -> str:
        return (
            f"MetricsTracker(metric={self.metric_name}, mode={self.mode}, "
            f"best={self.best_value:.4f} at epoch {self.best_epoch})"
        )


if __name__ == '__main__':
    import tempfile
    import time
    
    print("Testing TrainingLogger...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nTest directory: {tmpdir}")
        
        # Create config
        config = {
            'model': {'encoder': 'densenet121', 'decoder': 'lstm'},
            'training': {'batch_size': 32, 'learning_rate': 0.0001}
        }
        
        # Initialize logger
        logger = TrainingLogger(
            output_dir=os.path.join(tmpdir, 'training_run'),
            config=config,
            resume=False
        )
        
        # Update hardware info
        logger.update_hardware_info({
            'device': 'cuda',
            'gpu_name': 'GTX 1650Ti',
            'gpu_memory_gb': 4
        })
        
        print("\nLogging epochs...")
        # Log a few epochs
        for epoch in range(1, 4):
            train_metrics = {
                'loss': 2.5 - epoch * 0.3,
                'perplexity': 12.0 - epoch * 2
            }
            val_metrics = {
                'bleu_4': epoch * 0.15,
                'meteor': epoch * 0.12,
                'rouge_l': epoch * 0.18
            }
            
            logger.log_epoch_metrics(
                epoch=epoch,
                step=epoch * 100,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=0.0001,
                epoch_time=125.5
            )
            
            # Save sample outputs
            samples = [
                {
                    'uid': 1,
                    'reference': 'normal chest x ray',
                    'generated': 'normal chest radiograph'
                }
            ]
            logger.save_sample_outputs(epoch, samples)
            
            print(f"  Epoch {epoch}: loss={train_metrics['loss']:.3f}, BLEU-4={val_metrics['bleu_4']:.3f}")
        
        # Finalize
        logger.finalize(
            final_epoch=3,
            final_step=300,
            best_epoch=3,
            best_metric_name='val_bleu_4',
            best_metric_value=0.45,
            final_metrics={'bleu_4': 0.45, 'meteor': 0.36}
        )
        
        print(f"\nOutputs created:")
        print(f"  Manifest: {logger.manifest_path.exists()}")
        print(f"  Metrics CSV: {logger.metrics_csv_path.exists()}")
        print(f"  Sample outputs: {len(list(logger.samples_dir.glob('*.json')))}")
        
        # Test MetricsTracker
        print("\n" + "=" * 50)
        print("Testing MetricsTracker...")
        
        tracker = MetricsTracker(
            metric_name='val_bleu_4',
            mode='max',
            patience=2
        )
        
        # Simulate improving then plateauing
        for epoch in range(1, 6):
            metrics = {'val_bleu_4': 0.2 + epoch * 0.05 if epoch < 4 else 0.35}
            is_best = tracker.update(epoch, metrics)
            
            print(f"  Epoch {epoch}: BLEU={metrics['val_bleu_4']:.3f}, is_best={is_best}, should_stop={tracker.should_stop}")
        
        print(f"\nBest: {tracker.get_best_info()}")
        
        print("\n✓ Logging utilities test passed!")
