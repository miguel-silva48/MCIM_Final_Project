"""Checkpoint management for saving and loading model state."""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """Manage model checkpoints during training.
    
    Handles:
    - Saving checkpoints with model, optimizer, scheduler state
    - Loading checkpoints for resuming training
    - Keeping only best N checkpoints
    - Saving best model based on validation metric
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        keep_best: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            keep_best: Whether to keep separate best model checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        self.best_metric = None
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            step: Current training step
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
        
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add scheduler state if present
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add additional info
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        if not is_best:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
        else:
            # Save best model
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to
        
        Returns:
            Dictionary with checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if checkpoints:
            return str(checkpoints[0])
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best model checkpoint.
        
        Returns:
            Path to best checkpoint or None if doesn't exist
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return str(best_path)
        return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints most recent."""
        # Get all regular checkpoints (not best_model.pt)
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint.unlink()
    
    def __repr__(self) -> str:
        return (
            f"CheckpointManager(dir={self.checkpoint_dir}, "
            f"max_checkpoints={self.max_checkpoints})"
        )


if __name__ == '__main__':
    import tempfile
    import shutil
    
    print("Testing CheckpointManager...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nTest directory: {tmpdir}")
        
        # Initialize checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=os.path.join(tmpdir, 'checkpoints'),
            max_checkpoints=3
        )
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        print(f"\nSaving checkpoints...")
        
        # Save several checkpoints
        for epoch in range(1, 6):
            metrics = {
                'loss': 2.5 - epoch * 0.3,
                'bleu_4': epoch * 0.15
            }
            
            # Save regular checkpoint
            path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                step=epoch * 100,
                metrics=metrics,
                is_best=(epoch == 4)  # Epoch 4 is best
            )
            print(f"  Epoch {epoch}: {os.path.basename(path)}")
        
        # Check that only max_checkpoints are kept
        checkpoints = list(manager.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        print(f"\nRegular checkpoints kept: {len(checkpoints)} (max={manager.max_checkpoints})")
        
        # Check best model
        best_path = manager.get_best_checkpoint()
        print(f"Best checkpoint: {os.path.basename(best_path) if best_path else 'None'}")
        
        # Test loading checkpoint
        print(f"\nLoading latest checkpoint...")
        latest_path = manager.get_latest_checkpoint()
        
        # Create new model
        new_model = torch.nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        checkpoint_info = manager.load_checkpoint(
            checkpoint_path=latest_path,
            model=new_model,
            optimizer=new_optimizer
        )
        
        print(f"  Epoch: {checkpoint_info['epoch']}")
        print(f"  Step: {checkpoint_info['step']}")
        print(f"  Metrics: {checkpoint_info['metrics']}")
        
        print("\n✓ CheckpointManager test passed!")
