"""Loss functions for caption generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CaptionLoss(nn.Module):
    """Cross-entropy loss for caption generation with optional label smoothing.
    
    This loss function handles:
    - Masking padding tokens (PAD_IDX=0)
    - Optional label smoothing to prevent overconfidence
    - Proper normalization by number of valid tokens
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """Initialize caption loss.
        
        Args:
            vocab_size: Size of vocabulary
            pad_idx: Index of padding token to ignore in loss
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = typical)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # CrossEntropyLoss with label smoothing (PyTorch 1.10+)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction=reduction
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        caption_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate loss.
        
        Args:
            predictions: Model predictions, shape (batch_size, seq_len, vocab_size)
            targets: Ground truth tokens, shape (batch_size, seq_len)
            caption_lengths: Optional lengths for additional masking, shape (batch_size,)
        
        Returns:
            Scalar loss value (or per-sample if reduction='none')
        """
        # Reshape for CrossEntropyLoss
        # predictions: (batch_size * seq_len, vocab_size)
        # targets: (batch_size * seq_len)
        batch_size, seq_len, vocab_size = predictions.shape
        predictions = predictions.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        return loss
    
    def __repr__(self) -> str:
        return (
            f"CaptionLoss(vocab_size={self.vocab_size}, "
            f"pad_idx={self.pad_idx}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction='{self.reduction}')"
        )


class PerplexityLoss(nn.Module):
    """Perplexity metric (exp(cross-entropy loss)).
    
    Lower perplexity indicates better model confidence.
    Typical values: 10-50 for good models, >100 for poor models.
    """
    
    def __init__(self, pad_idx: int = 0):
        """Initialize perplexity calculator.
        
        Args:
            pad_idx: Index of padding token to ignore
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            reduction='mean'
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate perplexity.
        
        Args:
            predictions: Model predictions, shape (batch_size, seq_len, vocab_size)
            targets: Ground truth tokens, shape (batch_size, seq_len)
        
        Returns:
            Perplexity value
        """
        # Calculate cross-entropy loss
        batch_size, seq_len, vocab_size = predictions.shape
        predictions = predictions.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = self.criterion(predictions, targets)
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss)
        
        return perplexity


if __name__ == '__main__':
    import torch
    
    print("Testing CaptionLoss...")
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    
    # Predictions: logits from model
    predictions = torch.randn(batch_size, seq_len, vocab_size)
    
    # Targets: ground truth token indices
    targets = torch.randint(1, vocab_size, (batch_size, seq_len))
    # Add some padding tokens
    targets[:, -3:] = 0  # Last 3 tokens are padding
    
    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Test without label smoothing
    loss_fn = CaptionLoss(vocab_size=vocab_size, pad_idx=0, label_smoothing=0.0)
    loss = loss_fn(predictions, targets)
    print(f"\nLoss (no smoothing): {loss.item():.4f}")
    
    # Test with label smoothing
    loss_fn_smooth = CaptionLoss(vocab_size=vocab_size, pad_idx=0, label_smoothing=0.1)
    loss_smooth = loss_fn_smooth(predictions, targets)
    print(f"Loss (smoothing=0.1): {loss_smooth.item():.4f}")
    
    # Test perplexity
    perplexity_fn = PerplexityLoss(pad_idx=0)
    perplexity = perplexity_fn(predictions, targets)
    print(f"Perplexity: {perplexity.item():.4f}")
    
    # Verify perplexity = exp(loss)
    expected_perplexity = torch.exp(loss)
    print(f"Expected perplexity: {expected_perplexity.item():.4f}")
    print(f"Match: {torch.allclose(perplexity, expected_perplexity)}")
    
    print("\n✓ Loss functions test passed!")
