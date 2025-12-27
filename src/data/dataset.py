"""PyTorch Dataset for chest X-ray images and captions."""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset

from .vocabulary import Vocabulary


class ChestXrayDataset(Dataset):
    """Dataset for chest X-ray images with caption annotations.
    
    Loads preprocessed data from CSV files containing image paths and captions.
    Each sample returns an image tensor, tokenized caption, and metadata.
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        vocabulary: Vocabulary,
        transform: Optional[Callable] = None,
        max_caption_length: int = 50,
        caption_column: str = 'impression'
    ):
        """Initialize dataset.
        
        Args:
            csv_file: Path to CSV file (train.csv, val.csv, or test.csv)
            image_dir: Root directory containing images
            vocabulary: Vocabulary object for text encoding
            transform: Optional image transforms
            max_caption_length: Maximum caption length (including <START>, <END>)
            caption_column: Column name containing captions in CSV
        """
        self.csv_file = csv_file
        self.image_dir = Path(image_dir)
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_caption_length = max_caption_length
        self.caption_column = caption_column
        
        # Load data
        self.data = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['uid', 'filename', caption_column]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"CSV missing required column: {col}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, int]:
        """Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple containing:
                - image: Tensor of shape (3, H, W)
                - caption: Tensor of token indices, shape (max_caption_length,)
                - caption_length: Actual caption length (including <START>, <END>)
                - image_path: Path to image file
                - uid: Unique identifier for the sample
        """
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['filename']
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get caption text
        caption_text = str(row[self.caption_column])
        
        # Encode caption to token indices (with <START> and <END>)
        caption_tokens = self.vocabulary.encode(
            caption_text, 
            add_special_tokens=True
        )
        
        # Truncate if too long (keeping <START> and <END>)
        if len(caption_tokens) > self.max_caption_length:
            # Keep <START>, truncate middle, add <END>
            caption_tokens = (
                caption_tokens[:self.max_caption_length-1] + 
                [self.vocabulary.END_IDX]
            )
        
        # Record actual length before padding
        caption_length = len(caption_tokens)
        
        # Pad caption to max length
        caption_tokens = caption_tokens + [self.vocabulary.PAD_IDX] * (
            self.max_caption_length - len(caption_tokens)
        )
        
        # Convert to tensors
        caption = torch.tensor(caption_tokens, dtype=torch.long)
        
        return (
            image, 
            caption, 
            caption_length,
            str(image_path),
            int(row['uid'])
        )
    
    def get_sample_metadata(self, idx: int) -> dict:
        """Get metadata for a sample without loading image.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with sample metadata
        """
        row = self.data.iloc[idx]
        return {
            'uid': row['uid'],
            'filename': row['filename'],
            'projection': row.get('projection', 'Unknown'),
            'caption': row[self.caption_column],
            'split': row.get('split', 'Unknown')
        }


if __name__ == '__main__':
    import sys
    from .transforms import get_transforms
    
    print("Testing ChestXrayDataset...")
    
    # Paths
    vocab_file = "data/processed/first_frontal_impression/vocabulary.txt"
    train_csv = "data/processed/first_frontal_impression/train.csv"
    image_dir = "data/images/images_normalized"
    
    # Check if files exist
    if not os.path.exists(vocab_file):
        print(f"Vocabulary file not found: {vocab_file}")
        sys.exit(1)
    if not os.path.exists(train_csv):
        print(f"Training CSV not found: {train_csv}")
        sys.exit(1)
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = Vocabulary(vocab_file)
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Create transforms
    print("\nCreating transforms...")
    transform = get_transforms(mode='train', image_size=224)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = ChestXrayDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        vocabulary=vocab,
        transform=transform,
        max_caption_length=50
    )
    print(f"  Dataset size: {len(dataset)}")
    
    # Test loading a sample
    print("\nLoading sample...")
    image, caption, caption_length, image_path, uid = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Caption shape: {caption.shape}")
    print(f"  Caption length: {caption_length}")
    print(f"  UID: {uid}")
    print(f"  Image path: {image_path}")
    
    # Decode caption
    caption_text = vocab.decode(caption.tolist())
    print(f"  Decoded caption: '{caption_text}'")
    
    # Test metadata
    metadata = dataset.get_sample_metadata(0)
    print(f"\nMetadata:")
    print(f"  Projection: {metadata['projection']}")
    print(f"  Original caption: '{metadata['caption']}'")
    
    # Test multiple samples
    print("\nLoading 5 samples...")
    for i in range(min(5, len(dataset))):
        image, caption, caption_length, _, uid = dataset[i]
        print(f"  Sample {i}: image {image.shape}, caption length {caption_length}, uid {uid}")
    
    print("\n✓ Dataset test passed!")
