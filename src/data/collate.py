"""Custom collate function for batching variable-length captions."""

import torch
from typing import List, Tuple


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
    """Collate function for DataLoader.
    
    Handles batching of variable-length captions by sorting and padding.
    This is required for efficient packing with pack_padded_sequence.
    
    Args:
        batch: List of tuples from Dataset.__getitem__:
               (image, caption, caption_length, image_path, uid)
    
    Returns:
        Tuple containing:
            - images: Tensor of shape (batch_size, 3, H, W)
            - captions: Tensor of shape (batch_size, max_caption_length)
            - caption_lengths: Tensor of shape (batch_size,) - sorted descending
            - image_paths: List of image path strings
            - uids: Tensor of shape (batch_size,) - unique identifiers
    """
    # Sort batch by caption length (descending) for pack_padded_sequence
    # This is required for LSTM efficiency
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    
    # Separate batch components
    images, captions, caption_lengths, image_paths, uids = zip(*batch)
    
    # Stack images (already tensors)
    images = torch.stack(images, dim=0)
    
    # Stack captions (already padded to same length in Dataset)
    captions = torch.stack(captions, dim=0)
    
    # Convert lengths to tensor
    caption_lengths = torch.tensor(caption_lengths, dtype=torch.long)
    
    # Convert UIDs to tensor
    uids = torch.tensor(uids, dtype=torch.long)
    
    # Keep image_paths as list (strings don't need to be tensors)
    image_paths = list(image_paths)
    
    return images, captions, caption_lengths, image_paths, uids


if __name__ == '__main__':
    import sys
    import os
    from torch.utils.data import DataLoader
    from .dataset import ChestXrayDataset
    from .vocabulary import Vocabulary
    from .transforms import get_transforms
    
    print("Testing collate_fn with DataLoader...")
    
    # Paths
    vocab_file = "data/processed/first_frontal_impression/vocabulary.txt"
    train_csv = "data/processed/first_frontal_impression/train.csv"
    image_dir = "data/images/images_normalized"
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [vocab_file, train_csv, image_dir]):
        print("Required files not found")
        sys.exit(1)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = Vocabulary(vocab_file)
    
    # Create dataset
    print("Creating dataset...")
    transform = get_transforms(mode='train', image_size=224)
    dataset = ChestXrayDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        vocabulary=vocab,
        transform=transform,
        max_caption_length=50
    )
    
    # Create DataLoader with custom collate function
    print("Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    # Test loading a batch
    print("\nLoading batch...")
    images, captions, caption_lengths, image_paths, uids = next(iter(dataloader))
    
    print(f"\nBatch information:")
    print(f"  Batch size: {images.shape[0]}")
    print(f"  Images shape: {images.shape}")
    print(f"  Captions shape: {captions.shape}")
    print(f"  Caption lengths shape: {caption_lengths.shape}")
    print(f"  Caption lengths: {caption_lengths.tolist()}")
    print(f"  UIDs shape: {uids.shape}")
    print(f"  Number of image paths: {len(image_paths)}")
    
    # Verify sorting (lengths should be descending)
    lengths_list = caption_lengths.tolist()
    is_sorted = all(lengths_list[i] >= lengths_list[i+1] for i in range(len(lengths_list)-1))
    print(f"\nCaption lengths sorted (descending): {is_sorted}")
    
    # Decode and print first caption
    print(f"\nFirst caption in batch:")
    caption_text = vocab.decode(captions[0].tolist())
    print(f"  Length: {caption_lengths[0]}")
    print(f"  Text: '{caption_text}'")
    
    # Test loading multiple batches
    print("\nLoading 3 batches...")
    for i, (imgs, caps, lens, paths, ids) in enumerate(dataloader):
        if i >= 3:
            break
        print(f"  Batch {i}: images {imgs.shape}, captions {caps.shape}, lengths {lens.tolist()}")
    
    print("\n✓ Collate function test passed!")
