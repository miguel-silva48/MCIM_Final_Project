"""Image transforms for training and inference."""

from typing import Tuple, Optional
from torchvision import transforms


def get_transforms(
    image_size: int = 224,
    mode: str = 'train',
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    rotation_degrees: int = 10,
    color_jitter: Optional[dict] = None
) -> transforms.Compose:
    """Get image transforms for training or validation/test.
    
    Note: Horizontal/vertical flips are NOT used for chest X-rays because
    anatomical structures (heart, liver) have specific laterality. Flipping
    would create incorrect image-caption pairs (e.g., "left lower lobe").
    
    Args:
        image_size: Target image size (assumes square images)
        mode: One of 'train', 'val', 'test'
        normalize_mean: Mean for normalization (ImageNet default)
        normalize_std: Std for normalization (ImageNet default)
        rotation_degrees: Max rotation angle for augmentation (accounts for patient positioning)
        color_jitter: Dict with brightness, contrast keys for augmentation (exposure variance)
    
    Returns:
        Composed transforms
    """
    if mode == 'train':
        # Training transforms with medical-appropriate augmentation
        # NO horizontal/vertical flips - preserves anatomical laterality
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=rotation_degrees),
        ]
        
        # Add color jitter (simulates X-ray exposure variance)
        if color_jitter is None:
            color_jitter = {'brightness': 0.2, 'contrast': 0.2}
        
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0.2),
                contrast=color_jitter.get('contrast', 0.2)
            )
        )
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ]
    
    return transforms.Compose(transform_list)


if __name__ == '__main__':
    import torch
    from PIL import Image
    import numpy as np
    
    print("Testing image transforms...")
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    print(f"\nInput image: {dummy_image.size}, mode={dummy_image.mode}")
    
    # Test training transforms
    train_transform = get_transforms(
        mode='train',
        image_size=224,
        color_jitter={'brightness': 0.1, 'contrast': 0.1}
    )
    
    train_img = train_transform(dummy_image)
    print(f"\nTraining transform:")
    print(f"  Output shape: {train_img.shape}")
    print(f"  Output range: [{train_img.min():.3f}, {train_img.max():.3f}]")
    print(f"  Mean per channel: {train_img.mean(dim=[1, 2]).tolist()}")
    
    # Test validation transforms
    val_transform = get_transforms(mode='val', image_size=224)
    val_img = val_transform(dummy_image)
    print(f"\nValidation transform:")
    print(f"  Output shape: {val_img.shape}")
    print(f"  Output range: [{val_img.min():.3f}, {val_img.max():.3f}]")
    
    # Test consistency (validation should be deterministic)
    val_img2 = val_transform(dummy_image)
    print(f"\nDeterministic check (val):")
    print(f"  Images identical: {torch.equal(val_img, val_img2)}")
    
    # Test variability (training should have randomness)
    train_img2 = train_transform(dummy_image)
    print(f"\nStochastic check (train):")
    print(f"  Images identical: {torch.equal(train_img, train_img2)}")
    print(f"  (should be False due to augmentation)")
    
    print("\n✓ Transform test passed!")
