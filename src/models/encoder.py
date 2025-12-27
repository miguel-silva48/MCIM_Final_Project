"""
Image encoder based on DenseNet-121.

Uses pretrained DenseNet-121 (ImageNet) as visual feature extractor
for chest X-ray images.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ImageEncoder(nn.Module):
    """
    DenseNet-121 based encoder for chest X-ray images.
    
    Extracts spatial visual features from images for the decoder to attend over.
    """
    
    def __init__(
        self,
        architecture: str = "densenet121",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_until_layer: Optional[str] = None,
        output_feature_dim: int = 1024
    ):
        """
        Initialize image encoder.
        
        Args:
            architecture: Model architecture (currently only "densenet121" supported)
            pretrained: Whether to load ImageNet pretrained weights
            freeze_backbone: Whether to freeze all encoder weights
            freeze_until_layer: Layer name to freeze up to (e.g., "denseblock3")
                              None = freeze all if freeze_backbone=True
            output_feature_dim: Expected output feature dimension
        """
        super(ImageEncoder, self).__init__()
        
        if architecture != "densenet121":
            raise ValueError(f"Unsupported architecture: {architecture}. Only 'densenet121' is supported.")
        
        # Load pretrained DenseNet-121
        if pretrained:
            # Use weights parameter for newer PyTorch versions
            try:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                self.densenet = models.densenet121(weights=weights)
            except AttributeError:
                # Fallback for older PyTorch versions
                self.densenet = models.densenet121(pretrained=True)
        else:
            self.densenet = models.densenet121(pretrained=False)
        
        # Remove classification layer - we only want features
        # DenseNet features are in self.densenet.features
        self.features = self.densenet.features
        
        # Verify output feature dimension matches DenseNet-121
        # DenseNet-121 outputs 1024 features
        if output_feature_dim != 1024:
            raise ValueError(f"DenseNet-121 outputs 1024 features, got {output_feature_dim}")
        
        self.output_feature_dim = output_feature_dim
        
        # Adaptive pooling to ensure consistent spatial dimensions
        # Output: [batch, 1024, 7, 7] regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Freeze layers if requested
        if freeze_backbone:
            if freeze_until_layer is None:
                # Freeze all layers
                for param in self.features.parameters():
                    param.requires_grad = False
            else:
                # Freeze up to specified layer
                freeze_flag = True
                for name, module in self.features.named_children():
                    if freeze_flag:
                        for param in module.parameters():
                            param.requires_grad = False
                    if name == freeze_until_layer:
                        freeze_flag = False
        
        self.frozen_backbone = freeze_backbone
        self.frozen_until = freeze_until_layer
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            images: Batch of images [batch_size, 3, H, W]
        
        Returns:
            features: Spatial features [batch_size, feature_dim, 7, 7]
        """
        # Extract features from DenseNet
        features = self.features(images)  # [batch, 1024, H', W']
        
        # Apply adaptive pooling for consistent spatial dimensions
        features = self.adaptive_pool(features)  # [batch, 1024, 7, 7]
        
        return features
    
    def get_output_dim(self) -> int:
        """Get the feature dimension output by this encoder."""
        return self.output_feature_dim
    
    def get_spatial_size(self) -> int:
        """Get the spatial dimension (H=W) of output features."""
        return 7  # After adaptive pooling
    
    def unfreeze_layers(self, layer_name: Optional[str] = None):
        """
        Unfreeze encoder layers for fine-tuning.
        
        Args:
            layer_name: Name of layer to start unfreezing from.
                       If None, unfreezes all layers.
        """
        if layer_name is None:
            # Unfreeze all
            for param in self.features.parameters():
                param.requires_grad = True
            self.frozen_backbone = False
            self.frozen_until = None
        else:
            # Unfreeze from specified layer onward
            unfreeze_flag = False
            for name, module in self.features.named_children():
                if name == layer_name:
                    unfreeze_flag = True
                if unfreeze_flag:
                    for param in module.parameters():
                        param.requires_grad = True
            self.frozen_until = layer_name
    
    def get_trainable_parameters(self) -> int:
        """Count number of trainable parameters in encoder."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Count total number of parameters in encoder."""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # Quick test
    # Note: Using pretrained=True downloads weights, set to False for quick testing
    encoder = ImageEncoder(pretrained=True, freeze_backbone=True)
    
    print(f"Encoder initialized:")
    print(f"  Total parameters: {encoder.get_total_parameters():,}")
    print(f"  Trainable parameters: {encoder.get_trainable_parameters():,}")
    print(f"  Frozen backbone: {encoder.frozen_backbone}")
    print(f"  Output feature dim: {encoder.get_output_dim()}")
    print(f"  Output spatial size: {encoder.get_spatial_size()}x{encoder.get_spatial_size()}")
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224)
    features = encoder(dummy_images)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_images.shape}")
    print(f"  Output shape: {features.shape}")
