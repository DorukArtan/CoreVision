"""
backbone.py - Shared Feature Extractor

Uses EfficientNet-B0 pretrained on ImageNet as the shared backbone.
Outputs both spatial feature maps (for detection) and pooled features (for classification).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class SharedBackbone(nn.Module):
    """
    Shared backbone using EfficientNet-B0.
    
    Outputs:
        feature_map: (B, 1280, 7, 7) - spatial features for detection head
        pooled:      (B, 1280)       - global features for classification head
    """
    
    def __init__(self, pretrained=True):
        super(SharedBackbone, self).__init__()
        
        # Load EfficientNet-B0 with pretrained weights
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(weights=None)
        
        # Extract the feature extraction layers (everything except classifier)
        self.features = efficientnet.features  # Output: (B, 1280, 7, 7) for 224x224 input
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension
        self.feature_dim = 1280
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, 224, 224)
            
        Returns:
            feature_map: Spatial features (B, 1280, 7, 7)
            pooled: Global features (B, 1280)
        """
        # Extract spatial feature map
        feature_map = self.features(x)  # (B, 1280, 7, 7)
        
        # Global average pooling
        pooled = self.avgpool(feature_map)  # (B, 1280, 1, 1)
        pooled = torch.flatten(pooled, 1)   # (B, 1280)
        
        return feature_map, pooled
    
    def freeze(self):
        """Freeze all backbone parameters (for warmup training phases)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all backbone parameters (for joint fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
    
    def partial_unfreeze(self, num_blocks=3):
        """
        Unfreeze only the last N blocks of the backbone.
        Useful for gradual fine-tuning.
        
        Args:
            num_blocks: Number of blocks from the end to unfreeze
        """
        # First freeze everything
        self.freeze()
        
        # Unfreeze the last num_blocks
        total_blocks = len(self.features)
        for i in range(max(0, total_blocks - num_blocks), total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True


if __name__ == "__main__":
    # Quick test
    model = SharedBackbone(pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    feature_map, pooled = model(x)
    print(f"Input shape:       {x.shape}")
    print(f"Feature map shape: {feature_map.shape}")  # Expected: (2, 1280, 7, 7)
    print(f"Pooled shape:      {pooled.shape}")        # Expected: (2, 1280)
    print(f"Backbone params:   {sum(p.numel() for p in model.parameters()):,}")
