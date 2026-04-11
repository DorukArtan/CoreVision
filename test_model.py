"""Quick test to verify the model architecture works correctly."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model.multitask_net import MultiTaskNet
from model.losses import MultiTaskLoss

print("=" * 50)
print("Multi-Task Model Test")
print("=" * 50)

# Create model (no pretrained weights to save download time)
TEST_NUM_CLASSES = 100  # Placeholder for testing (real count is auto-discovered)
model = MultiTaskNet(TEST_NUM_CLASSES, pretrained_backbone=False)
x = torch.randn(2, 3, 224, 224)

# Test classification
out = model(x, task='classify')
cls_shape = out['class_logits'].shape
print(f"Classification: {cls_shape}")
assert cls_shape == (2, TEST_NUM_CLASSES), f"Expected (2, {TEST_NUM_CLASSES}), got {cls_shape}"

# Test detection
out = model(x, task='detect')
bbox_shape = out['bbox'].shape
print(f"Detection bbox: {bbox_shape}")
print(f"BBox values:    {out['bbox'][0].detach().numpy()}")
assert bbox_shape == (2, 4), f"Expected (2, 4), got {bbox_shape}"

# Test all tasks
out = model(x, task='all')
print(f"All tasks: cls={out['class_logits'].shape}, bbox={out['bbox'].shape}")

# Test loss function
loss_fn = MultiTaskLoss()

# Classification loss only
preds_cls = {'class_logits': torch.randn(2, TEST_NUM_CLASSES)}
targets_cls = {'class_labels': torch.randint(0, TEST_NUM_CLASSES, (2,))}
total, details = loss_fn(preds_cls, targets_cls, task='classify')
print(f"\nClassification loss: {total.item():.4f}")

# Detection loss only
preds_det = {'bbox': torch.sigmoid(torch.randn(2, 4))}
targets_det = {'bbox_labels': torch.rand(2, 4)}
total, details = loss_fn(preds_det, targets_det, task='detect')
print(f"Detection loss:      {total.item():.4f}")

# Joint loss
preds_all = {**preds_cls, **preds_det}
targets_all = {**targets_cls, **targets_det}
total, details = loss_fn(preds_all, targets_all)
print(f"Joint loss:          {total.item():.4f}")
print(f"Loss details:        {details}")

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal params:  {total_params:,}")
print(f"Model size:    {total_params * 4 / 1024**2:.1f} MB")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
