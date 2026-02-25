"""
losses.py - Multi-Task Loss Functions

Implements:
1. Individual task losses (classification, detection)
2. Uncertainty-weighted multi-task loss (Kendall et al., 2018)
   "Multi-Task Learning Using Uncertainty to Weigh Losses"
"""

import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """
    Combined detection loss for bounding box regression.
    
    Uses SmoothL1Loss for coordinate regression + GIoU loss for
    better geometric alignment.
    """
    
    def __init__(self, smooth_l1_weight=1.0, giou_weight=1.0):
        super(DetectionLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.smooth_l1_weight = smooth_l1_weight
        self.giou_weight = giou_weight
    
    def forward(self, pred_bbox, gt_bbox):
        """
        Args:
            pred_bbox: (B, 4) predicted [cx, cy, w, h] normalized
            gt_bbox:   (B, 4) ground-truth [cx, cy, w, h] normalized
            
        Returns:
            Combined detection loss
        """
        # SmoothL1 loss on coordinates
        l1_loss = self.smooth_l1(pred_bbox, gt_bbox)
        
        # GIoU loss
        giou_loss = self._giou_loss(pred_bbox, gt_bbox)
        
        return self.smooth_l1_weight * l1_loss + self.giou_weight * giou_loss
    
    def _cxcywh_to_xyxy(self, boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _giou_loss(self, pred, target):
        """Generalized IoU loss."""
        pred_xyxy = self._cxcywh_to_xyxy(pred)
        target_xyxy = self._cxcywh_to_xyxy(target)
        
        # Intersection
        inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
        inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
        inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
        inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * \
                    (pred_xyxy[..., 3] - pred_xyxy[..., 1])
        target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]) * \
                      (target_xyxy[..., 3] - target_xyxy[..., 1])
        
        union_area = pred_area + target_area - inter_area + 1e-7
        
        iou = inter_area / union_area
        
        # Enclosing box
        enc_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
        enc_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
        enc_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
        enc_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-7
        
        giou = iou - (enc_area - union_area) / enc_area
        
        return (1 - giou).mean()


class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics" (Kendall et al., CVPR 2018).
    
    Each task has a learnable log-variance parameter that automatically
    balances the loss contributions. Tasks with higher uncertainty get
    lower weight, preventing any single task from dominating.
    
    Formula per task:
        weighted_loss_i = (1 / (2 * exp(log_var_i))) * loss_i + log_var_i / 2
    """
    
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        
        # Learnable log-variance parameters (one per task)
        # Initialized to 0 → initial weight = 1/(2*1) = 0.5 per task
        self.log_var_cls = nn.Parameter(torch.zeros(1))  # Classification
        self.log_var_det = nn.Parameter(torch.zeros(1))  # Detection
        
        # Individual loss functions
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.det_loss_fn = DetectionLoss()
    
    def forward(self, predictions, targets, task=None):
        """
        Compute weighted multi-task loss.
        
        Only computes losses for tasks that have labels in the current batch.
        This handles the disjoint dataset scenario where vehicle images don't
        have plate labels and vice versa.
        
        Args:
            predictions: dict from model forward pass
                'class_logits': (B, num_classes) 
                'bbox':         (B, 4)
            targets: dict with available ground truth
                'class_labels': (B,) class indices (vehicle dataset)
                'bbox_labels':  (B, 4) normalized [cx, cy, w, h] (plate dataset)
            task: Override which losses to compute
                  'classify', 'detect', or None (auto-detect from targets)
            
        Returns:
            total_loss: Weighted sum of active task losses
            loss_dict: Individual loss values for logging
        """
        total_loss = torch.tensor(0.0, device=self._get_device())
        loss_dict = {}
        
        # ---- Classification Loss ----
        if task in ('classify', None) and 'class_labels' in targets and 'class_logits' in predictions:
            cls_loss = self.cls_loss_fn(
                predictions['class_logits'], 
                targets['class_labels']
            )
            
            # Uncertainty weighting
            precision_cls = torch.exp(-self.log_var_cls)
            weighted_cls = precision_cls * cls_loss + self.log_var_cls
            
            total_loss = total_loss + weighted_cls.squeeze()
            loss_dict['cls_loss'] = cls_loss.item()
            loss_dict['cls_weight'] = precision_cls.item()
        
        # ---- Detection Loss ----
        if task in ('detect', None) and 'bbox_labels' in targets and 'bbox' in predictions:
            det_loss = self.det_loss_fn(
                predictions['bbox'], 
                targets['bbox_labels']
            )
            
            # Uncertainty weighting
            precision_det = torch.exp(-self.log_var_det)
            weighted_det = precision_det * det_loss + self.log_var_det
            
            total_loss = total_loss + weighted_det.squeeze()
            loss_dict['det_loss'] = det_loss.item()
            loss_dict['det_weight'] = precision_det.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _get_device(self):
        return self.log_var_cls.device


if __name__ == "__main__":
    # Test losses
    print("Testing Multi-Task Loss")
    print("=" * 50)
    
    loss_fn = MultiTaskLoss()
    
    # Simulate classification batch (vehicle dataset)
    preds_cls = {'class_logits': torch.randn(4, 196)}
    targets_cls = {'class_labels': torch.randint(0, 196, (4,))}
    
    total, details = loss_fn(preds_cls, targets_cls, task='classify')
    print(f"\nClassification only:")
    print(f"  Total loss: {total.item():.4f}")
    print(f"  Details: {details}")
    
    # Simulate detection batch (plate dataset)
    preds_det = {'bbox': torch.sigmoid(torch.randn(4, 4))}
    targets_det = {'bbox_labels': torch.rand(4, 4)}
    
    total, details = loss_fn(preds_det, targets_det, task='detect')
    print(f"\nDetection only:")
    print(f"  Total loss: {total.item():.4f}")
    print(f"  Details: {details}")
    
    # Simulate mixed batch (joint fine-tuning — both targets available)
    preds_all = {**preds_cls, **preds_det}
    targets_all = {**targets_cls, **targets_det}
    
    total, details = loss_fn(preds_all, targets_all)
    print(f"\nJoint loss (both tasks):")
    print(f"  Total loss: {total.item():.4f}")
    print(f"  Details: {details}")
    
    print(f"\nLearnable parameters:")
    print(f"  log_var_cls = {loss_fn.log_var_cls.item():.4f}")
    print(f"  log_var_det = {loss_fn.log_var_det.item():.4f}")
