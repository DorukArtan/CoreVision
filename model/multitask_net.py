"""
multitask_net.py - Multi-Task Vehicle Recognition Network

Combines the shared backbone with task-specific heads into a single
end-to-end model that performs:
1. Vehicle model classification
2. License plate detection (bounding box)
3. License plate OCR (text reading)
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

from model.backbone import SharedBackbone
from model.heads import VehicleClassificationHead, PlateDetectionHead, PlateOCRHead


class MultiTaskNet(nn.Module):
    """
    Multi-task network for vehicle model classification and license plate recognition.
    
    Architecture:
        Input Image → SharedBackbone (EfficientNet-B0)
                        ├── pooled features → VehicleClassificationHead → vehicle model
                        └── feature map     → PlateDetectionHead → plate bbox
                                              └── crop plate region → PlateOCRHead → plate text
    
    Training modes:
        task='classify' : Only runs classification (for vehicle dataset batches)
        task='detect'   : Only runs detection (for plate dataset batches)
        task='all'      : Runs all heads (inference mode)
    """
    
    def __init__(self, num_vehicle_classes=196, ocr_languages=None, pretrained_backbone=True):
        super(MultiTaskNet, self).__init__()
        
        # Shared backbone
        self.backbone = SharedBackbone(pretrained=pretrained_backbone)
        feature_dim = self.backbone.feature_dim  # 1280
        
        # Task-specific heads
        self.classification_head = VehicleClassificationHead(
            feature_dim=feature_dim,
            num_classes=num_vehicle_classes
        )
        
        self.detection_head = PlateDetectionHead(
            feature_dim=feature_dim
        )
        
        self.ocr_head = PlateOCRHead(
            languages=ocr_languages
        )
        
        self.num_vehicle_classes = num_vehicle_classes
    
    def forward(self, x, task='all', gt_boxes=None, original_images=None):
        """
        Forward pass with task-selective execution.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            task: Which task(s) to run
                  'classify' - vehicle classification only
                  'detect'   - plate detection only
                  'plate'    - plate detection + OCR
                  'all'      - all tasks (inference)
            gt_boxes: (B, 4) ground-truth plate boxes for training
                      Format: normalized [cx, cy, w, h]
                      If provided, used instead of predicted boxes for OCR
            original_images: List of original PIL images (for OCR cropping)
                             Required for 'plate' and 'all' tasks
        
        Returns:
            dict with available outputs based on task:
                'class_logits': (B, num_classes) classification scores
                'bbox':         (B, 4)           predicted plate bbox [cx, cy, w, h]
                'ocr_results':  list of dicts    [{text, confidence}, ...]
        """
        output = {}
        
        # Always run the backbone
        feature_map, pooled = self.backbone(x)
        
        # ---- Vehicle Classification ----
        if task in ('classify', 'all'):
            output['class_logits'] = self.classification_head(pooled)
        
        # ---- Plate Detection ----
        if task in ('detect', 'plate', 'all'):
            output['bbox'] = self.detection_head(feature_map)
        
        # ---- Plate OCR ----
        if task in ('plate', 'all') and original_images is not None:
            # Use GT boxes during training, predicted boxes during inference
            boxes = gt_boxes if gt_boxes is not None else output.get('bbox')
            
            if boxes is not None:
                # Crop plate regions from original images
                plate_crops = self._crop_plates(original_images, boxes)
                output['ocr_results'] = self.ocr_head(plate_crops)
        
        return output
    
    def _crop_plates(self, images, boxes):
        """
        Crop plate regions from images using bounding boxes.
        
        Args:
            images: List of PIL Images or tensor (B, 3, H, W)
            boxes: (B, 4) normalized [cx, cy, w, h]
            
        Returns:
            List of cropped PIL Images
        """
        crops = []
        boxes_np = boxes.detach().cpu()
        
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL
                img_pil = TF.to_pil_image(img.cpu())
            else:
                img_pil = img
            
            w_img, h_img = img_pil.size
            
            # Convert [cx, cy, w, h] → [x1, y1, x2, y2] in pixel coords
            cx, cy, bw, bh = boxes_np[i]
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            # Crop (ensure minimum size)
            if x2 - x1 < 5 or y2 - y1 < 5:
                # Fallback: use center crop if bbox is too small
                crop = img_pil.crop((
                    w_img // 4, h_img * 2 // 3,
                    w_img * 3 // 4, h_img
                ))
            else:
                crop = img_pil.crop((x1, y1, x2, y2))
            
            crops.append(crop)
        
        return crops
    
    def predict(self, image, transform=None):
        """
        High-level inference method for a single image.
        
        Args:
            image: PIL Image
            transform: torchvision transform to apply (if None, uses default)
            
        Returns:
            dict with prediction results
        """
        self.eval()
        
        if transform is None:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Prepare input
        x = transform(image).unsqueeze(0)
        
        device = next(self.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            output = self.forward(
                x,
                task='all',
                original_images=[image]
            )
        
        # Post-process classification
        if 'class_logits' in output:
            probs = torch.softmax(output['class_logits'], dim=1)
            conf, pred_class = probs.max(dim=1)
            output['predicted_class'] = pred_class.item()
            output['class_confidence'] = conf.item()
        
        # Post-process detection
        if 'bbox' in output:
            output['bbox_coords'] = output['bbox'][0].cpu().numpy()
        
        return output
    
    def get_param_groups(self, backbone_lr=1e-5, head_lr=3e-4):
        """
        Create parameter groups with differential learning rates.
        Lower LR for pretrained backbone, higher LR for randomly initialized heads.
        
        Args:
            backbone_lr: Learning rate for backbone parameters
            head_lr: Learning rate for head parameters
            
        Returns:
            List of param group dicts for optimizer
        """
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.classification_head.parameters(), 'lr': head_lr},
            {'params': self.detection_head.parameters(), 'lr': head_lr},
            # OCR head has no trainable parameters (pretrained EasyOCR)
        ]


if __name__ == "__main__":
    # Quick architecture test
    print("=" * 60)
    print("Multi-Task Vehicle Recognition Network")
    print("=" * 60)
    
    model = MultiTaskNet(num_vehicle_classes=196, pretrained_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    
    # Test classification task
    out = model(x, task='classify')
    print(f"\nClassification output: {out['class_logits'].shape}")
    
    # Test detection task
    out = model(x, task='detect')
    print(f"Detection output:      {out['bbox'].shape}")
    print(f"BBox sample:           {out['bbox'][0].detach().numpy()}")
    
    # Test all tasks (without OCR since it needs real images)
    out = model(x, task='all')
    print(f"\nAll tasks (no OCR):")
    print(f"  class_logits: {out['class_logits'].shape}")
    print(f"  bbox:         {out['bbox'].shape}")
    
    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Memory estimate
    param_mb = total * 4 / (1024 ** 2)  # float32
    print(f"Estimated model size: {param_mb:.1f} MB")
