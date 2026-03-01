"""
inference.py - Single-Image Inference Pipeline

Takes a car image and returns:
1. Vehicle model name + confidence
2. License plate bounding box
3. License plate text + confidence
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from model.multitask_net import MultiTaskNet


# Stanford Cars class names (subset — full list would be loaded from file)
# This is a representative subset; the full 196 classes would be loaded from a labels file
DEFAULT_CLASS_NAMES = None  # Will be loaded from file or passed in


class VehicleInferencePipeline:
    """
    End-to-end inference pipeline for the multi-task model.
    
    Usage:
        pipeline = VehicleInferencePipeline('weights/best_model.pth')
        result = pipeline.predict('car_photo.jpg')
        print(result['vehicle_model'], result['plate_text'])
    """
    
    def __init__(self, model_path=None, num_classes=196, class_names=None, device=None):
        """
        Args:
            model_path: Path to saved model weights (None for random init)
            num_classes: Number of vehicle classes
            class_names: List of class name strings
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        
        # Initialize model
        self.model = MultiTaskNet(
            num_vehicle_classes=num_classes,
            pretrained_backbone=True
        )
        
        # Load trained weights if available
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model weights from: {model_path}")
        else:
            print("WARNING: No model weights loaded — using random initialization")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image, return_annotated=True):
        """
        Run full prediction on a single image.
        
        Args:
            image: PIL Image, file path string, or numpy array
            return_annotated: If True, return image with drawn bounding box
            
        Returns:
            dict with:
                'vehicle_model':      str - predicted vehicle name
                'vehicle_confidence': float - classification confidence
                'plate_bbox':         [x1, y1, x2, y2] in pixel coords
                'plate_text':         str - OCR result
                'plate_confidence':   float - OCR confidence
                'annotated_image':    PIL Image (if return_annotated=True)
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_image = image.copy()
        w_orig, h_orig = original_image.size
        
        # Preprocess
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run model
        with torch.no_grad():
            output = self.model(
                x, 
                task='all',
                original_images=[original_image]
            )
        
        result = {}
        
        # ---- Vehicle Classification ----
        if 'class_logits' in output:
            probs = torch.softmax(output['class_logits'], dim=1)
            conf, pred_idx = probs.max(dim=1)
            pred_idx = pred_idx.item()
            
            if self.class_names and pred_idx < len(self.class_names):
                result['vehicle_model'] = self.class_names[pred_idx]
            else:
                result['vehicle_model'] = f"Class_{pred_idx}"
            
            result['vehicle_confidence'] = round(conf.item(), 4)
            result['vehicle_class_idx'] = pred_idx
            
            # Top-5 predictions
            top5_probs, top5_idx = probs.topk(5, dim=1)
            result['top5'] = [
                {
                    'model': self.class_names[idx.item()] if self.class_names else f"Class_{idx.item()}",
                    'confidence': round(prob.item(), 4)
                }
                for prob, idx in zip(top5_probs[0], top5_idx[0])
            ]
        
        # ---- Plate Detection ----
        if 'bbox' in output:
            bbox = output['bbox'][0].cpu().numpy()  # [cx, cy, w, h] normalized
            
            # Convert to pixel coordinates [x1, y1, x2, y2]
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * w_orig)
            y1 = int((cy - bh / 2) * h_orig)
            x2 = int((cx + bw / 2) * w_orig)
            y2 = int((cy + bh / 2) * h_orig)
            
            # Clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_orig, x2)
            y2 = min(h_orig, y2)
            
            result['plate_bbox'] = [x1, y1, x2, y2]
            result['plate_bbox_normalized'] = bbox.tolist()
        
        # ---- Plate OCR ----
        if 'ocr_results' in output and output['ocr_results']:
            ocr = output['ocr_results'][0]
            result['plate_text'] = ocr['text']
            result['plate_confidence'] = round(ocr['confidence'], 4)
        else:
            result['plate_text'] = ""
            result['plate_confidence'] = 0.0
        
        # ---- Annotated Image ----
        if return_annotated and 'plate_bbox' in result:
            result['annotated_image'] = self._draw_annotations(
                original_image, result
            )
        
        return result
    
    def _draw_annotations(self, image, result):
        """Draw bounding box and labels on the image."""
        # Convert to RGBA for alpha compositing
        img = image.copy().convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(img)
        
        # Try to load a nice font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font
        
        # Draw plate bounding box
        if 'plate_bbox' in result:
            x1, y1, x2, y2 = result['plate_bbox']
            draw.rectangle([x1, y1, x2, y2], outline='#00FF00', width=3)
            
            # Draw plate text label
            plate_text = result.get('plate_text', '')
            if plate_text:
                label = f"Plate: {plate_text}"
                bbox_text = draw.textbbox((x1, y1 - 22), label, font=font)
                draw_overlay.rectangle(bbox_text, fill=(0, 200, 0, 200))
                draw_overlay.text((x1, y1 - 22), label, fill=(0, 0, 0, 255), font=font)
        
        # Draw vehicle model label at top
        vehicle_model = result.get('vehicle_model', '')
        confidence = result.get('vehicle_confidence', 0)
        if vehicle_model:
            label = f"{vehicle_model} ({confidence:.1%})"
            bbox_text = draw.textbbox((10, 10), label, font=font)
            # Semi-transparent dark background
            draw_overlay.rectangle(bbox_text, fill=(0, 0, 0, 180))
            draw_overlay.text((10, 10), label, fill=(255, 255, 255, 255), font=font)
        
        # Composite overlay and convert back to RGB
        img = Image.alpha_composite(img, overlay).convert('RGB')
        return img


if __name__ == "__main__":
    print("Vehicle Inference Pipeline")
    print("=" * 50)
    
    # Test with random weights (no trained model)
    pipeline = VehicleInferencePipeline(model_path=None, num_classes=196)
    
    # Create a dummy test image
    dummy_img = Image.new('RGB', (640, 480), color='gray')
    
    result = pipeline.predict(dummy_img, return_annotated=True)
    
    print(f"\nResults:")
    print(f"  Vehicle model:  {result['vehicle_model']}")
    print(f"  Confidence:     {result['vehicle_confidence']:.4f}")
    print(f"  Plate bbox:     {result.get('plate_bbox', 'N/A')}")
    print(f"  Plate text:     {result.get('plate_text', 'N/A')}")
    print(f"  Plate conf:     {result.get('plate_confidence', 'N/A')}")
