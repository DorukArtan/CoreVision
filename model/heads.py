"""
heads.py - Task-Specific Heads

Three specialized heads attached to the shared backbone:
1. VehicleClassificationHead - Classifies vehicle model (e.g., Toyota Corolla)
2. PlateDetectionHead        - Localizes license plate bounding box
3. PlateOCRHead              - Reads license plate text using EasyOCR
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class VehicleClassificationHead(nn.Module):
    """
    Classifies the vehicle model from pooled backbone features.
    
    Architecture: FC(1280→512) → BN → ReLU → Dropout → FC(512→num_classes)
    Loss: CrossEntropyLoss
    """
    
    def __init__(self, feature_dim=1280, num_classes=196, dropout=0.3):
        super(VehicleClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, pooled_features):
        """
        Args:
            pooled_features: (B, 1280) from backbone global avg pool
            
        Returns:
            logits: (B, num_classes) raw class scores
        """
        return self.classifier(pooled_features)


class PlateDetectionHead(nn.Module):
    """
    Predicts license plate bounding box from spatial feature maps.
    Assumes one plate per image (single-plate regression).
    
    Architecture: Conv layers → AdaptiveAvgPool → FC → 4 bbox coordinates
    Output: Normalized [cx, cy, w, h] in range [0, 1]
    Loss: SmoothL1Loss
    """
    
    def __init__(self, feature_dim=1280):
        super(PlateDetectionHead, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
    
    def forward(self, feature_map):
        """
        Args:
            feature_map: (B, 1280, 7, 7) spatial features from backbone
            
        Returns:
            bbox: (B, 4) normalized [cx, cy, w, h] coordinates
        """
        x = self.conv_layers(feature_map)   # (B, 64, 7, 7)
        x = self.pool(x)                     # (B, 64, 1, 1)
        x = torch.flatten(x, 1)              # (B, 64)
        bbox = self.regressor(x)              # (B, 4)
        return bbox


class PlateOCRHead(nn.Module):
    """
    Reads license plate text using EasyOCR.
    
    This is a wrapper around the pretrained EasyOCR model.
    It takes a cropped plate region and returns the recognized text.
    
    Note: EasyOCR runs on CPU/GPU independently — it's not a trainable
    PyTorch module in the traditional sense, but wrapping it here keeps
    the interface clean and modular.
    """
    
    def __init__(self, languages=None):
        super(PlateOCRHead, self).__init__()
        
        # Lazy initialization — EasyOCR is heavy, load only when needed
        self._reader = None
        self._languages = languages or ['en']  # English covers Latin chars + digits
    
    @property
    def reader(self):
        """Lazy-load EasyOCR reader on first use."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                self._languages,
                gpu=torch.cuda.is_available()
            )
        return self._reader
    
    def _preprocess_plate(self, img_np):
        """
        Preprocess a cropped plate image for better OCR accuracy.
        
        Steps:
        1. Convert to grayscale
        2. Apply CLAHE (adaptive contrast enhancement)
        3. Upscale 2x for small plates
        4. Bilateral filter to reduce noise while keeping edges
        
        Returns:
            Preprocessed numpy array
        """
        try:
            import cv2
        except ImportError:
            return img_np  # Fallback: no preprocessing
        
        # Ensure uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Upscale small plates (height < 50px gets 2x scale)
        h, w = gray.shape[:2]
        if h < 50:
            scale = max(2, 50 // h)
            gray = cv2.resize(gray, (w * scale, h * scale),
                              interpolation=cv2.INTER_CUBIC)
        
        # CLAHE — adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter — smooth noise, keep edges
        filtered = cv2.bilateralFilter(enhanced, d=5,
                                        sigmaColor=50, sigmaSpace=50)
        
        return filtered
    
    def forward(self, plate_images):
        """
        Args:
            plate_images: List of PIL Images or numpy arrays (cropped plate regions)
            
        Returns:
            texts: List of recognized plate text strings
            confidences: List of confidence scores
        """
        results = []
        
        for img in plate_images:
            # Convert to numpy if needed
            if isinstance(img, Image.Image):
                img_np = np.array(img)
            elif isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img
            
            # Preprocess for better OCR
            processed = self._preprocess_plate(img_np)
            
            # Run EasyOCR
            detections = self.reader.readtext(processed)
            
            if detections:
                # Concatenate all detected text segments
                plate_text = ' '.join([det[1] for det in detections])
                avg_confidence = np.mean([det[2] for det in detections])
            else:
                plate_text = ""
                avg_confidence = 0.0
            
            results.append({
                'text': plate_text,
                'confidence': float(avg_confidence)
            })
        
        return results
    
    def read_single(self, plate_image):
        """Convenience method for single image OCR."""
        return self.forward([plate_image])[0]


if __name__ == "__main__":
    # Test classification head
    cls_head = VehicleClassificationHead(1280, 196)
    pooled = torch.randn(2, 1280)
    logits = cls_head(pooled)
    print(f"Classification output: {logits.shape}")  # (2, 196)
    
    # Test detection head
    det_head = PlateDetectionHead(1280)
    feat_map = torch.randn(2, 1280, 7, 7)
    bbox = det_head(feat_map)
    print(f"Detection output: {bbox.shape}")  # (2, 4)
    print(f"BBox values (should be 0-1): {bbox[0].detach().numpy()}")
