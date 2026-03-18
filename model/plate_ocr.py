"""
plate_ocr.py - License Plate Text Recognition

Uses PaddleOCR for multi-language license plate text reading.
Supports 80+ languages including Latin, Cyrillic, Arabic, CJK characters.
Falls back to EasyOCR if PaddleOCR is unavailable.
"""

import numpy as np
from PIL import Image
import re

# Try PaddleOCR first, fall back to EasyOCR
PADDLE_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

if not PADDLE_AVAILABLE and not EASYOCR_AVAILABLE:
    print("WARNING: Neither PaddleOCR nor EasyOCR installed!")
    print("Install one: pip install paddleocr  OR  pip install easyocr")


class PlateOCR:
    """
    Read license plate text from cropped plate images.
    
    Supports international plates with multi-language OCR.
    
    Usage:
        ocr = PlateOCR()
        result = ocr.read_plate(plate_crop)
        print(result['text'], result['confidence'])
    """
    
    def __init__(self, engine='auto', languages=None):
        """
        Args:
            engine: 'paddle', 'easyocr', or 'auto' (try paddle first)
            languages: List of language codes (default: ['en'])
                       For PaddleOCR: 'en', 'ch', 'ar', 'hi', 'ru', etc.
                       For EasyOCR: 'en', 'tr', 'ar', 'ko', 'ja', etc.
        """
        self.languages = languages or ['en']
        self._reader = None
        
        # Select engine
        if engine == 'auto':
            if PADDLE_AVAILABLE:
                self.engine = 'paddle'
            elif EASYOCR_AVAILABLE:
                self.engine = 'easyocr'
            else:
                raise ImportError("No OCR engine available. Install paddleocr or easyocr")
        else:
            if engine == 'paddle' and not PADDLE_AVAILABLE:
                raise ImportError(
                    "PaddleOCR is not installed. Install it with: pip install paddleocr"
                )
            if engine == 'easyocr' and not EASYOCR_AVAILABLE:
                raise ImportError(
                    "EasyOCR is not installed. Install it with: pip install easyocr"
                )
            self.engine = engine
    
    @property
    def reader(self):
        """Lazy-load OCR engine on first use (heavy initialization)."""
        if self._reader is None:
            if self.engine == 'paddle':
                device = 'gpu' if self._gpu_available() else 'cpu'
                self._reader = PaddleOCR(
                    lang=self.languages[0] if self.languages else 'en',
                    use_textline_orientation=True,
                    device=device,
                    enable_mkldnn=False
                )
            elif self.engine == 'easyocr':
                try:
                    import easyocr as easyocr_module
                except ImportError as exc:
                    raise ImportError(
                        "EasyOCR is required for engine='easyocr'. Install it with: pip install easyocr"
                    ) from exc
                self._reader = easyocr_module.Reader(
                    self.languages,
                    gpu=self._gpu_available()
                )
        return self._reader
    
    def _gpu_available(self):
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def preprocess_plate(self, img_np):
        """
        Preprocess a cropped plate image for better OCR accuracy.
        
        New Pipeline:
        1. Keep original color (PaddleOCR prefers raw RGB/BGR)
        2. Upscale small plates
        3. Add a generous synthetic border (padding) so edge characters aren't cut off
        
        Returns:
            Preprocessed numpy array
        """
        try:
            import cv2
        except ImportError:
            return img_np
            
        # Ensure uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
            
        # 1. Upscale
        h, w = img_np.shape[:2]
        if h < 60:
            scale = max(2, 60 // h)
            img_np = cv2.resize(img_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            
        # 2. Add Border Padding
        # This helps the convolutional neural network read edge characters
        pad_y = int(img_np.shape[0] * 0.2)
        pad_x = int(img_np.shape[1] * 0.1)
        
        # Determine border color (use the average edge color or just black)
        padded = cv2.copyMakeBorder(
            img_np,
            pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0] # black padding
        )
        
        return padded
    
    def read_plate(self, image, preprocess=True):
        """
        Read text from a cropped license plate image.
        
        Args:
            image: PIL Image, numpy array, or file path
            preprocess: Whether to apply preprocessing
            
        Returns:
            dict with:
                'text': str - cleaned plate text
                'raw_text': str - raw OCR output
                'confidence': float - OCR confidence (0-1)
                'detections': list - raw OCR detection details
        """
        # Convert to numpy
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Preprocess
        if preprocess:
            processed = self.preprocess_plate(img_np)
        else:
            processed = img_np
        
        # Run OCR
        if self.engine == 'paddle':
            return self._read_paddle(processed, img_np)
        else:
            return self._read_easyocr(processed, img_np)
    
    def _read_paddle(self, processed, original):
        """Read plate with PaddleOCR."""
        # Newer PaddleOCR versions deprecate cls and route through predict().
        result = self.reader.ocr(processed)
        
        if result and result[0]:
            texts = []
            confidences = []
            detections = []

            first_page = result[0]

            # PaddleOCR >=3 returns OCRResult objects with rec_texts/rec_scores/dt_polys.
            if hasattr(first_page, 'get') and first_page.get('rec_texts') is not None:
                rec_texts = first_page.get('rec_texts', [])
                rec_scores = first_page.get('rec_scores', [])
                dt_polys = first_page.get('dt_polys', [])

                for idx, text in enumerate(rec_texts):
                    confidence = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                    bbox = dt_polys[idx] if idx < len(dt_polys) else []
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    texts.append(text)
                    confidences.append(confidence)
                    detections.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            else:
                # Legacy PaddleOCR output format.
                for line in first_page:
                    bbox, (text, confidence) = line[0], (line[1][0], line[1][1])
                    texts.append(text)
                    confidences.append(confidence)
                    detections.append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': bbox
                    })
            
            raw_text = ' '.join(texts)
            cleaned = self._clean_plate_text(raw_text)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            
            return {
                'text': cleaned,
                'raw_text': raw_text,
                'confidence': round(avg_conf, 4),
                'detections': detections
            }
        
        return {
            'text': '',
            'raw_text': '',
            'confidence': 0.0,
            'detections': []
        }
    
    def _read_easyocr(self, processed, original):
        """Read plate with EasyOCR."""
        # Force the engine to strictly use uppercase characters and digits
        plate_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'
        
        detections = self.reader.readtext(processed, allowlist=plate_allowlist)
        
        if detections:
            texts = [det[1] for det in detections]
            confidences = [det[2] for det in detections]
            
            raw_text = ' '.join(texts)
            cleaned = self._clean_plate_text(raw_text)
            avg_conf = float(np.mean(confidences))
            
            return {
                'text': cleaned,
                'raw_text': raw_text,
                'confidence': round(avg_conf, 4),
                'detections': [
                    {'text': det[1], 'confidence': float(det[2]), 'bbox': det[0]}
                    for det in detections
                ]
            }
        
        return {
            'text': '',
            'raw_text': '',
            'confidence': 0.0,
            'detections': []
        }
    
    def _clean_plate_text(self, text):
        """
        Clean and normalize OCR output for license plates.
        
        - Remove special characters (keep strictly alphanumeric, spaces, hyphens)
        - Uppercase
        - Strip whitespace
        """
        # Uppercase
        text = text.upper().strip()
        
        # Keep only alphanumeric, spaces, hyphens (removed dots)
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def read_batch(self, images, preprocess=True):
        """
        Read plates from multiple images.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            List of result dicts (same format as read_plate)
        """
        return [self.read_plate(img, preprocess=preprocess) for img in images]


if __name__ == "__main__":
    print("PlateOCR - Multi-Language License Plate Reader")
    print("=" * 50)
    
    engine = "paddle" if PADDLE_AVAILABLE else ("easyocr" if EASYOCR_AVAILABLE else "none")
    print(f"Available engine: {engine}")
    
    if engine != "none":
        ocr = PlateOCR()
        # Test with a simple white image with text-like noise
        dummy = Image.new('RGB', (200, 50), color='white')
        result = ocr.read_plate(dummy)
        print(f"Text: '{result['text']}' | Confidence: {result['confidence']}")
