"""
plate_ocr.py - License Plate Text Recognition

Uses PaddleOCR for multi-language license plate text reading.
Supports 80+ languages including Latin, Cyrillic, Arabic, CJK characters.
Falls back to EasyOCR if PaddleOCR is unavailable.
"""

import numpy as np
from PIL import Image
import re
import itertools

# Try PaddleOCR first, fall back to EasyOCR
PADDLE_AVAILABLE = False
EASYOCR_AVAILABLE = False
PADDLE_IMPORT_ERROR = None

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except (ImportError, TypeError) as exc:
    # Some PaddleOCR dependency stacks fail at import time on Python < 3.9.
    PADDLE_IMPORT_ERROR = exc

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
                if PADDLE_IMPORT_ERROR is not None:
                    raise ImportError(
                        "No OCR engine available. PaddleOCR import failed "
                        f"({type(PADDLE_IMPORT_ERROR).__name__}: {PADDLE_IMPORT_ERROR}). "
                        "Install EasyOCR (pip install easyocr) or use Python 3.9+ for PaddleOCR."
                    )
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
        
        Pipeline:
        1. Upscale small plates aggressively (target height ~120px)
        2. Apply CLAHE contrast enhancement
        3. Sharpen to make character edges crisper
        4. Add white border padding so edge characters aren't cut off
        
        Returns:
            Preprocessed numpy array (color-enhanced version)
        """
        try:
            import cv2
        except ImportError:
            return img_np
            
        # Ensure uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        h, w = img_np.shape[:2]
            
        # 1. Aggressive upscale — OCR needs at least ~120px height for
        #    reliable character segmentation
        target_h = 120
        if h < target_h:
            scale = max(2, target_h // h)
            img_np = cv2.resize(img_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to LAB and apply CLAHE on L-channel for contrast boost
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Sharpen with unsharp mask
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            
        # 4. Add white border padding (white matches typical plate background
        #    and doesn't create harsh edges that confuse text detection)
        pad_y = int(enhanced.shape[0] * 0.15)
        pad_x = int(enhanced.shape[1] * 0.10)
        
        # Use the mean color of the plate edges for a more natural border
        border_color = self._estimate_border_color(enhanced)
        
        padded = cv2.copyMakeBorder(
            enhanced,
            pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT,
            value=border_color
        )
        
        return padded
    
    def preprocess_plate_binary(self, img_np):
        """
        Create a high-contrast binary version of the plate for OCR fallback.
        
        This produces dark text on a white background, which is ideal
        for OCR engines that struggle with colored/textured plates.
        
        Returns:
            Preprocessed numpy array (binary, 3-channel)
        """
        try:
            import cv2
        except ImportError:
            return img_np
            
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        h, w = img_np.shape[:2]
        
        # Upscale
        target_h = 120
        if h < target_h:
            scale = max(2, target_h // h)
            img_np = cv2.resize(img_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        
        # Adaptive threshold — handles uneven lighting across the plate
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=10
        )
        
        # Light morphological cleanup — remove small noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Convert back to 3-channel for OCR engines that expect color input
        binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Add white padding
        pad_y = int(binary_3ch.shape[0] * 0.15)
        pad_x = int(binary_3ch.shape[1] * 0.10)
        padded = cv2.copyMakeBorder(
            binary_3ch,
            pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        return padded
    
    def _estimate_border_color(self, img):
        """Estimate the dominant background color from the plate edges."""
        h, w = img.shape[:2]
        # Sample pixels from the edges
        edge_pixels = []
        edge_width = max(3, min(w // 10, 10))
        edge_height = max(3, min(h // 10, 10))
        
        # Top and bottom strips
        edge_pixels.append(img[:edge_height, :].reshape(-1, 3))
        edge_pixels.append(img[-edge_height:, :].reshape(-1, 3))
        # Left and right strips
        edge_pixels.append(img[:, :edge_width].reshape(-1, 3))
        edge_pixels.append(img[:, -edge_width:].reshape(-1, 3))
        
        all_pixels = np.vstack(edge_pixels)
        mean_color = np.mean(all_pixels, axis=0).astype(int).tolist()
        
        # If the mean is very dark, default to white (most plates have
        # white/light backgrounds)
        if sum(mean_color) < 200:
            return [255, 255, 255]
        return mean_color
    
    def read_plate(self, image, preprocess=True):
        """
        Read text from a cropped license plate image.
        
        Uses a multi-attempt strategy: tries OCR on both a color-enhanced
        and a binarized version of the plate, then picks the result with
        the highest confidence.
        
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
        # Convert to numpy (BGR for cv2 compatibility)
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            # PIL gives RGB; OpenCV preprocessing expects BGR
            try:
                import cv2
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            except ImportError:
                pass
        else:
            img_np = image
        
        if not preprocess:
            # No preprocessing — run OCR directly
            if self.engine == 'paddle':
                return self._read_paddle(img_np, img_np)
            else:
                return self._read_easyocr(img_np, img_np)
        
        # --- Multi-attempt OCR ---
        # Attempt 1: Color-enhanced preprocessing
        processed_color = self.preprocess_plate(img_np)
        if self.engine == 'paddle':
            result_color = self._read_paddle(processed_color, img_np)
        else:
            result_color = self._read_easyocr(processed_color, img_np)
        
        # If color result is already very confident, skip binary attempt
        if result_color['confidence'] >= 0.90 and result_color['text']:
            return result_color
        
        # Attempt 2: Binary (adaptive threshold) preprocessing
        processed_binary = self.preprocess_plate_binary(img_np)
        if self.engine == 'paddle':
            result_binary = self._read_paddle(processed_binary, img_np)
        else:
            result_binary = self._read_easyocr(processed_binary, img_np)
        
        # Pick the better result
        # Prefer whichever has higher confidence, but also prefer results
        # that actually found text
        if not result_color['text'] and result_binary['text']:
            return result_binary
        if result_color['text'] and not result_binary['text']:
            return result_color
        if result_binary['confidence'] > result_color['confidence']:
            return result_binary
        return result_color
    
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
            
            # Filter: if multiple detections, keep only plate-region text
            if len(detections) > 1:
                filtered = self._filter_plate_detections(detections, processed.shape)
                if filtered:
                    texts = [d['text'] for d in filtered]
                    confidences = [d['confidence'] for d in filtered]
                    raw_text = ' '.join(texts)
                    cleaned = self._clean_plate_text(raw_text)
                    avg_conf = float(np.mean(confidences))
                    detections = filtered
            
            # If the cleaned text looks like it contains extra non-plate text
            # (e.g. dealer badges), try to extract just the plate portion
            if cleaned and len(cleaned) > 12:
                plate_only = self._extract_plate_from_noisy_text(cleaned)
                if plate_only:
                    cleaned = plate_only
            
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
    
    def _extract_plate_from_noisy_text(self, text):
        """
        Try to extract a valid license plate pattern from noisy OCR text
        that may contain extra text (dealer badges, watermarks, etc.).
        
        Scans the text for known plate patterns:
        - Turkish: 2-digit city + 1-3 letters + 2-4 digits
        - Polish: 2-3 letters + 4-5 digits
        """
        text = text.upper().strip()
        
        # Try to find a Turkish-style plate pattern anywhere in the text
        # Pattern: DD + space/sep + L{1,3} + space/sep + D{2,4}
        turkish = re.search(
            r'(\d{2})\s*([A-Z]{1,3})\s*(\d{2,4})',
            text
        )
        if turkish:
            city = turkish.group(1)
            valid_cities = set(f'{i:02d}' for i in range(1, 82))
            if city in valid_cities:
                return f'{city} {turkish.group(2)} {turkish.group(3)}'
        
        # Try Polish-style plate pattern
        polish = re.search(r'([A-Z]{2,3})\s*(\d{4,5})', text)
        if polish:
            return f'{polish.group(1)} {polish.group(2)}'
        
        return None
    
    def _clean_plate_text(self, text):
        """
        Clean and normalize OCR output for license plates.
        
        - Remove special characters (keep strictly alphanumeric, spaces, hyphens)
        - Uppercase
        - Strip whitespace
        - Apply Polish plate noise trimming
        - Apply Turkish plate format correction
        """
        # Uppercase
        text = text.upper().strip()
        
        # Keep only alphanumeric, spaces, hyphens (removed dots)
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()

        # Try Polish plate normalization first (e.g., trim noisy suffixes like "7D")
        polish_normalized = self._normalize_polish_plate(text)
        if polish_normalized:
            return polish_normalized
        
        # Try Turkish plate format correction
        corrected = self._correct_turkish_plate_by_position(text)
        if corrected:
            return corrected

        corrected = self._correct_turkish_plate(text)
        if corrected:
            return corrected
        
        return text

    def _normalize_polish_plate(self, text):
        """
        Normalize likely Polish format plates and remove OCR tail noise.

        Expected core format: 2-3 letters + 4-5 digits
        Examples:
            WZY 54495
            WZY-54495
            WZY-54495 7D  -> WZY 54495
            WZY544957D    -> WZY 54495
        """
        # Format with optional separators and an optional short noisy suffix.
        m = re.match(r'^([A-Z]{2,3})[\s\-]?(\d{4,5})(?:[\s\-]?([A-Z0-9]{1,2}))?$', text)
        if m:
            area = m.group(1)
            digits = m.group(2)
            suffix = m.group(3)
            # If suffix exists, treat it as OCR noise for this Polish format.
            if suffix:
                return f'{area} {digits}'
            return f'{area} {digits}'

        # Handle compact text where noise is glued to the end with no separators.
        m = re.match(r'^([A-Z]{2,3})(\d{4,5})([A-Z0-9]{1,2})$', text)
        if m:
            return f'{m.group(1)} {m.group(2)}'

        return None

    def _correct_turkish_plate_by_position(self, text):
        """
        Fast, position-aware Turkish correction for already-separated text.

        Expected token layout:
            {city_code} {letters} {digits}
        where:
            city_code -> 2 digits
            letters   -> 1-3 letters
            digits    -> 2-4 digits
        """
        LETTER_TO_DIGIT = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', 'J': '1',
            'Z': '2', 'S': '5', 'G': '6', 'T': '7', 'B': '8',
        }
        DIGIT_TO_LETTER = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '7': 'T', '8': 'B',
        }

        tokens = [t for t in re.split(r'[\s\-]+', text.strip().upper()) if t]
        if len(tokens) != 3:
            return None

        city_raw, letters_raw, digits_raw = tokens
        valid_cities = set(f'{i:02d}' for i in range(1, 82))

        if not (1 <= len(letters_raw) <= 3 and 2 <= len(digits_raw) <= 4):
            return None

        city = ''.join(ch if ch.isdigit() else LETTER_TO_DIGIT.get(ch, ch) for ch in city_raw)
        letters = ''.join(ch if ch.isalpha() else DIGIT_TO_LETTER.get(ch, ch) for ch in letters_raw)
        digits = ''.join(ch if ch.isdigit() else LETTER_TO_DIGIT.get(ch, ch) for ch in digits_raw)

        if city not in valid_cities:
            return None
        if not letters.isalpha():
            return None
        if not digits.isdigit():
            return None

        return f'{city} {letters} {digits}'
    
    def _correct_turkish_plate(self, text):
        """
        Correct OCR misreads using Turkish plate format rules.
        
        Turkish plates: {2-digit city} {1-3 letters} {2-4 digits}
        Examples: 07 SM 014, 34 ABC 1234, 06 A 1234
        
        Common OCR confusions:
            In digit positions: O→0, I→1, S→5, Z→2, B→8, G→6, T→7, L→1
            In letter positions: 0→O, 1→I, 5→S, 2→Z, 8→B, 6→G, 7→T
        """
        # Character correction maps
        LETTER_TO_DIGIT = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', 'J': '1',
            'Z': '2',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8',
        }
        DIGIT_TO_LETTER = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B',
        }
        LETTER_ALTERNATIVES = {
            # Common Turkish plate OCR confusions between letters
            'I': [('A', 0.08)],
            'R': [('B', 0.14)],
            'P': [('B', 0.18)],
            'Y': [('V', 0.03)],
            'V': [('Y', 0.06)],
        }
        
        # Remove all spaces/hyphens to get raw characters
        raw = re.sub(r'[\s\-]', '', text)
        
        # Strip trailing country code "TR" (Turkey)
        raw_no_tr = re.sub(r'TR$', '', raw)
        
        # Build candidates: original, without TR, reversed versions
        candidates = [raw, raw_no_tr]
        if len(raw_no_tr) >= 5:
            candidates.append(raw_no_tr[::-1])
        
        # Valid Turkish city codes: 01-81
        valid_cities = set(f'{i:02d}' for i in range(1, 82))
        
        best_result = None
        best_score = float('inf')
        best_letters = None
        
        for candidate in candidates:
            if len(candidate) < 5 or len(candidate) > 9:
                continue
            
            # Try all possible splits: DD + L(1-3) + D(2-4)
            for n_letters in [1, 2, 3]:
                for n_trail_digits in [2, 3, 4]:
                    expected_len = 2 + n_letters + n_trail_digits
                    if len(candidate) != expected_len:
                        continue
                    
                    # Split into parts
                    city_raw = candidate[:2]
                    letters_raw = candidate[2:2+n_letters]
                    digits_raw = candidate[2+n_letters:]
                    
                    # City: deterministic normalization (must become 2 digits)
                    city = ''
                    city_penalty = 0.0
                    for ch in city_raw:
                        if ch.isdigit():
                            city += ch
                        elif ch in LETTER_TO_DIGIT:
                            city += LETTER_TO_DIGIT[ch]
                            city_penalty += 0.12
                        else:
                            city += ch
                            city_penalty += 1.0

                    if city not in valid_cities:
                        continue

                    # Build letter options with penalties.
                    letter_choices = []
                    for idx, ch in enumerate(letters_raw):
                        opts = []
                        if ch.isalpha():
                            opts.append((ch, 0.0))
                        if ch in DIGIT_TO_LETTER:
                            opts.append((DIGIT_TO_LETTER[ch], 0.12))
                        # Additional letter-vs-letter alternatives for OCR confusion.
                        for alt, penalty in LETTER_ALTERNATIVES.get(ch, []):
                            opts.append((alt, penalty))
                        if not opts:
                            opts.append((ch, 1.0))
                        letter_choices.append(opts)

                    # Build digit options with penalties.
                    digit_choices = []
                    for ch in digits_raw:
                        opts = []
                        if ch.isdigit():
                            opts.append((ch, 0.0))
                        if ch in LETTER_TO_DIGIT:
                            opts.append((LETTER_TO_DIGIT[ch], 0.12))
                        if not opts:
                            opts.append((ch, 1.0))
                        digit_choices.append(opts)

                    for letters_combo in itertools.product(*letter_choices):
                        letters = ''.join(ch for ch, _ in letters_combo)
                        letters_penalty = sum(p for _, p in letters_combo)
                        if not letters.isalpha():
                            continue

                        for digits_combo in itertools.product(*digit_choices):
                            digits_base = ''.join(ch for ch, _ in digits_combo)
                            digits_penalty = sum(p for _, p in digits_combo)
                            if not digits_base.isdigit():
                                continue

                            # Missing narrow 0 is common; if only 2 digits seen, try inserting 0.
                            digit_variants = [(digits_base, 0.0)]
                            if len(digits_base) == 2:
                                digit_variants.extend([
                                    (f'{digits_base[0]}0{digits_base[1]}', 0.06),
                                    (f'0{digits_base}', 0.15),
                                    (f'{digits_base}0', 0.18),
                                ])

                            for digits, insert_penalty in digit_variants:
                                if len(digits) < 2 or len(digits) > 4:
                                    continue
                                if not digits.isdigit():
                                    continue

                                result = f'{city} {letters} {digits}'

                                score = city_penalty + letters_penalty + digits_penalty + insert_penalty
                                score += {3: 0.0, 2: 0.18, 1: 0.35}.get(n_letters, 0.35)
                                score += {3: 0.0, 4: 0.10, 2: 0.30}.get(len(digits), 0.30)

                                # If OCR dropped one trailing digit (2-digit read) and
                                # last letter was read as R/P, B is often a visual confusion.
                                if (
                                    len(digits_base) == 2
                                    and letters_raw
                                    and letters_raw[-1] in {'R', 'P'}
                                    and letters.endswith('B')
                                ):
                                    score -= 0.10

                                should_replace = False
                                if best_result is None or score < best_score:
                                    should_replace = True
                                elif abs(score - best_score) <= 0.02 and best_letters is not None:
                                    # Tie-breaker for a frequent OCR confusion: prefer V over Y
                                    # when scores are effectively equal.
                                    if letters.count('Y') < best_letters.count('Y'):
                                        should_replace = True

                                if should_replace:
                                    best_result = result
                                    best_score = score
                                    best_letters = letters
        
        return best_result
    
    def _filter_plate_detections(self, detections, img_shape):
        """
        Filter OCR detections to keep only actual plate text,
        excluding dealership badges/text that appear below the plate.
        
        Strategy:
        1. Vertical filter: keep text in the upper 60% of the image
        2. Size heuristic: the plate text is typically the tallest detection;
           smaller text below it is likely a dealer badge
        """
        if not detections:
            return detections
        
        img_h = img_shape[0]
        max_y_threshold = img_h * 0.60  # Only keep text in upper 60%
        
        def _get_bbox_info(det):
            """Extract y_center and height from a detection bbox."""
            bbox = det.get('bbox', [])
            if not bbox:
                return None, None
            try:
                if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                    if isinstance(bbox[0], (list, np.ndarray)):
                        y_coords = [pt[1] for pt in bbox]
                    else:
                        y_coords = [bbox[1], bbox[3]]
                    y_center = sum(y_coords) / len(y_coords)
                    y_height = max(y_coords) - min(y_coords)
                    return y_center, y_height
            except (TypeError, IndexError):
                pass
            return None, None
        
        # First pass: vertical position filter
        filtered = []
        for det in detections:
            y_center, y_height = _get_bbox_info(det)
            if y_center is None:
                filtered.append(det)  # Keep if no bbox info
                continue
            if y_center <= max_y_threshold:
                filtered.append(det)
        
        if filtered:
            return filtered
        
        # Second pass (fallback): if vertical filter removed everything,
        # find the tallest detection (plate text) and keep only detections
        # whose vertical center is within the same band
        det_info = []
        for det in detections:
            y_center, y_height = _get_bbox_info(det)
            if y_center is not None and y_height is not None:
                det_info.append((det, y_center, y_height))
            else:
                det_info.append((det, 0, 0))
        
        if det_info:
            # The tallest text is most likely the plate number
            tallest = max(det_info, key=lambda x: x[2])
            tallest_y = tallest[1]
            tallest_h = tallest[2]
            
            # Keep detections that overlap vertically with the tallest one
            band_filtered = []
            for det, y_center, y_height in det_info:
                if y_center is None or abs(y_center - tallest_y) <= tallest_h * 0.8:
                    band_filtered.append(det)
            
            if band_filtered:
                return band_filtered
        
        return detections  # Ultimate fallback: return all
    
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
