"""
pipeline.py - End-to-End Vehicle Recognition Pipeline

Orchestrates all components:
1. Video → frame extraction
2. Frame → vehicle + plate detection (YOLOv8)
3. Vehicle crop → car model classification (EfficientNetV2-S)
4. Plate crop → OCR text reading (PaddleOCR)
5. Plate text → country identification (regex + lookup)
"""

import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torchvision import transforms
try:
    import timm
except ImportError:
    timm = None

from model.video_processor import VideoProcessor
from model.detector import VehiclePlateDetector
from model.car_classifier import CarClassifier
from model.clip_brand_classifier import CLIPBrandClassifier
from model.model_sub_classifier import BrandSubClassifierRouter
from model.plate_ocr import PlateOCR
from model.country_identifier import CountryIdentifier


class BrandClassifierBest:
    """
    Brand classifier backed by brand_classifier_best.pth.

    This checkpoint was saved as a raw timm EfficientNetV2-S model
    (no nn.Sequential custom head), so it must be loaded differently
    from the legacy CarClassifier weights.

    Class names are read from data/test_cars.txt (sorted, one per line).
    """

    def __init__(self, weights_path, class_names_path, device=None):
        if timm is None:
            raise ImportError("timm is required. Install: pip install timm")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load class names (must be sorted — matches training order)
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = sorted([line.strip() for line in f if line.strip()])
        self.num_classes = len(self.class_names)

        # Build raw timm model (no custom Sequential head)
        self.model = timm.create_model(
            'tf_efficientnetv2_s',
            num_classes=self.num_classes
        )

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"BrandClassifierBest loaded ({self.num_classes} brands from test_cars.txt)")

    def predict(self, image, top_k=5):
        """
        Predict brand from a cropped vehicle image.

        Returns dict compatible with the existing pipeline result format:
            'make_model': str  (best brand name)
            'confidence': float
            'top_k': list of {'make_model': str, 'confidence': float}
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')

        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)

        predictions = [
            {
                'make_model': self.class_names[idx.item()],
                'confidence': round(prob.item(), 4)
            }
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

        best = predictions[0] if predictions else {'make_model': 'Unknown', 'confidence': 0.0}
        return {
            'make_model': best['make_model'],
            'confidence': best['confidence'],
            'top_k': predictions
        }


class VehicleRecognitionPipeline:
    """
    Full pipeline: Video/Image → Car Model + Plate Text + Country.
    
    Usage:
        pipeline = VehicleRecognitionPipeline(
            car_weights='weights/car_classifier.pth',
            plate_det_weights='weights/plate_detector.pt'
        )
        
        # From video
        results = pipeline.process_video('dashcam.mp4')
        
        # From single image
        result = pipeline.process_image('car.jpg')
    """
    
    def __init__(self,
                 car_weights=None,
                 plate_det_weights=None,
                 car_class_names=None,
                 num_car_classes=None,
                 clip_brand_list=None,
                 brand_confidence_threshold=0.3,
                 target_fps=2,
                 device=None):
        """
        Args:
            car_weights: Path to EfficientNetV2-S car classifier weights
            plate_det_weights: Path to YOLOv8 plate detector weights
            car_class_names: Path to car class names text file
            num_car_classes: Number of car model classes
            clip_brand_list: Path to brand names file for CLIP zero-shot fallback
            brand_confidence_threshold: If car confidence < this, use CLIP brand fallback
            target_fps: Frames per second for video extraction
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print("Initializing Vehicle Recognition Pipeline...")
        start = time.time()
        
        # Set default weight paths
        if car_weights is None:
            car_weights = os.path.join(PROJECT_ROOT, 'weights', 'car_classifier.pth')
        if plate_det_weights is None:
            plate_det_weights = os.path.join(PROJECT_ROOT, 'weights', 'plate_detector.pt')
        if car_class_names is None:
            car_class_names = os.path.join(PROJECT_ROOT, 'data', 'vmmrdb_classes.txt')
        
        # Initialize components
        self.video_processor = VideoProcessor(target_fps=target_fps)
        
        self.detector = VehiclePlateDetector(
            plate_weights=plate_det_weights if os.path.exists(plate_det_weights) else None,
            device=device
        )
        
        # Auto-discover num_classes from class names file
        if num_car_classes is None and os.path.exists(car_class_names):
            with open(car_class_names, 'r', encoding='utf-8') as f:
                num_car_classes = sum(1 for line in f if line.strip())
        if num_car_classes is None:
            num_car_classes = 100  # fallback default
        
        self.car_classifier = CarClassifier(
            weights_path=car_weights if os.path.exists(car_weights) else None,
            class_names_path=car_class_names if os.path.exists(car_class_names) else None,
            num_classes=num_car_classes,
            device=device
        )
        
        # Brand classifier (EfficientNetV2-S best checkpoint — uses test_cars.txt)
        self.brand_confidence_threshold = brand_confidence_threshold
        brand_weights = os.path.join(PROJECT_ROOT, 'weights', 'brand_classifier_best.pth')
        brand_class_names = os.path.join(PROJECT_ROOT, 'data', 'test_cars.txt')

        if os.path.exists(brand_weights) and os.path.exists(brand_class_names):
            try:
                self.brand_classifier = BrandClassifierBest(
                    weights_path=brand_weights,
                    class_names_path=brand_class_names,
                    device=device
                )
            except Exception as e:
                self.brand_classifier = None
                print(f"Brand classifier failed to load: {e}")
        else:
            self.brand_classifier = None
            print("Brand classifier not available (brand_classifier_best.pth or test_cars.txt missing)")

        # Brand-specific model classifiers (e.g. Audi -> A3/A4/A6)
        sub_weights_dir = os.path.join(PROJECT_ROOT, 'weights')
        try:
            self.brand_sub_classifier = BrandSubClassifierRouter(
                weights_dir=sub_weights_dir,
                device=device
            )
            available = self.brand_sub_classifier.available_brands()
            if available:
                print(f"Brand sub-classifiers ready ({len(available)} brands)")
            else:
                print("Brand sub-classifiers not found in weights directory")
        except Exception as e:
            self.brand_sub_classifier = None
            print(f"Brand sub-classifiers failed to load: {e}")
        
        # CLIP zero-shot classifier (additional fallback for unknown brands)
        if clip_brand_list is None:
            clip_brand_list = os.path.join(PROJECT_ROOT, 'data', 'clip_brands.txt')
        
        try:
            self.clip_classifier = CLIPBrandClassifier(
                brand_list_path=clip_brand_list,
                device=device
            )
        except Exception as e:
            self.clip_classifier = None
            print(f"CLIP brand classifier not available: {e}")
        
        self.plate_ocr = PlateOCR(engine='auto')
        self.country_identifier = CountryIdentifier()
        
        elapsed = time.time() - start
        print(f"Pipeline ready in {elapsed:.1f}s")
    
    def process_image(self, image, return_annotated=True):
        """
        Process a single image through the full pipeline.
        
        Args:
            image: PIL Image, file path, or numpy array
            return_annotated: Whether to return an annotated image
            
        Returns:
            dict with per-vehicle results
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Step 1: Detect vehicles and plates
        detections = self.detector.detect(image, return_crops=True)
        
        # Step 2: Process each vehicle
        vehicle_results = []
        
        for i, vehicle in enumerate(detections['vehicles']):
            result = {
                'vehicle_bbox': vehicle['bbox'],
                'vehicle_type': vehicle['class'],
                'vehicle_det_confidence': vehicle['confidence'],
            }
            
            # Classify car model + brand
            if 'crop' in vehicle:
                car_result = self.car_classifier.predict(vehicle['crop'])
                result['car_make_model'] = car_result['make_model']
                result['car_confidence'] = car_result['confidence']
                result['car_top_k'] = car_result['top_k']
                
                # Brand prediction (EfficientNet)
                if self.brand_classifier is not None:
                    brand_result = self.brand_classifier.predict(vehicle['crop'])
                    result['brand'] = brand_result['make_model']
                    result['brand_confidence'] = brand_result['confidence']
                    result['brand_top_k'] = brand_result['top_k']
                
                # CLIP fallback (zero-shot, works for any brand)
                if self.clip_classifier is not None:
                    clip_result = self.clip_classifier.predict(vehicle['crop'])
                    result['clip_brand'] = clip_result['brand']
                    result['clip_brand_confidence'] = clip_result['confidence']
                    result['clip_brand_top_k'] = clip_result['top_k']

                # Brand-specific model classifier (strict routing):
                # only run the small model for the top-1 brand predicted by
                # brand_classifier_best.pth. Do not use CLIP/top-k brands.
                if self.brand_sub_classifier is not None:
                    routing_brand = result.get('brand')
                    if routing_brand and self.brand_sub_classifier.has_brand(routing_brand):
                        sub_result = self.brand_sub_classifier.predict(
                            image=vehicle['crop'],
                            brand=routing_brand,
                            top_k=3
                        )
                        if sub_result.get('available'):
                            result['brand_model'] = sub_result.get('make_model')
                            result['brand_model_confidence'] = sub_result.get('confidence')
                            result['brand_model_top_k'] = sub_result.get('top_k')
                            result['brand_model_brand'] = sub_result.get('brand')
                            result['brand_subclassifier_results'] = [
                                {
                                    'brand': sub_result.get('brand'),
                                    'make_model': sub_result.get('make_model'),
                                    'confidence': sub_result.get('confidence'),
                                    'top_k': sub_result.get('top_k'),
                                    'weights_path': sub_result.get('weights_path'),
                                    'source': 'brand_top1',
                                    'source_confidence': result.get('brand_confidence'),
                                }
                            ]
                            # Explicit third output for API/UI consumers
                            result['model'] = sub_result.get('make_model')
                            result['model_confidence'] = sub_result.get('confidence')
                            result['model_brand'] = sub_result.get('brand')
                            result['model_source'] = 'brand_sub_classifier'
            
            # Find associated plates
            associated_plates = [
                p for p in detections['plates'] if p.get('vehicle_idx') == i
            ]
            
            result['plates'] = []
            for plate in associated_plates:
                plate_result = {
                    'plate_bbox': plate['bbox'],
                    'plate_det_confidence': plate['confidence']
                }
                
                # Read plate text
                if 'crop' in plate:
                    ocr_result = self.plate_ocr.read_plate(plate['crop'])
                    plate_result['text'] = ocr_result['text']
                    plate_result['ocr_confidence'] = ocr_result['confidence']
                    
                    # Identify country
                    if ocr_result['text']:
                        country_result = self.country_identifier.identify(
                            plate_text=ocr_result['text'],
                            plate_image=plate['crop']
                        )
                        plate_result['country'] = country_result['country']
                        plate_result['country_code'] = country_result['country_code']
                        plate_result['country_confidence'] = country_result['confidence']
                        plate_result['country_method'] = country_result['method']
                
                result['plates'].append(plate_result)
            
            vehicle_results.append(result)
        
        # Also process unassociated plates (vehicle_idx == -1)
        unassociated_plates = [
            p for p in detections['plates'] if p.get('vehicle_idx', -1) == -1
        ]
        
        standalone_plates = []
        for plate in unassociated_plates:
            plate_result = {
                'plate_bbox': plate['bbox'],
                'plate_det_confidence': plate['confidence']
            }
            
            if 'crop' in plate:
                ocr_result = self.plate_ocr.read_plate(plate['crop'])
                plate_result['text'] = ocr_result['text']
                plate_result['ocr_confidence'] = ocr_result['confidence']
                
                if ocr_result['text']:
                    country_result = self.country_identifier.identify(
                        plate_text=ocr_result['text'],
                        plate_image=plate['crop']
                    )
                    plate_result['country'] = country_result['country']
                    plate_result['country_code'] = country_result['country_code']
                    plate_result['country_confidence'] = country_result['confidence']
            
            standalone_plates.append(plate_result)
        
        output = {
            'vehicles': vehicle_results,
            'standalone_plates': standalone_plates,
            'total_vehicles': len(vehicle_results),
            'total_plates': len(detections['plates']),
        }
        
        # Annotated image
        if return_annotated:
            output['annotated_image'] = self._draw_annotations(image, output)
        
        return output
    
    def process_video(self, video_path, progress_callback=None):
        """
        Process an entire video through the pipeline.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional function(frame_idx, total) for progress
            
        Returns:
            dict with:
                'video_info': video metadata
                'frame_results': per-frame results
                'summary': aggregated best results
        """
        # Extract frames
        extraction = self.video_processor.extract_frames(video_path)
        frames = extraction['frames']
        total = len(frames)
        
        frame_results = []
        all_vehicles = {}  # Track unique vehicles across frames
        
        for idx, frame_data in enumerate(frames):
            if progress_callback:
                progress_callback(idx + 1, total)
            
            result = self.process_image(
                frame_data['image'],
                return_annotated=False
            )
            
            result['frame_number'] = frame_data['frame_number']
            result['timestamp'] = frame_data['timestamp']
            
            frame_results.append(result)
            
            # Track best detections
            for v in result['vehicles']:
                key = v.get('car_make_model', 'Unknown')
                if key not in all_vehicles or v.get('car_confidence', 0) > all_vehicles[key]['car_confidence']:
                    all_vehicles[key] = v
        
        # Build summary
        summary = self._build_summary(frame_results)
        
        return {
            'video_info': {
                'fps': extraction['video_fps'],
                'duration': extraction['duration'],
                'total_frames': extraction['total_frames'],
                'frames_analyzed': total
            },
            'frame_results': frame_results,
            'summary': summary
        }
    
    def _build_summary(self, frame_results):
        """Aggregate results across all frames into a summary."""
        all_cars = {}
        all_plates = {}
        
        for frame in frame_results:
            for vehicle in frame.get('vehicles', []):
                car = vehicle.get('car_make_model', 'Unknown')
                conf = vehicle.get('car_confidence', 0)
                
                if car not in all_cars or conf > all_cars[car]['confidence']:
                    all_cars[car] = {
                        'make_model': car,
                        'confidence': conf,
                        'vehicle_type': vehicle.get('vehicle_type', ''),
                        'frame_timestamp': frame.get('timestamp', 0)
                    }
                
                for plate in vehicle.get('plates', []):
                    text = plate.get('text', '')
                    if text:
                        ocr_conf = plate.get('ocr_confidence', 0)
                        if text not in all_plates or ocr_conf > all_plates[text]['ocr_confidence']:
                            all_plates[text] = {
                                'text': text,
                                'ocr_confidence': ocr_conf,
                                'country': plate.get('country', 'Unknown'),
                                'country_code': plate.get('country_code', ''),
                                'associated_car': car,
                                'frame_timestamp': frame.get('timestamp', 0)
                            }
        
        return {
            'unique_vehicles': list(all_cars.values()),
            'unique_plates': list(all_plates.values()),
            'total_unique_vehicles': len(all_cars),
            'total_unique_plates': len(all_plates)
        }
    
    def _draw_annotations(self, image, results):
        """Draw bounding boxes and labels on the image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 11)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font
        
        # Draw vehicles
        for v in results.get('vehicles', []):
            bbox = v['vehicle_bbox']
            x1, y1, x2, y2 = bbox
            
            # Vehicle bounding box (cyan)
            draw.rectangle([x1, y1, x2, y2], outline='#00E5FF', width=2)
            
            # Car model label — use the highest confidence brand
            brand_conf = v.get('brand_confidence') or 0
            clip_conf = v.get('clip_brand_confidence') or 0
            if brand_conf >= clip_conf and v.get('brand'):
                car_label = v.get('brand')
                car_conf = brand_conf
            elif v.get('clip_brand'):
                car_label = v.get('clip_brand')
                car_conf = clip_conf
            else:
                car_label = 'Unknown'
                car_conf = 0

            # If we have a brand-specific model prediction, prefer showing it.
            if v.get('brand_model'):
                base_brand = v.get('brand_model_brand') or v.get('brand') or car_label
                car_label = f"{base_brand} {v.get('brand_model')}".strip()
                car_conf = v.get('brand_model_confidence', car_conf)

            label = f"{car_label} ({car_conf:.0%})"
            
            text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
            draw.rectangle(text_bbox, fill='#00E5FF')
            draw.text((x1, y1 - 18), label, fill='black', font=font)
            
            # Draw associated plates
            for plate in v.get('plates', []):
                px1, py1, px2, py2 = plate['plate_bbox']
                
                # Plate bounding box (green)
                draw.rectangle([px1, py1, px2, py2], outline='#00FF00', width=2)
                
                # Plate text label
                plate_text = plate.get('text', '')
                country = plate.get('country', '')
                if plate_text:
                    p_label = f"{plate_text}"
                    if country and country != 'Unknown':
                        p_label += f" [{country}]"
                    
                    text_bbox = draw.textbbox((px1, py1 - 16), p_label, font=font_small)
                    draw.rectangle(text_bbox, fill='#00FF00')
                    draw.text((px1, py1 - 16), p_label, fill='black', font=font_small)
        
        return img


if __name__ == "__main__":
    print("VehicleRecognitionPipeline")
    print("=" * 50)
    print("Usage:")
    print("  pipeline = VehicleRecognitionPipeline()")
    print("  result = pipeline.process_image('car.jpg')")
    print("  result = pipeline.process_video('dashcam.mp4')")
