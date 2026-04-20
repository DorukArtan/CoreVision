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

from model.video_processor import VideoProcessor
from model.detector import VehiclePlateDetector
from model.car_classifier import CarClassifier
from model.clip_brand_classifier import CLIPBrandClassifier
from model.plate_ocr import PlateOCR
from model.country_identifier import CountryIdentifier


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
        
        # Brand classifier (EfficientNet — primary brand prediction)
        self.brand_confidence_threshold = brand_confidence_threshold
        brand_weights = os.path.join(PROJECT_ROOT, 'weights', 'brand_classifier_latest.pth')
        brand_class_names = os.path.join(PROJECT_ROOT, 'data', 'vmmrdb_brand_classes.txt')
        
        if os.path.exists(brand_weights) and os.path.exists(brand_class_names):
            num_brands = sum(1 for line in open(brand_class_names, 'r', encoding='utf-8') if line.strip())
            self.brand_classifier = CarClassifier(
                weights_path=brand_weights,
                class_names_path=brand_class_names,
                num_classes=num_brands,
                device=device
            )
            print(f"Brand classifier loaded ({num_brands} brands)")
        else:
            self.brand_classifier = None
            print("Brand classifier not available (no weights found)")
        
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
            
            # Car model label — prefer CLIP/brand over raw classifier class
            car_label = v.get('clip_brand') or v.get('brand') or 'Unknown'
            car_conf = v.get('clip_brand_confidence') or v.get('brand_confidence') or 0
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
