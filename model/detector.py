"""
detector.py - YOLOv8 Vehicle & License Plate Detection

Uses Ultralytics YOLOv8 to detect:
1. Vehicles (car, truck, bus, motorcycle) - pretrained COCO
2. License plates - custom fine-tuned model

Both detections run on each frame, returning bounding boxes
and cropped regions for downstream processing.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")


# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


class VehiclePlateDetector:
    """
    Detect vehicles and license plates in images using YOLOv8.
    
    Uses two models:
    - Vehicle detection: YOLOv8n pretrained on COCO (built-in)
    - Plate detection: YOLOv8n fine-tuned on plate dataset (custom weights)
    
    Usage:
        detector = VehiclePlateDetector(plate_weights='weights/plate_detector.pt')
        results = detector.detect(pil_image)
    """
    
    def __init__(self, 
                 vehicle_model='yolov8n.pt',
                 plate_weights=None,
                 vehicle_conf=0.4,
                 plate_conf=0.3,
                 device=None):
        """
        Args:
            vehicle_model: YOLOv8 model name or path for vehicle detection
            plate_weights: Path to fine-tuned plate detection weights
            vehicle_conf: Confidence threshold for vehicle detections
            plate_conf: Confidence threshold for plate detections
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if YOLO is None:
            raise ImportError("ultralytics is required. Install: pip install ultralytics")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        
        # Load vehicle detection model (pretrained COCO)
        self.vehicle_model = YOLO(vehicle_model)
        print(f"Loaded vehicle detector: {vehicle_model}")
        
        # Load plate detection model (custom or fallback)
        if plate_weights and Path(plate_weights).exists():
            self.plate_model = YOLO(plate_weights)
            print(f"Loaded plate detector: {plate_weights}")
        else:
            # Use vehicle model as fallback - won't detect plates separately
            # but system still works (plate detection from vehicle crop)
            self.plate_model = None
            print("WARNING: No plate detection weights found. "
                  "Plate detection will rely on OCR scanning the vehicle crop.")
    
    def detect(self, image, return_crops=True):
        """
        Detect vehicles and plates in a single image.
        
        Args:
            image: PIL Image or numpy array
            return_crops: If True, include cropped regions in results
            
        Returns:
            dict with:
                'vehicles': List of vehicle detections
                'plates': List of plate detections  
                'frame_annotated': Annotated PIL Image (optional)
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        h, w = img_np.shape[:2]
        
        # --- Vehicle Detection ---
        vehicle_results = self.vehicle_model(
            img_np, 
            conf=self.vehicle_conf,
            device=self.device,
            verbose=False
        )[0]
        
        vehicles = []
        for box in vehicle_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASS_IDS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                vehicle = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(conf, 4),
                    'class': VEHICLE_CLASS_IDS[cls_id],
                    'class_id': cls_id
                }
                
                if return_crops:
                    # Clamp coordinates
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w, x2), min(h, y2)
                    crop = Image.fromarray(img_np[cy1:cy2, cx1:cx2])
                    vehicle['crop'] = crop
                
                vehicles.append(vehicle)
        
        # --- Plate Detection ---
        plates = []
        
        if self.plate_model is not None:
            plate_results = self.plate_model(
                img_np,
                conf=self.plate_conf,
                device=self.device,
                verbose=False
            )[0]
            
            for box in plate_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                plate = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(conf, 4)
                }
                
                if return_crops:
                    # Pad the plate crop to avoid cutting off edge characters
                    pad = 10
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                    crop = Image.fromarray(img_np[cy1:cy2, cx1:cx2])
                    plate['crop'] = crop
                
                # Associate plate with nearest vehicle
                plate['vehicle_idx'] = self._find_nearest_vehicle(
                    plate['bbox'], vehicles
                )
                
                plates.append(plate)
        else:
            # Fallback: try to detect plates within each vehicle crop
            plates = self._scan_vehicles_for_plates(vehicles, img_np)
        
        return {
            'vehicles': vehicles,
            'plates': plates,
            'image_size': (w, h)
        }
    
    def _find_nearest_vehicle(self, plate_bbox, vehicles):
        """Find which vehicle a plate belongs to based on overlap."""
        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        best_idx = -1
        for i, v in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = v['bbox']
            # Check if plate center is inside vehicle bbox
            if vx1 <= plate_center[0] <= vx2 and vy1 <= plate_center[1] <= vy2:
                best_idx = i
                break
        
        return best_idx
    
    def _scan_vehicles_for_plates(self, vehicles, img_np):
        """
        Fallback: When no plate detector is available, 
        scan the lower portion of each vehicle crop for plates.
        Returns estimated plate regions.
        """
        plates = []
        h, w = img_np.shape[:2]
        
        for i, vehicle in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            
            # License plates are typically in the lower 40% of a vehicle
            plate_y1 = int(vy1 + (vy2 - vy1) * 0.6)
            plate_y2 = vy2
            
            # Clamp
            plate_y1 = max(0, plate_y1)
            plate_y2 = min(h, plate_y2)
            plate_x1 = max(0, vx1)
            plate_x2 = min(w, vx2)
            
            crop = Image.fromarray(img_np[plate_y1:plate_y2, plate_x1:plate_x2])
            
            plates.append({
                'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                'confidence': 0.5,  # Low confidence since it's a heuristic
                'crop': crop,
                'vehicle_idx': i,
                'is_estimated': True
            })
        
        return plates


if __name__ == "__main__":
    print("VehiclePlateDetector - YOLOv8-based Detection")
    print("=" * 50)
    
    # Quick test with a dummy image
    if YOLO is not None:
        detector = VehiclePlateDetector(plate_weights=None)
        dummy = Image.new('RGB', (640, 480), color='gray')
        results = detector.detect(dummy)
        print(f"Vehicles detected: {len(results['vehicles'])}")
        print(f"Plates detected: {len(results['plates'])}")
    else:
        print("Install ultralytics to test: pip install ultralytics")
