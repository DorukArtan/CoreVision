import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Import local PlateOCR module
# (Assuming test_ocr.py is in the indrit root and model/ folder is present)
from model.plate_ocr import PlateOCR

# 1. Load the YOLO detector
model_path = os.path.join("weights", "plate_detector.pt")
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

print("Loading Plate Detector...")
detector = YOLO(model_path)

# 2. Load the OCR Engine
print("Loading OCR Engine (this might take a few seconds)...")
# Initialize PlateOCR with automatic engine selection.
# It will prefer PaddleOCR when available, otherwise EasyOCR.
ocr_system = PlateOCR(engine='auto')

# Grab the test image
image_path = "test_car.jpg"
if not os.path.exists(image_path):
    print(f"Error: test image not found at {image_path}")
    sys.exit(1)

print(f"Running prediction & OCR on {image_path}...")
image = cv2.imread(image_path)
results = detector(image)

# 3. Process each detected plate
for i, r in enumerate(results):
    im_array = r.plot()
    
    # Check if any plates were found
    if len(r.boxes) == 0:
        print("No license plates detected in this image.")
        continue
        
    for j, box in enumerate(r.boxes):
        # Extract bounding box coordinates from YOLO
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Expand bounding box slightly (YOLO often cuts off the edges of plates)
        pad = 10
        img_h, img_w = image.shape[:2]
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(img_w, x2 + pad)
        crop_y2 = min(img_h, y2 + pad)
        
        # Crop the plate from the enlarged bounds
        plate_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Save the exact crop being sent to OCR so we can see what the AI is looking at
        cv2.imwrite(f"debug_crop_{j}.jpg", plate_crop)
        
        # Also save preprocessed versions for debugging
        debug_color = ocr_system.preprocess_plate(plate_crop.copy())
        debug_binary = ocr_system.preprocess_plate_binary(plate_crop.copy())
        cv2.imwrite(f"debug_color_{j}.jpg", debug_color)
        cv2.imwrite(f"debug_binary_{j}.jpg", debug_binary)
        
        # Run OCR on the crop (multi-attempt: color + binary)
        ocr_result = ocr_system.read_plate(plate_crop)
        
        plate_text = ocr_result.get('text', '')
        confidence = ocr_result.get('confidence', 0.0)
        
        print(f"\n🚘 Plate #{j+1} Detected!")
        print(f"   => Read Text:   [{plate_text}]")
        print(f"   => Confidence:  {confidence:.2f}")
        
        # Draw the read text above the bounding box on the image
        cv2.putText(im_array, plate_text, (x1, max(y1-10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the final image with boxes and text
    output_path = f"ocr_result_{i}.jpg"
    cv2.imwrite(output_path, im_array)
    print(f"\n✅ Result image with OCR overlay saved to '{output_path}'")
