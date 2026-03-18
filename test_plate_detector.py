import cv2
import sys
import os
from ultralytics import YOLO

# 1. Load the YOLOv8 model for plate detection
model_path = os.path.join("weights", "plate_detector.pt")
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

model = YOLO(model_path)

# 2. Grab the test image
image_path = "test_car.jpg"
if not os.path.exists(image_path):
    print(f"Error: test image not found at {image_path}. Please place an image there.")
    sys.exit(1)

print(f"Running prediction on {image_path}...")

# 3. Predict
results = model(image_path)

# 4. Save results to a file instead of popping up a window
for i, r in enumerate(results):
    im_array = r.plot()  # plots the bounding boxes
    output_path = f"detection_result_{i}.jpg"
    cv2.imwrite(output_path, im_array)
    
    print(f"\n✅ Success! The detection result has been saved as '{output_path}'.")
    print("Open it in your file explorer to see the bounding boxes!")
