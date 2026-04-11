"""Quick test of the brand classifier pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.pipeline import VehicleRecognitionPipeline

pipeline = VehicleRecognitionPipeline()
result = pipeline.process_image('test_car.jpg')

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
for i, v in enumerate(result['vehicles']):
    print(f"\nVehicle {i+1}:")
    print(f"  Model prediction: {v.get('car_make_model', 'N/A')} ({v.get('car_confidence', 0):.1%})")
    if v.get('brand_fallback'):
        print(f"  Brand (fallback):  {v.get('brand', 'N/A')} ({v.get('brand_confidence', 0):.1%})")
    for p in v.get('plates', []):
        print(f"  Plate: {p.get('text', 'N/A')}")

if not result['vehicles']:
    print("No vehicles detected.")
