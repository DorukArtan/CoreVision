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
    print(f"  Brand: {v.get('brand', 'N/A')} ({v.get('brand_confidence', 0):.1%})")
    if v.get('brand_model'):
        print(f"  Brand Model: {v.get('brand_model', 'N/A')} ({v.get('brand_model_confidence', 0):.1%})")
    if v.get('brand_subclassifier_results'):
        print("  Sub-classifier matches:")
        for s in v.get('brand_subclassifier_results', []):
            print(
                f"    - {s.get('brand', 'N/A')}: "
                f"{s.get('make_model', 'N/A')} ({s.get('confidence', 0):.1%}) "
                f"[{s.get('source', 'unknown')}]"
            )
    if v.get('clip_brand'):
        print(f"  CLIP (fallback): {v.get('clip_brand', 'N/A')} ({v.get('clip_brand_confidence', 0):.1%})")
    for p in v.get('plates', []):
        print(f"  Plate: {p.get('text', 'N/A')}")
        country = p.get('country', 'Unknown')
        country_code = p.get('country_code', '')
        if country_code:
            print(f"  Country: {country} ({country_code})")
        else:
            print(f"  Country: {country}")

if not result['vehicles']:
    print("No vehicles detected.")
