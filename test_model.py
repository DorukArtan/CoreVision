"""
test_model.py - Hierarchical brand -> model test script

Flow:
1) Predict brand with brand_classifier_best.pth
2) Route to matching small model checkpoint (Audi/opel/...)
3) Predict brand-specific model (A4, Passat, etc.)
"""

import argparse
import os
import sys
from typing import List

import torch
import timm
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "model"))

from model_sub_classifier import BrandSubClassifierRouter


def load_brand_classes(class_names_path: str) -> List[str]:
    with open(class_names_path, "r", encoding="utf-8") as f:
        # Matches training order used in existing codebase.
        return sorted([line.strip() for line in f if line.strip()])


def run_brand_classifier(
    image_path: str,
    weights_path: str,
    class_names_path: str,
    device: str,
    top_k: int = 5,
):
    class_names = load_brand_classes(class_names_path)
    num_classes = len(class_names)

    model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    k = min(max(1, top_k), num_classes)
    top_probs, top_indices = probs.topk(k, dim=1)
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append(
            {
                "brand": class_names[idx.item()],
                "confidence": round(prob.item(), 4),
            }
        )

    return {
        "best_brand": predictions[0]["brand"],
        "best_confidence": predictions[0]["confidence"],
        "top_k": predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test hierarchical brand -> brand-specific-model inference"
    )
    parser.add_argument(
        "--image",
        default="test_car.jpg",
        help="Path to test image (default: test_car.jpg)",
    )
    parser.add_argument(
        "--brand-weights",
        default="weights/brand_classifier_best.pth",
        help="Path to large brand classifier weights",
    )
    parser.add_argument(
        "--brand-classes",
        default="data/test_cars.txt",
        help="Path to brand class names",
    )
    parser.add_argument(
        "--sub-weights-dir",
        default="weights",
        help="Directory with *_model_classifier.pth files",
    )
    parser.add_argument(
        "--force-brand",
        default=None,
        help="Skip large brand classifier and force this brand (e.g. audi)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k to print for the brand-specific model classifier",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.sub_weights_dir):
        raise FileNotFoundError(f"Sub weights directory not found: {args.sub_weights_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("Hierarchical Brand -> Model Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image: {args.image}")

    if args.force_brand:
        selected_brand = args.force_brand
        print(f"\n[Step 1] Skipped brand classifier (--force-brand={selected_brand})")
    else:
        if not os.path.exists(args.brand_weights):
            raise FileNotFoundError(f"Brand weights not found: {args.brand_weights}")
        if not os.path.exists(args.brand_classes):
            raise FileNotFoundError(f"Brand classes not found: {args.brand_classes}")

        brand_result = run_brand_classifier(
            image_path=args.image,
            weights_path=args.brand_weights,
            class_names_path=args.brand_classes,
            device=device,
            top_k=5,
        )
        selected_brand = brand_result["best_brand"]

        print("\n[Step 1] Large brand classifier result")
        print(f"Predicted brand: {selected_brand} ({brand_result['best_confidence']:.2%})")
        print("Top-5 brands:")
        for item in brand_result["top_k"]:
            print(f"  - {item['brand']}: {item['confidence']:.2%}")

    router = BrandSubClassifierRouter(weights_dir=args.sub_weights_dir, device=device)
    available = router.available_brands()
    print(f"\nAvailable sub-model brands: {', '.join(available) if available else 'None'}")

    if args.force_brand:
        print(f"\n[Step 2] Brand-specific model classifier for '{selected_brand}'")
        sub_result = router.predict(args.image, selected_brand, top_k=args.top_k)
        if not sub_result.get("available"):
            print(sub_result.get("reason", "No matching sub-classifier"))
            available_now = sub_result.get("available_brands") or []
            if available_now:
                print(f"Available sub-model brands: {', '.join(available_now)}")
            return

        print(
            f"Predicted model: {sub_result['make_model']} "
            f"({sub_result['confidence']:.2%})"
        )
        print(f"Weights used: {sub_result['weights_path']}")
        print("Top model predictions:")
        for item in sub_result["top_k"]:
            print(f"  - {item['make_model']}: {item['confidence']:.2%}")
    else:
        print("\n[Step 2] Sub-classifier results for top-5 predicted brands")
        matched = False
        for item in brand_result["top_k"]:
            candidate_brand = item["brand"]
            candidate_conf = item["confidence"]
            if not router.has_brand(candidate_brand):
                continue
            matched = True
            sub_result = router.predict(args.image, candidate_brand, top_k=args.top_k)
            if not sub_result.get("available"):
                continue
            print(
                f"Brand {candidate_brand} ({candidate_conf:.2%}) -> "
                f"{sub_result['make_model']} ({sub_result['confidence']:.2%})"
            )
            print(f"  Weights: {sub_result['weights_path']}")
            top_models = ", ".join(
                f"{x['make_model']} ({x['confidence']:.2%})"
                for x in sub_result["top_k"]
            )
            print(f"  Top models: {top_models}")

        if not matched:
            print("No top-5 brand matches with available sub-model classifiers.")

    print("\nDone. The hierarchical brand -> model path is working.")


if __name__ == "__main__":
    main()
