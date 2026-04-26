"""
model_sub_classifier.py - Brand-specific car model classifiers

Loads per-brand checkpoints (e.g. audi/opel/renault) and routes inference
based on the predicted brand from the large brand classifier.
"""

import os
import glob
import re
import unicodedata
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None


def _normalize_brand_name(text: str) -> str:
    """Normalize brand names for robust dictionary matching."""
    if not text:
        return ""
    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_text.casefold())


class BrandSubClassifier:
    """
    Single brand-specific model classifier.

    Handles two checkpoint styles used in this repo:
    1) Raw timm model state_dict (e.g. classifier.weight keys)
    2) nn.Sequential head + backbone (keys prefixed with 0., 2., 6., ...)
    """

    def __init__(self, weights_path: str, device: Optional[str] = None):
        if timm is None:
            raise ImportError("timm is required. Install: pip install timm")

        self.weights_path = weights_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(weights_path, map_location="cpu")
        self.checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        self.class_names = self._load_class_names(self.checkpoint)
        self.num_classes = max(1, len(self.class_names))
        self.brand = str(
            self.checkpoint.get(
                "brand", self._brand_from_filename(os.path.basename(weights_path))
            )
        ).strip()
        self.backbone_name = str(self.checkpoint.get("backbone", "tf_efficientnetv2_s"))

        input_size = self.checkpoint.get("input_size", 224)
        if isinstance(input_size, (list, tuple)):
            input_size = input_size[0]
        self.input_size = int(input_size)

        state_dict = self._extract_state_dict(checkpoint)
        self.model = self._build_and_load_model(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _brand_from_filename(filename: str) -> str:
        return re.sub(
            r"_model_classifier\.pth$",
            "",
            filename,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint

        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    @staticmethod
    def _load_class_names(checkpoint: Dict) -> List[str]:
        names = checkpoint.get("class_names")
        if isinstance(names, (list, tuple)) and names:
            return [str(x).strip() for x in names if str(x).strip()]

        num_classes = int(checkpoint.get("num_classes", 0) or 0)
        return [f"Class_{i}" for i in range(num_classes)]

    @staticmethod
    def _is_sequential_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
        if not isinstance(state_dict, dict):
            return False
        if any(k.startswith("0.") for k in state_dict.keys()):
            return True
        return "2.weight" in state_dict and "6.weight" in state_dict

    @staticmethod
    def _detect_hidden_dim(checkpoint: Dict, state_dict: Dict[str, torch.Tensor]) -> int:
        hidden_dim = checkpoint.get("hidden_dim")
        if hidden_dim is not None:
            return int(hidden_dim)
        if "2.weight" in state_dict:
            return int(state_dict["2.weight"].shape[0])
        return 512

    @staticmethod
    def _build_sequential_model(backbone_name: str, num_classes: int, hidden_dim: int):
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feature_dim = backbone.num_features
        return nn.Sequential(
            backbone,
            nn.Dropout(0.3),
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_and_load_model(self, state_dict: Dict[str, torch.Tensor]):
        if self._is_sequential_checkpoint(state_dict):
            hidden_dim = self._detect_hidden_dim(self.checkpoint, state_dict)
            model = self._build_sequential_model(
                self.backbone_name, self.num_classes, hidden_dim
            )
        else:
            model = timm.create_model(
                self.backbone_name,
                pretrained=False,
                num_classes=self.num_classes,
            )

        model.load_state_dict(state_dict, strict=True)
        return model

    def predict(self, image, top_k: int = 3) -> Dict:
        """Predict model name for this brand."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        k = min(max(1, top_k), self.num_classes)
        top_probs, top_indices = probs.topk(k, dim=1)

        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            cls_idx = idx.item()
            name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Class_{cls_idx}"
            predictions.append(
                {
                    "make_model": name,
                    "confidence": round(prob.item(), 4),
                    "class_idx": cls_idx,
                }
            )

        best = predictions[0] if predictions else {
            "make_model": "Unknown",
            "confidence": 0.0,
            "class_idx": -1,
        }

        return {
            "brand": self.brand,
            "make_model": best["make_model"],
            "confidence": best["confidence"],
            "class_idx": best["class_idx"],
            "top_k": predictions,
        }


class BrandSubClassifierRouter:
    """
    Routes a brand string to a matching brand-specific model checkpoint.

    Checkpoint discovery pattern: *_model_classifier.pth in weights_dir.
    """

    BRAND_ALIASES = {
        "vw": "volkswagen",
    }

    def __init__(self, weights_dir: str, device: Optional[str] = None):
        self.weights_dir = weights_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._brand_to_weights = self._discover_weights(weights_dir)
        self._loaded_classifiers: Dict[str, BrandSubClassifier] = {}

    @classmethod
    def _canonicalize_brand(cls, brand: str) -> str:
        normalized = _normalize_brand_name(brand)
        return cls.BRAND_ALIASES.get(normalized, normalized)

    def _discover_weights(self, weights_dir: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        pattern = os.path.join(weights_dir, "*_model_classifier.pth")
        for path in sorted(glob.glob(pattern)):
            filename = os.path.basename(path)
            brand_part = re.sub(
                r"_model_classifier\.pth$",
                "",
                filename,
                flags=re.IGNORECASE,
            )
            canonical = self._canonicalize_brand(brand_part)
            if canonical:
                mapping[canonical] = path
        return mapping

    def available_brands(self) -> List[str]:
        return sorted(self._brand_to_weights.keys())

    def has_brand(self, brand: str) -> bool:
        return self._canonicalize_brand(brand) in self._brand_to_weights

    def _get_classifier(self, canonical_brand: str) -> BrandSubClassifier:
        if canonical_brand not in self._loaded_classifiers:
            weights_path = self._brand_to_weights[canonical_brand]
            self._loaded_classifiers[canonical_brand] = BrandSubClassifier(
                weights_path=weights_path,
                device=self.device,
            )
        return self._loaded_classifiers[canonical_brand]

    def predict(self, image, brand: str, top_k: int = 3) -> Dict:
        canonical = self._canonicalize_brand(brand)
        if not canonical or canonical not in self._brand_to_weights:
            return {
                "available": False,
                "requested_brand": brand,
                "available_brands": self.available_brands(),
                "reason": f"No brand-specific classifier found for '{brand}'",
            }

        classifier = self._get_classifier(canonical)
        result = classifier.predict(image=image, top_k=top_k)
        result["available"] = True
        result["weights_path"] = self._brand_to_weights[canonical]
        return result
