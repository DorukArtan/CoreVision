"""
car_classifier.py - EfficientNetV2-S Car Model Classification

Fine-tuned on VMMRdb dataset (Vehicle Make & Model Recognition).
Training is done on Google Colab (see notebooks/train_classifier.ipynb).
This module loads the trained weights for inference.
"""

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None
    print("WARNING: timm not installed. Run: pip install timm")


class CarClassifier:
    """
    Classify car make/model from a cropped vehicle image.
    
    Uses EfficientNetV2-S as backbone, fine-tuned on VMMRdb.
    
    Usage:
        classifier = CarClassifier(
            weights_path='weights/car_classifier.pth',
            class_names_path='data/vmmrdb_classes.txt'
        )
        result = classifier.predict(vehicle_crop)
        print(result['make_model'], result['confidence'])
    """
    
    def __init__(self, weights_path=None, class_names_path=None,
                 num_classes=100, hidden_dim=512, device=None):
        """
        Args:
            weights_path: Path to trained model weights (.pth)
            class_names_path: Path to text file with class names (one per line)
            num_classes: Number of car model classes
            hidden_dim: Hidden layer dimension (512 for model classifier, 256 for brand)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if timm is None:
            raise ImportError("timm is required. Install: pip install timm")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Load class names
        self.class_names = self._load_class_names(class_names_path)
        
        # Auto-detect hidden_dim from checkpoint if available
        if weights_path is not None:
            self.hidden_dim = self._detect_hidden_dim(weights_path, hidden_dim)
        
        # Build model
        self.model = self._build_model(num_classes, self.hidden_dim)
        
        # Load trained weights
        if weights_path is not None:
            self._load_weights(weights_path)
        else:
            print("WARNING: No weights loaded — predictions will be random")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self, num_classes, hidden_dim=512):
        """
        Build EfficientNetV2-S with custom classification head.
        
        Architecture:
            EfficientNetV2-S (pretrained backbone)
            → Global Average Pooling
            → Dropout(0.3)
            → FC(1280 → hidden_dim) → BN → ReLU → Dropout(0.2)
            → FC(hidden_dim → num_classes)
        """
        # Load EfficientNetV2-S with pretrained weights (ImageNet)
        backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            num_classes=0  # Remove default classifier
        )
        
        feature_dim = backbone.num_features  # 1280 for EfficientNetV2-S
        
        model = nn.Sequential(
            backbone,
            nn.Dropout(0.3),
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        return model
    
    def _detect_hidden_dim(self, weights_path, default=512):
        """Auto-detect hidden layer dimension from checkpoint weights."""
        import os
        if not os.path.exists(weights_path):
            return default
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state = checkpoint.get('model_state_dict', checkpoint)
            # The hidden FC layer is at index 2 (after backbone + dropout)
            for key in state:
                if key in ('2.weight', 'module.2.weight'):
                    return state[key].shape[0]  # out_features = hidden_dim
            return default
        except Exception:
            return default
    
    def _load_weights(self, weights_path):
        """Load trained weights from checkpoint."""
        import os
        if not os.path.exists(weights_path):
            print(f"WARNING: Weights file not found: {weights_path}")
            return
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'accuracy' in checkpoint:
                print(f"Loaded car classifier (accuracy: {checkpoint['accuracy']:.1%})")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"Loaded car classifier weights from: {weights_path}")
    
    def _load_class_names(self, class_names_path):
        """Load class names from file."""
        if class_names_path is None:
            return None
        
        import os
        if not os.path.exists(class_names_path):
            print(f"WARNING: Class names file not found: {class_names_path}")
            return None
        
        with open(class_names_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(names)} car model class names")
        return names
    
    def predict(self, image, top_k=5):
        """
        Classify a vehicle image.
        
        Args:
            image: PIL Image of a cropped vehicle
            top_k: Number of top predictions to return
            
        Returns:
            dict with:
                'make_model': str - best prediction (e.g., 'Toyota Corolla 2020')
                'confidence': float - confidence score
                'top_k': list of {make_model, confidence} dicts
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image = image.convert('RGB')
        
        # Preprocess
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        
        # Top-k predictions
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            name = self.class_names[idx.item()] if self.class_names else f"Class_{idx.item()}"
            predictions.append({
                'make_model': name,
                'confidence': round(prob.item(), 4)
            })
        
        best = predictions[0] if predictions else {'make_model': 'Unknown', 'confidence': 0.0}
        
        return {
            'make_model': best['make_model'],
            'confidence': best['confidence'],
            'top_k': predictions,
            'class_idx': top_indices[0][0].item()
        }


if __name__ == "__main__":
    print("CarClassifier - EfficientNetV2-S")
    print("=" * 50)
    
    if timm is not None:
        classifier = CarClassifier(weights_path=None, num_classes=100)
        dummy = Image.new('RGB', (224, 224), color='gray')
        result = classifier.predict(dummy)
        print(f"Prediction: {result['make_model']} ({result['confidence']:.4f})")
    else:
        print("Install timm to test: pip install timm")
