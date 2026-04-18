"""
clip_brand_classifier.py - Zero-Shot Vehicle Brand Classification using CLIP

Uses OpenAI CLIP (via open_clip) to identify vehicle brands without any
task-specific training. Works by comparing vehicle images against text
descriptions of each brand.

Key advantage: Can recognize ANY brand (BYD, Tesla, etc.) by simply
adding the brand name to the text list — no retraining needed.
"""

import os
import torch
from PIL import Image

try:
    import open_clip
except ImportError:
    open_clip = None
    print("WARNING: open-clip-torch not installed. Run: pip install open-clip-torch")


class CLIPBrandClassifier:
    """
    Zero-shot vehicle brand classifier using CLIP.
    
    Instead of training on a fixed set of brands, CLIP compares vehicle
    images against text descriptions like "a photo of a Toyota car".
    This means ANY brand can be recognized by adding it to the brand list.
    
    Usage:
        classifier = CLIPBrandClassifier(
            brand_list_path='data/clip_brands.txt'
        )
        result = classifier.predict(vehicle_crop)
        print(result['brand'], result['confidence'])
    """
    
    # CLIP model choices (speed vs accuracy trade-off):
    # - ViT-B-32: fastest, good enough for brand classification (~400MB)
    # - ViT-B-16: better accuracy, slower (~600MB)
    # - ViT-L-14: best accuracy, slowest (~900MB)
    DEFAULT_MODEL = 'ViT-B-32'
    DEFAULT_PRETRAINED = 'laion2b_s34b_b79k'
    
    def __init__(self, brand_list_path=None, brands=None,
                 model_name=None, pretrained=None, device=None):
        """
        Args:
            brand_list_path: Path to text file with brand names (one per line)
            brands: List of brand names (alternative to file)
            model_name: CLIP model variant (default: ViT-B-32)
            pretrained: Pretrained weights name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if open_clip is None:
            raise ImportError("open-clip-torch is required. Install: pip install open-clip-torch")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = model_name or self.DEFAULT_MODEL
        pretrained = pretrained or self.DEFAULT_PRETRAINED
        
        # Load CLIP model
        print(f"Loading CLIP model ({model_name})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        print(f"CLIP model loaded on {self.device}")
        
        # Load brand list
        self.brands = self._load_brands(brand_list_path, brands)
        
        # Pre-compute text embeddings for all brands (done once, reused forever)
        self._text_features = self._encode_brands()
        print(f"CLIP brand classifier ready ({len(self.brands)} brands)")
    
    def _load_brands(self, brand_list_path, brands_list):
        """Load brand names from file or list."""
        if brands_list is not None:
            return [b.strip().lower() for b in brands_list if b.strip()]
        
        if brand_list_path is None:
            # Default path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            brand_list_path = os.path.join(project_root, 'data', 'clip_brands.txt')
        
        if not os.path.exists(brand_list_path):
            print(f"WARNING: Brand list not found: {brand_list_path}")
            # Fallback to a minimal built-in list
            return [
                'acura', 'audi', 'bmw', 'byd', 'chevrolet', 'citroen',
                'dodge', 'fiat', 'ford', 'honda', 'hyundai', 'jaguar',
                'jeep', 'kia', 'land rover', 'lexus', 'mazda',
                'mercedes benz', 'mini', 'mitsubishi', 'nissan', 'opel',
                'peugeot', 'porsche', 'renault', 'seat', 'skoda',
                'subaru', 'suzuki', 'tesla', 'togg', 'toyota',
                'volkswagen', 'volvo'
            ]
        
        with open(brand_list_path, 'r', encoding='utf-8') as f:
            brands = [line.strip().lower() for line in f if line.strip()]
        
        return brands
    
    def _encode_brands(self):
        """
        Pre-compute text embeddings for all brands.
        
        Uses multiple prompt templates for better accuracy:
        - "a photo of a {brand} car"
        - "a {brand} vehicle"
        - "a {brand} automobile"
        
        The embeddings are averaged across templates (prompt ensembling).
        """
        templates = [
            "a photo of a {} car",
            "a {} vehicle",
            "a {} automobile",
            "a photo of a {} vehicle on the road",
        ]
        
        all_features = []
        
        with torch.no_grad():
            for brand in self.brands:
                # Create text prompts from templates
                texts = [t.format(brand) for t in templates]
                tokens = self.tokenizer(texts).to(self.device)
                
                # Encode and average (prompt ensembling)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                avg_feature = features.mean(dim=0)
                avg_feature = avg_feature / avg_feature.norm()
                
                all_features.append(avg_feature)
        
        # Stack into matrix: (num_brands, embed_dim)
        text_features = torch.stack(all_features)
        return text_features
    
    def predict(self, image, top_k=5):
        """
        Classify a vehicle image by brand using zero-shot CLIP.
        
        Args:
            image: PIL Image of a cropped vehicle
            top_k: Number of top predictions to return
            
        Returns:
            dict with:
                'brand': str - best predicted brand
                'confidence': float - confidence score (0-1)
                'top_k': list of {brand, confidence} dicts
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image = image.convert('RGB')
        
        # Preprocess and encode image
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity → softmax probabilities
            similarity = (image_features @ self._text_features.T).squeeze(0)
            probs = torch.softmax(similarity * 100, dim=0)  # temperature scaling
        
        # Top-k predictions
        top_probs, top_indices = probs.topk(min(top_k, len(self.brands)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'brand': self.brands[idx.item()],
                'confidence': round(prob.item(), 4)
            })
        
        best = predictions[0] if predictions else {'brand': 'Unknown', 'confidence': 0.0}
        
        return {
            'brand': best['brand'],
            'confidence': best['confidence'],
            'top_k': predictions,
            # Compatibility keys (for pipeline integration)
            'make_model': best['brand'],
        }
    
    def add_brand(self, brand_name):
        """
        Dynamically add a new brand at runtime.
        Re-computes text embeddings to include the new brand.
        
        Args:
            brand_name: Brand name to add (e.g., 'rivian')
        """
        brand_name = brand_name.strip().lower()
        if brand_name in self.brands:
            return  # Already exists
        
        self.brands.append(brand_name)
        self._text_features = self._encode_brands()
        print(f"Added brand '{brand_name}' — now tracking {len(self.brands)} brands")


if __name__ == "__main__":
    print("CLIPBrandClassifier - Zero-Shot Vehicle Brand Recognition")
    print("=" * 60)
    
    if open_clip is not None:
        classifier = CLIPBrandClassifier()
        
        # Test with a dummy image
        dummy = Image.new('RGB', (224, 224), color='gray')
        result = classifier.predict(dummy)
        top5 = [(r['brand'], f"{r['confidence']:.4f}") for r in result['top_k']]
        print(f"\nDummy test → Brand: {result['brand']} ({result['confidence']:.4f})")
        print(f"Top-5: {top5}")
    else:
        print("Install open-clip-torch to test: pip install open-clip-torch")
