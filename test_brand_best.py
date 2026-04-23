import os
import sys
import torch
import timm
from PIL import Image
from torchvision import transforms

def main():
    print("=" * 50)
    print("Testing brand_classifier_best.pth")
    print("=" * 50)
    
    weights_path = 'weights/brand_classifier_best.pth'
    class_names_path = 'data/test_cars.txt'
    test_image_path = 'test_car.jpg'
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
        
    if not os.path.exists(class_names_path):
        print(f"Error: {class_names_path} not found.")
        return
    
    # Load class names
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = sorted([line.strip() for line in f if line.strip()])
    num_classes = len(class_names)
    print(f"Discovered {num_classes} classes from {class_names_path}")
    
    # The 'best' model was saved as a raw timm model without the nn.Sequential custom head
    model = timm.create_model('tf_efficientnetv2_s', num_classes=num_classes)
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image {test_image_path} not found. Please provide one.")
        return
        
    print(f"\nProcessing {test_image_path}...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(test_image_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        
    top_probs, top_indices = probs.topk(5, dim=1)
    
    print("\nResults:")
    best_idx = top_indices[0][0].item()
    print(f"  Best Match: {class_names[best_idx]} (Confidence: {top_probs[0][0].item():.2%})")
    
    print("\n  Top 5 Predictions:")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"    {class_names[idx.item()]}: {prob.item():.2%}")

if __name__ == "__main__":
    main()
