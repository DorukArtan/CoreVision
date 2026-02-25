"""
dataset.py - Dataset Classes for Multi-Task Training

Two separate datasets for the disjoint training setup:
1. StanfordCarsDataset - Vehicle model classification
2. TurkishPlateDataset - License plate detection (YOLO format)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars dataset for vehicle model classification.
    
    Expected directory structure:
        stanford_cars/
        ├── train/
        │   ├── Audi_A4/
        │   │   ├── 00001.jpg
        │   │   ├── 00002.jpg
        │   │   └── ...
        │   ├── BMW_320i/
        │   │   └── ...
        │   └── ...
        └── test/
            └── (same structure)
    
    If using a flat structure with a labels file, subclass this and
    override __getitem__ and __len__.
    """
    
    def __init__(self, root_dir, split='train', input_size=224):
        """
        Args:
            root_dir: Path to stanford_cars directory
            split: 'train' or 'test'
            input_size: Image resize dimension
        """
        self.root_dir = os.path.join(root_dir, split)
        self.input_size = input_size
        
        # Build image list and class mapping
        self.images = []
        self.labels = []
        self.class_names = []
        
        if os.path.exists(self.root_dir):
            # Get sorted class directories
            class_dirs = sorted([
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])
            
            self.class_names = class_dirs
            
            for class_idx, class_name in enumerate(class_dirs):
                class_path = os.path.join(self.root_dir, class_name)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        
        # Augmentation transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, 224, 224) normalized tensor
            label: int class index
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        image = self.transform(image)
        
        return {
            'image': image,
            'class_label': torch.tensor(label, dtype=torch.long),
            'task': 'classify'
        }


class TurkishPlateDataset(Dataset):
    """
    Turkish License Plate Dataset (YOLO format) for plate detection.
    
    Expected directory structure (YOLO format):
        turkish_plates/
        ├── images/
        │   ├── train/
        │   │   ├── img001.jpg
        │   │   └── ...
        │   └── val/
        │       └── ...
        └── labels/
            ├── train/
            │   ├── img001.txt  (YOLO format: class cx cy w h)
            │   └── ...
            └── val/
                └── ...
    
    YOLO format per line: <class_id> <cx> <cy> <width> <height>
    All values are normalized to [0, 1].
    """
    
    def __init__(self, root_dir, split='train', input_size=224):
        """
        Args:
            root_dir: Path to turkish_plates directory
            split: 'train' or 'val'
            input_size: Image resize dimension
        """
        self.root_dir = root_dir
        self.input_size = input_size
        
        # Find images
        img_dir = os.path.join(root_dir, 'images', split)
        label_dir = os.path.join(root_dir, 'labels', split)
        
        self.images = []
        self.label_files = []
        
        if os.path.exists(img_dir):
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(img_dir, img_name)
                    
                    # Find corresponding label file
                    label_name = os.path.splitext(img_name)[0] + '.txt'
                    label_path = os.path.join(label_dir, label_name)
                    
                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.label_files.append(label_path)
        
        # Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Transform without normalization (for OCR cropping)
        self.to_pil_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def _parse_yolo_label(self, label_path):
        """
        Parse YOLO format label file.
        Takes the first bounding box (assumes one plate per image).
        
        Returns:
            bbox: [cx, cy, w, h] normalized coordinates
        """
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return [0.5, 0.5, 0.1, 0.05]  # Default center bbox
        
        # Take first detection (main plate)
        parts = lines[0].strip().split()
        # Skip class_id (parts[0]), take cx, cy, w, h
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        return [cx, cy, w, h]
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, 224, 224) normalized tensor
            bbox: (4,) normalized [cx, cy, w, h]
            original_image: PIL Image (for OCR)
        """
        img_path = self.images[idx]
        label_path = self.label_files[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_image = self.to_pil_transform(image)
        
        # Parse YOLO bbox
        bbox = self._parse_yolo_label(label_path)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'bbox_label': torch.tensor(bbox, dtype=torch.float32),
            'task': 'detect',
            'img_path': img_path
        }


def create_dataloaders(config):
    """
    Create DataLoaders for both datasets.
    
    Args:
        config: Config object with dataset paths and training params
        
    Returns:
        vehicle_train_loader, vehicle_val_loader,
        plate_train_loader, plate_val_loader
    """
    from torch.utils.data import DataLoader
    
    # Vehicle classification dataset
    vehicle_train = StanfordCarsDataset(
        config.VEHICLE_DATASET_DIR, split='train', input_size=config.INPUT_SIZE
    )
    vehicle_val = StanfordCarsDataset(
        config.VEHICLE_DATASET_DIR, split='test', input_size=config.INPUT_SIZE
    )
    
    # Plate detection dataset
    plate_train = TurkishPlateDataset(
        config.PLATE_DATASET_DIR, split='train', input_size=config.INPUT_SIZE
    )
    plate_val = TurkishPlateDataset(
        config.PLATE_DATASET_DIR, split='val', input_size=config.INPUT_SIZE
    )
    
    # Create loaders
    vehicle_train_loader = DataLoader(
        vehicle_train, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    vehicle_val_loader = DataLoader(
        vehicle_val, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    plate_train_loader = DataLoader(
        plate_train, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    plate_val_loader = DataLoader(
        plate_val, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Vehicle train: {len(vehicle_train)} images, {len(vehicle_train_loader)} batches")
    print(f"Vehicle val:   {len(vehicle_val)} images")
    print(f"Plate train:   {len(plate_train)} images, {len(plate_train_loader)} batches")
    print(f"Plate val:     {len(plate_val)} images")
    
    return vehicle_train_loader, vehicle_val_loader, plate_train_loader, plate_val_loader


if __name__ == "__main__":
    from training.config import Config
    print("Dataset module loaded successfully.")
    print(f"Vehicle dataset dir: {Config.VEHICLE_DATASET_DIR}")
    print(f"Plate dataset dir:   {Config.PLATE_DATASET_DIR}")
