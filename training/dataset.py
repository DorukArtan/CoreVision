"""
dataset.py - Dataset Classes for Multi-Task Training

Two separate datasets for the disjoint training setup:
1. StanfordCarsDataset - Vehicle model classification (Kaggle .mat annotations)
2. TurkishPlateDataset - License plate detection (YOLO format, flat structure)
"""

import os
import random
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms
import numpy as np


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars dataset for vehicle model classification.
    
    Supports the Kaggle flat structure with .mat annotation files:
        stanford_cars/
        ├── cars_train/
        │   └── cars_train/   (flat numbered images: 00001.jpg ...)
        ├── cars_test/
        │   └── cars_test/    (flat numbered images)
        └── car_devkit/
            └── devkit/
                ├── cars_train_annos.mat
                ├── cars_test_annos.mat
                └── cars_meta.mat
    """
    
    def __init__(self, root_dir, split='train', input_size=224):
        """
        Args:
            root_dir: Path to stanford_cars directory
            split: 'train' or 'test'
            input_size: Image resize dimension
        """
        self.root_dir = root_dir
        self.input_size = input_size
        self.split = split
        
        # Build image list and class mapping
        self.images = []
        self.labels = []
        self.class_names = []
        
        self._load_dataset(split)
        
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
    
    def _load_dataset(self, split):
        """Load images and labels from .mat annotation file or class folders.
        
        Note: Stanford Cars test annotations don't include class labels,
        so we split the train set 80/20 for train/val instead.
        """
        devkit_dir = os.path.join(self.root_dir, 'car_devkit', 'devkit')
        train_mat = os.path.join(devkit_dir, 'cars_train_annos.mat')
        meta_file = os.path.join(devkit_dir, 'cars_meta.mat')
        
        if os.path.exists(train_mat):
            self._load_from_mat(split, train_mat, meta_file)
        else:
            # Fallback: class subdirectory structure
            self._load_from_class_dirs(split)
    
    def _load_from_mat(self, split, train_mat_file, meta_file,
                       val_ratio=0.2, seed=42):
        """Load from .mat annotation files (Kaggle/Stanford format).
        
        Since the test set has no class labels, we split the train
        annotations into train (80%) and val (20%).
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            print("WARNING: scipy not installed. Run: pip install scipy")
            print("Falling back to class directory structure...")
            self._load_from_class_dirs(split)
            return
        
        # Load class names
        if os.path.exists(meta_file):
            meta = loadmat(meta_file)
            self.class_names = [str(name[0]) for name in meta['class_names'][0]]
        
        # Find train image directory (handles nested: cars_train/cars_train/)
        img_dir = os.path.join(self.root_dir, 'cars_train')
        nested_dir = os.path.join(img_dir, 'cars_train')
        if os.path.exists(nested_dir):
            img_dir = nested_dir
        
        # Load ALL train annotations
        annos = loadmat(train_mat_file)
        annotations = annos['annotations'][0]
        
        all_images = []
        all_labels = []
        
        for anno in annotations:
            fname = str(anno['fname'][0])
            img_path = os.path.join(img_dir, fname)
            
            if os.path.exists(img_path):
                class_id = int(anno['class'][0][0]) - 1
                all_images.append(img_path)
                all_labels.append(class_id)
        
        # Split into train/val deterministically
        n = len(all_images)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split_idx = int(n * (1 - val_ratio))
        
        if split == 'train':
            selected = indices[:split_idx]
        else:  # 'test' or 'val'
            selected = indices[split_idx:]
        
        self.images = [all_images[i] for i in selected]
        self.labels = [all_labels[i] for i in selected]
        
        print(f"  Stanford Cars ({split}): {len(self.images)} images, "
              f"{len(self.class_names)} classes (from {n} total, "
              f"{val_ratio:.0%} val split)")
    
    def _load_from_class_dirs(self, split):
        """Fallback: load from class subdirectory structure."""
        split_dir = os.path.join(self.root_dir, split)
        if not os.path.exists(split_dir):
            print(f"  WARNING: Directory not found: {split_dir}")
            return
        
        class_dirs = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        
        self.class_names = class_dirs
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(split_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)
        
        print(f"  Stanford Cars ({split}): {len(self.images)} images, "
              f"{len(self.class_names)} classes loaded from directories")
    
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
    
    Supports the flat Kaggle structure:
        turkish_plates/
        ├── images/          (1.jpg, 1.txt, 2.jpg, 2.txt, ...)
        └── label/           (1.txt, 2.txt, ...)
    
    Images are in images/, labels are in label/ (YOLO format).
    YOLO format per line: <class_id> <cx> <cy> <width> <height>
    All values are normalized to [0, 1].
    """
    
    def __init__(self, root_dir, split='train', input_size=224,
                 val_ratio=0.15, seed=42):
        """
        Args:
            root_dir: Path to turkish_plates directory
            split: 'train' or 'val'
            input_size: Image resize dimension
            val_ratio: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.root_dir = root_dir
        self.input_size = input_size
        
        # Try standard YOLO structure first, then flat Kaggle structure
        self.images = []
        self.label_files = []
        
        self._find_data_pairs(split, val_ratio, seed)
        
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
    
    def _find_data_pairs(self, split, val_ratio, seed):
        """Find image-label pairs, handling both YOLO and flat structures."""
        
        # Structure 1: Standard YOLO (images/train/, labels/train/)
        yolo_img_dir = os.path.join(self.root_dir, 'images', split)
        yolo_label_dir = os.path.join(self.root_dir, 'labels', split)
        
        if os.path.exists(yolo_img_dir) and os.path.exists(yolo_label_dir):
            self._load_paired_dirs(yolo_img_dir, yolo_label_dir)
            return
        
        # Structure 2: Flat Kaggle (images/ has images, label/ has labels)
        img_dir = os.path.join(self.root_dir, 'images')
        label_dir = os.path.join(self.root_dir, 'label')
        
        if not os.path.exists(label_dir):
            label_dir = os.path.join(self.root_dir, 'labels')
        
        if os.path.exists(img_dir) and os.path.exists(label_dir):
            all_images = []
            all_labels = []
            
            # Find all image-label pairs
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    stem = os.path.splitext(fname)[0]
                    label_path = os.path.join(label_dir, stem + '.txt')
                    if os.path.exists(label_path):
                        all_images.append(os.path.join(img_dir, fname))
                        all_labels.append(label_path)
            
            # Split into train/val
            n = len(all_images)
            indices = list(range(n))
            rng = random.Random(seed)
            rng.shuffle(indices)
            
            split_idx = int(n * (1 - val_ratio))
            
            if split == 'train':
                selected = indices[:split_idx]
            else:
                selected = indices[split_idx:]
            
            self.images = [all_images[i] for i in selected]
            self.label_files = [all_labels[i] for i in selected]
            
            print(f"  Turkish Plates ({split}): {len(self.images)} images "
                  f"(from {n} total, {val_ratio:.0%} val split)")
            return
        
        print(f"  WARNING: No plate data found in {self.root_dir}")
    
    def _load_paired_dirs(self, img_dir, label_dir):
        """Load paired image/label directories."""
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    self.images.append(os.path.join(img_dir, img_name))
                    self.label_files.append(label_path)
    
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
    
    # Plate detection dataset (auto-splits into train/val)
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
