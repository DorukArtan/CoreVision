# -*- coding: utf-8 -*-
"""
Fine-tune Brand Classifier — Add Missing European Brands
=========================================================
Run this in Google Colab with a GPU runtime.

Steps:
  1. Mount Google Drive (expects existing weights there)
  2. Scrape car images for missing brands from the web
  3. Expand the classifier head to include new brands
  4. Fine-tune with frozen backbone (fast)
  5. Export new weights + updated class list

Upload your existing files to Google Drive before running:
  - weights/brand_classifier_latest.pth
  - data/vmmrdb_brand_classes.txt
"""

# ============================================================
# Cell 1: Setup & Install Dependencies
# ============================================================

# !pip install -q timm torch torchvision Pillow icrawler

import os
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import timm

# ============================================================
# Cell 2: Mount Google Drive & Configure Paths
# ============================================================

# from google.colab import drive
# drive.mount('/content/drive')

# === CONFIGURE THESE PATHS ===
# Point to your existing weights and class names on Google Drive
DRIVE_BASE = "/content/drive/MyDrive/CoreVision"

EXISTING_WEIGHTS = os.path.join(DRIVE_BASE, "weights", "brand_classifier_latest.pth")
EXISTING_CLASSES = os.path.join(DRIVE_BASE, "data", "vmmrdb_brand_classes.txt")

# Output paths (new weights will be saved here)
OUTPUT_WEIGHTS = os.path.join(DRIVE_BASE, "weights", "brand_classifier_expanded.pth")
OUTPUT_CLASSES = os.path.join(DRIVE_BASE, "data", "brand_classes_expanded.txt")

# Local working directories
DATASET_DIR = "/content/car_brands_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# ============================================================
# Cell 3: Define Missing Brands to Add
# ============================================================

# These brands are common in Turkey/Europe but missing from VMMRdb
NEW_BRANDS = [
    "seat",
    "renault",
    "opel",
    "skoda",
    "dacia",
    "citroen",
    "cupra",
    "togg",
]

# How many images to scrape per brand
IMAGES_PER_BRAND = 200

# Also include some existing brands for balanced training
# (prevents catastrophic forgetting)
EXISTING_BRANDS_TO_INCLUDE = [
    "volkswagen",
    "bmw",
    "mercedes benz",
    "audi",
    "toyota",
    "hyundai",
    "ford",
    "honda",
    "fiat",
    "peugeot",
    "kia",
    "nissan",
    "mazda",
    "volvo",
    "suzuki",
    "mitsubishi",
]

IMAGES_PER_EXISTING_BRAND = 100  # fewer since model already knows these

# ============================================================
# Cell 4: Scrape Car Images Using icrawler
# ============================================================

def scrape_brand_images(brand_name, output_dir, max_num=200):
    """
    Scrape car images from the web for a given brand.
    Uses icrawler (Google/Bing image crawler).
    """
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

    brand_dir = os.path.join(output_dir, brand_name.replace(" ", "_"))
    os.makedirs(brand_dir, exist_ok=True)

    # Check if already scraped
    existing = len([f for f in os.listdir(brand_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    if existing >= max_num * 0.8:
        print(f"  ✓ {brand_name}: already have {existing} images, skipping")
        return brand_dir

    # Search queries designed to get front/side views of cars
    queries = [
        f"{brand_name} car front view",
        f"{brand_name} car",
        f"{brand_name} automobile",
        f"new {brand_name} car",
    ]

    images_per_query = max_num // len(queries) + 1

    for query in queries:
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": brand_dir},
                log_level=40  # ERROR only
            )
            crawler.crawl(
                keyword=query,
                max_num=images_per_query,
                min_size=(200, 200),
            )
        except Exception as e:
            print(f"  Warning: query '{query}' failed: {e}")

        time.sleep(1)  # Rate limiting

    final_count = len([f for f in os.listdir(brand_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    print(f"  ✓ {brand_name}: {final_count} images collected")
    return brand_dir


print("=" * 60)
print("Scraping images for MISSING brands...")
print("=" * 60)
for brand in NEW_BRANDS:
    scrape_brand_images(brand, DATASET_DIR, IMAGES_PER_BRAND)

print("\n" + "=" * 60)
print("Scraping images for EXISTING brands (anti-forgetting)...")
print("=" * 60)
for brand in EXISTING_BRANDS_TO_INCLUDE:
    scrape_brand_images(brand, DATASET_DIR, IMAGES_PER_EXISTING_BRAND)

# ============================================================
# Cell 5: Validate & Clean Downloaded Images
# ============================================================

def validate_images(dataset_dir):
    """Remove corrupted/unreadable images."""
    removed = 0
    total = 0
    for brand_dir in os.listdir(dataset_dir):
        brand_path = os.path.join(dataset_dir, brand_dir)
        if not os.path.isdir(brand_path):
            continue
        for fname in os.listdir(brand_path):
            fpath = os.path.join(brand_path, fname)
            total += 1
            try:
                img = Image.open(fpath).convert("RGB")
                if img.size[0] < 50 or img.size[1] < 50:
                    os.remove(fpath)
                    removed += 1
                    continue
                img.close()
            except Exception:
                os.remove(fpath)
                removed += 1
    print(f"Validated {total} images, removed {removed} corrupted/tiny files")


print("\nValidating downloaded images...")
validate_images(DATASET_DIR)

# ============================================================
# Cell 6: Create Dataset & DataLoaders
# ============================================================

class CarBrandDataset(Dataset):
    """Simple image dataset organized in brand-name folders."""

    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        self.samples = []
        for class_name in class_names:
            folder_name = class_name.replace(" ", "_")
            class_dir = os.path.join(root_dir, folder_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name]
                    ))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# Cell 7: Load Existing Classes & Build Expanded Class List
# ============================================================

# Load existing class names
with open(EXISTING_CLASSES, 'r', encoding='utf-8') as f:
    existing_classes = [line.strip().lower() for line in f if line.strip()]

print(f"Existing brand classes: {len(existing_classes)}")

# Build expanded class list (existing + new)
expanded_classes = list(existing_classes)
for brand in NEW_BRANDS:
    if brand.lower() not in expanded_classes:
        expanded_classes.append(brand.lower())
        print(f"  + Adding new class: {brand}")
    else:
        print(f"  ✓ Already exists: {brand}")

NUM_OLD_CLASSES = len(existing_classes)
NUM_NEW_CLASSES = len(expanded_classes)
print(f"\nExpanded: {NUM_OLD_CLASSES} → {NUM_NEW_CLASSES} classes")

# Create dataset with only the brands we have images for
available_brands = []
for brand in expanded_classes:
    folder = os.path.join(DATASET_DIR, brand.replace(" ", "_"))
    if os.path.isdir(folder) and len(os.listdir(folder)) > 10:
        available_brands.append(brand)

# Use all expanded classes for model, but only train on available
dataset = CarBrandDataset(DATASET_DIR, expanded_classes, transform=train_transform)
print(f"\nTraining samples: {len(dataset)}")

# Print per-class counts
class_counts = {}
for _, label in dataset.samples:
    name = expanded_classes[label]
    class_counts[name] = class_counts.get(name, 0) + 1
for name, count in sorted(class_counts.items()):
    marker = " ← NEW" if name in NEW_BRANDS else ""
    print(f"  {name:25s}: {count:4d} images{marker}")

# Split train/val
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Override transform for validation set
val_set.dataset = CarBrandDataset(DATASET_DIR, expanded_classes, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                        num_workers=2, pin_memory=True)

# ============================================================
# Cell 8: Build Model & Load Existing Weights
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

def build_model(num_classes, hidden_dim=512):
    """Build EfficientNetV2-S — same architecture as CarClassifier."""
    backbone = timm.create_model(
        'tf_efficientnetv2_s',
        pretrained=True,
        num_classes=0
    )
    feature_dim = backbone.num_features  # 1280

    model = nn.Sequential(
        backbone,          # 0
        nn.Dropout(0.3),   # 1
        nn.Linear(feature_dim, hidden_dim),  # 2
        nn.BatchNorm1d(hidden_dim),          # 3
        nn.ReLU(inplace=True),               # 4
        nn.Dropout(0.2),                     # 5
        nn.Linear(hidden_dim, num_classes),  # 6
    )
    return model


# Step 1: Build model with OLD number of classes
model = build_model(NUM_OLD_CLASSES)

# Step 2: Load existing weights
if os.path.exists(EXISTING_WEIGHTS):
    checkpoint = torch.load(EXISTING_WEIGHTS, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print(f"✓ Loaded existing weights ({NUM_OLD_CLASSES} classes)")
else:
    print(f"⚠ No existing weights found at {EXISTING_WEIGHTS}")
    print("  Training from ImageNet pretrained backbone only")

# Step 3: Detect hidden_dim from loaded weights
hidden_dim = model[2].weight.shape[0]
print(f"  Hidden dim: {hidden_dim}")

# Step 4: Expand the final classifier layer to include new classes
if NUM_NEW_CLASSES > NUM_OLD_CLASSES:
    old_fc = model[6]  # Last Linear layer
    new_fc = nn.Linear(hidden_dim, NUM_NEW_CLASSES)

    # Copy old weights
    with torch.no_grad():
        new_fc.weight[:NUM_OLD_CLASSES] = old_fc.weight
        new_fc.bias[:NUM_OLD_CLASSES] = old_fc.bias
        # Initialize new class weights (Xavier init)
        nn.init.xavier_uniform_(new_fc.weight[NUM_OLD_CLASSES:])
        new_fc.bias[NUM_OLD_CLASSES:].zero_()

    model[6] = new_fc
    print(f"✓ Expanded classifier: {NUM_OLD_CLASSES} → {NUM_NEW_CLASSES} classes")

model = model.to(device)

# ============================================================
# Cell 9: Freeze Backbone, Train Only Head
# ============================================================

# Freeze the backbone (layer 0) — only train the classifier head
for param in model[0].parameters():
    param.requires_grad = False

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable: {trainable:,} / {total:,} parameters "
      f"({100 * trainable / total:.1f}%)")

# ============================================================
# Cell 10: Training Loop
# ============================================================

# Hyperparameters
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Use class weights to handle imbalanced data
class_weights = torch.ones(NUM_NEW_CLASSES)
for i, name in enumerate(expanded_classes):
    count = class_counts.get(name, 0)
    if count > 0:
        class_weights[i] = 1.0 / max(count, 1)
    else:
        class_weights[i] = 0.0  # No images for this class
class_weights = class_weights / class_weights.sum() * NUM_NEW_CLASSES
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
best_epoch = 0

print(f"\n{'='*60}")
print(f"Training: {EPOCHS} epochs, lr={LEARNING_RATE}, batch=32")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    model[0].eval()  # Keep backbone in eval mode (frozen BN)

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()

    train_acc = train_correct / train_total
    train_loss = train_loss / train_total

    # --- Validate ---
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = val_correct / val_total if val_total > 0 else 0

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.1%} | "
          f"Val Acc: {val_acc:.1%} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': NUM_NEW_CLASSES,
            'hidden_dim': hidden_dim,
            'accuracy': val_acc,
            'epoch': epoch + 1,
            'class_names': expanded_classes,
        }, OUTPUT_WEIGHTS)
        print(f"  ★ Saved best model (val_acc={val_acc:.1%})")

print(f"\n{'='*60}")
print(f"Training complete! Best: epoch {best_epoch}, val_acc={best_acc:.1%}")
print(f"{'='*60}")

# ============================================================
# Cell 11: Unfreeze Backbone & Fine-tune End-to-End (Optional)
# ============================================================

print("\nPhase 2: Full fine-tuning (unfrozen backbone)...")

# Unfreeze backbone
for param in model[0].parameters():
    param.requires_grad = True

# Lower learning rate for full fine-tuning
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(5):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()
    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = val_correct / val_total if val_total > 0 else 0

    print(f"FT Epoch {epoch+1}/5 | "
          f"Train Acc: {train_acc:.1%} | "
          f"Val Acc: {val_acc:.1%}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': NUM_NEW_CLASSES,
            'hidden_dim': hidden_dim,
            'accuracy': val_acc,
            'epoch': f"ft_{epoch+1}",
            'class_names': expanded_classes,
        }, OUTPUT_WEIGHTS)
        print(f"  ★ Saved best model (val_acc={val_acc:.1%})")

# ============================================================
# Cell 12: Save Updated Class List
# ============================================================

with open(OUTPUT_CLASSES, 'w', encoding='utf-8') as f:
    for name in expanded_classes:
        f.write(name + '\n')

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
print(f"\nNew weights saved to: {OUTPUT_WEIGHTS}")
print(f"New class list saved to: {OUTPUT_CLASSES}")
print(f"Total classes: {NUM_NEW_CLASSES}")
print(f"Best validation accuracy: {best_acc:.1%}")
print(f"\nTo use in your project:")
print(f"  1. Download {OUTPUT_WEIGHTS}")
print(f"     → Place in CoreVision/weights/brand_classifier_latest.pth")
print(f"  2. Download {OUTPUT_CLASSES}")
print(f"     → Place in CoreVision/data/vmmrdb_brand_classes.txt")
print(f"  3. Restart the backend server")
