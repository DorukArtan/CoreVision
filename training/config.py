"""
config.py - Training Configuration

Hyperparameters tuned for RTX 4060 Ti (8GB / 16GB VRAM).
All paths and settings for training the multi-task model.
"""

import os


class Config:
    """Central configuration for training."""
    
    # ---- Project Paths ----
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
    
    # Dataset paths (update these to your local paths)
    VEHICLE_DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'stanford_cars')
    PLATE_DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'turkish_plates')
    
    # ---- Model ----
    NUM_VEHICLE_CLASSES = 196          # Stanford Cars has 196 classes
    PRETRAINED_BACKBONE = True         # Use ImageNet pretrained weights
    INPUT_SIZE = 224                   # Input image size (224x224)
    
    # ---- Training Phases ----
    # Phase 1: Warmup classification head (backbone frozen)
    PHASE1_EPOCHS = 10
    PHASE1_LR = 1e-3
    PHASE1_BACKBONE_FROZEN = True
    
    # Phase 2: Warmup detection head (backbone frozen)
    PHASE2_EPOCHS = 10
    PHASE2_LR = 1e-3
    PHASE2_BACKBONE_FROZEN = True
    
    # Phase 3: Joint fine-tuning (all params unfrozen)
    PHASE3_EPOCHS = 20
    PHASE3_BACKBONE_LR = 1e-5         # Lower LR for pretrained backbone
    PHASE3_HEAD_LR = 3e-4             # Higher LR for heads
    
    # ---- Optimization ----
    BATCH_SIZE = 12                    # 8GB VRAM: 12, 16GB VRAM: 24
    NUM_WORKERS = 4                    # DataLoader workers
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_NORM = 1.0           # Max gradient norm
    USE_AMP = True                     # Mixed precision training (saves ~40% VRAM)
    
    # ---- Scheduler ----
    SCHEDULER = 'cosine'               # 'cosine' or 'step'
    STEP_LR_GAMMA = 0.1
    STEP_LR_MILESTONES = [15, 25]
    
    # ---- Logging ----
    LOG_INTERVAL = 50                  # Print every N batches
    SAVE_INTERVAL = 5                  # Save checkpoint every N epochs
    
    # ---- Image Normalization (ImageNet) ----
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories."""
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)
        
    @classmethod
    def summary(cls):
        """Print config summary."""
        print("=" * 50)
        print("Training Configuration")
        print("=" * 50)
        print(f"  Vehicle classes:   {cls.NUM_VEHICLE_CLASSES}")
        print(f"  Input size:        {cls.INPUT_SIZE}x{cls.INPUT_SIZE}")
        print(f"  Batch size:        {cls.BATCH_SIZE}")
        print(f"  Mixed precision:   {cls.USE_AMP}")
        print(f"  Phase 1 epochs:    {cls.PHASE1_EPOCHS} (cls warmup)")
        print(f"  Phase 2 epochs:    {cls.PHASE2_EPOCHS} (det warmup)")
        print(f"  Phase 3 epochs:    {cls.PHASE3_EPOCHS} (joint fine-tune)")
        print(f"  Total epochs:      {cls.PHASE1_EPOCHS + cls.PHASE2_EPOCHS + cls.PHASE3_EPOCHS}")
        print(f"  Gradient clipping: {cls.GRADIENT_CLIP_NORM}")
        print("=" * 50)


if __name__ == "__main__":
    Config.summary()
