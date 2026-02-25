# Multi-Task AI Network for Vehicle Model Classification & License Plate Recognition

A single end-to-end multi-task neural network that performs **vehicle model classification** and **license plate recognition** from a single image, served through a web interface.

## Architecture

```
Input Image (224×224)
    │
    ▼
┌──────────────────────────┐
│  Shared Backbone         │
│  (EfficientNet-B0)       │
│  Pretrained on ImageNet  │
└────────┬─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Global │ │ Feature  │
│ AvgPool│ │ Map 7×7  │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌────────┐ ┌──────────┐     ┌──────────┐
│Vehicle │ │  Plate   │────►│  Plate   │
│ClassHd │ │  DetHd   │crop │  OCR Hd  │
└───┬────┘ └────┬─────┘     └────┬─────┘
    │           │                │
    ▼           ▼                ▼
 Car Model   BBox [x,y,w,h]   Plate Text
 (196 cls)                    (EasyOCR)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
python backend/app.py
```
Open **http://localhost:8000** in your browser.

### 3. Train the Model (Optional)
Place your datasets in:
- `data/stanford_cars/train/` and `data/stanford_cars/test/` (vehicle images organized by class)
- `data/turkish_plates/images/train/` and `data/turkish_plates/labels/train/` (YOLO format)

```bash
python -m training.train
```

## Project Structure
```
├── model/
│   ├── backbone.py        # EfficientNet-B0 shared feature extractor
│   ├── heads.py           # Classification, Detection, OCR heads
│   ├── multitask_net.py   # Combined multi-task model
│   ├── losses.py          # Uncertainty-weighted multi-task loss
│   └── inference.py       # Single-image inference pipeline
├── training/
│   ├── dataset.py         # Dataset loaders (Stanford Cars, Turkish Plates)
│   ├── train.py           # 3-phase training loop
│   └── config.py          # Hyperparameters for RTX 4060 Ti
├── backend/
│   └── app.py             # FastAPI server
├── frontend/
│   ├── index.html         # Upload & results UI
│   ├── style.css          # Dark glassmorphism theme
│   └── script.js          # Frontend logic
├── weights/               # Model checkpoints
└── requirements.txt
```

## Training Strategy

| Phase | Description | Backbone | Epochs |
|-------|-------------|----------|--------|
| 1 | Classification head warmup | Frozen | 10 |
| 2 | Detection head warmup | Frozen | 10 |
| 3 | Joint fine-tuning (alternating batches) | Unfrozen | 20 |

The model uses **uncertainty-weighted multi-task loss** (Kendall et al., 2018) to automatically balance task contributions during joint training.

## Key Techniques
- **Multi-task learning** with shared backbone + task-specific heads
- **Disjoint dataset training** via alternating batches with masked losses
- **Uncertainty weighting** for automatic loss balancing
- **Differential learning rates** (1e-5 backbone, 3e-4 heads)
- **Mixed precision training** (AMP) for RTX 4060 Ti optimization
- **GIoU loss** for better bounding box regression
- **EasyOCR** for plate text reading (no OCR training data needed)

## Datasets
- **Vehicle Classification**: Stanford Cars (196 classes, ~16k images)
- **Plate Detection**: Turkish License Plate Dataset (YOLO format, ~3GB)

## Tech Stack
PyTorch • EfficientNet-B0 • EasyOCR • FastAPI • Vanilla JS
