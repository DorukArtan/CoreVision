# CoreVision v2 — Video-Based Vehicle Recognition System

## Project Overview

CoreVision is an end-to-end deep learning system that identifies **car brand/model** and reads **license plate text + country of origin** from uploaded videos or images. The system uses entirely self-trained models with no external AI APIs, running inference locally through a FastAPI backend and a web-based frontend.

### Problem Statement

Automatic vehicle identification from visual media is a multi-step challenge requiring object detection, fine-grained image classification, and optical character recognition to work together in a unified pipeline. CoreVision addresses this by combining purpose-trained models into a single coherent system.

---

## System Architecture

```
Video / Image Upload
        │
        ▼
  Frame Extraction (OpenCV)
        │
        ▼
  YOLOv8-nano (fine-tuned)
   ┌────┴────┐
   ▼         ▼
Vehicle    License Plate
 Crop        Crop
   │         │
   ▼         ▼
EfficientNetV2-S    PaddleOCR
(Car Classifier)    (Plate Reader)
   │                  │
   ▼                  ▼
Brand + Model     Plate Text ──► Country Identifier
                                  (Regex + Lookup)
```

The pipeline processes media in five stages:

1. **Frame Extraction** — Video is sampled at a configurable FPS using OpenCV
2. **Object Detection** — YOLOv8-nano detects vehicles and license plates with bounding boxes
3. **Car Classification** — Vehicle crops are classified into brand+model using EfficientNetV2-S
4. **Plate OCR** — License plate crops are read using PaddleOCR (80+ language support)
5. **Country Identification** — Plate text patterns are matched to 60+ countries via regex rules

---

## Models & Training

### Car Brand+Model Classifier

| Attribute       | Details                                    |
|-----------------|--------------------------------------------|
| Architecture    | EfficientNetV2-S (ImageNet pre-trained)    |
| Dataset         | The Car Connection Picture Dataset (Kaggle)|
| Classes         | ~312 brand+model combinations (year-merged)|
| Training        | 2-phase progressive unfreezing on Colab T4 |
| Phase 1         | Epochs 1–5: backbone frozen, head warmup   |
| Phase 2         | Epochs 6–20: full fine-tune with lower LR  |
| Input Size      | 224×224                                    |
| Optimizer       | AdamW with cosine annealing LR scheduler   |
| Augmentation    | RandomResizedCrop, HorizontalFlip, ColorJitter, RandomErasing |

**Year-merging strategy:** Year variants (e.g., "Honda_Civic_2019" and "Honda_Civic_2020") are merged into a single "Honda_Civic" class. This reduces class count and increases images per class, improving accuracy.

### License Plate Detector

| Attribute    | Details                              |
|--------------|--------------------------------------|
| Architecture | YOLOv8-nano (fine-tuned)             |
| Dataset      | Roboflow multi-country plate dataset |
| Target       | 90%+ mAP                            |

### Plate OCR

| Attribute    | Details                                     |
|--------------|---------------------------------------------|
| Engine       | PaddleOCR (pre-trained, 80+ languages)      |
| Fallback     | EasyOCR (optional)                          |
| Target       | 90–95% character accuracy                   |

### Country Identifier

| Attribute    | Details                                     |
|--------------|---------------------------------------------|
| Method       | Rule-based regex + lookup table             |
| Coverage     | 60+ countries                               |

---

## Project Structure

```
CoreVision/
├── model/                          # Core ML modules
│   ├── pipeline.py                 # End-to-end orchestrator
│   ├── video_processor.py          # Frame extraction from video
│   ├── detector.py                 # YOLOv8 vehicle + plate detection
│   ├── car_classifier.py           # EfficientNetV2-S classification
│   ├── plate_ocr.py                # PaddleOCR multi-language reader
│   ├── country_identifier.py       # Country code matching
│   ├── backbone.py                 # Shared backbone utilities
│   ├── heads.py                    # Classification heads
│   ├── losses.py                   # Loss functions
│   ├── multitask_net.py            # Multi-task network definition
│   └── inference.py                # Inference utilities
├── backend/
│   └── app.py                      # FastAPI REST server
├── frontend/
│   ├── index.html                  # Web UI
│   ├── script.js                   # Client-side logic
│   └── style.css                   # Styling
├── notebooks/
│   ├── train_classifier.ipynb      # Colab: car classifier training
│   └── train_detector.ipynb        # Colab: plate detector training
├── training/                       # Training utilities & config
├── weights/                        # Trained model weights
├── data/                           # Class name mappings
├── scripts/                        # Setup & helper scripts
└── requirements.txt                # Python dependencies
```

---

## Technology Stack

| Layer           | Technology                                          |
|-----------------|-----------------------------------------------------|
| Deep Learning   | PyTorch ≥2.1, torchvision, timm                    |
| Detection       | Ultralytics YOLOv8                                  |
| OCR             | PaddleOCR (PaddlePaddle backend)                    |
| Video/Image     | OpenCV, Pillow, NumPy                               |
| Backend         | FastAPI, Uvicorn                                    |
| Frontend        | HTML, CSS, JavaScript                               |
| Training        | Google Colab (T4 GPU), mixed-precision (AMP)        |
| Language        | Python 3.11                                         |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models on Google Colab

- **Plate Detector:** Open `notebooks/train_detector.ipynb` in Colab → GPU T4 → run all cells
- **Car Classifier:** Open `notebooks/train_classifier.ipynb` in Colab → GPU T4 → run all cells

Download the trained weights and place them in the `weights/` directory.

### 3. Start the application

```bash
python backend/app.py
```

Access the web interface at **http://localhost:8000**.

---

## Key Design Decisions

1. **Progressive unfreezing** — Training starts with the backbone frozen to quickly warm up the classification head, then gradually unfreezes for fine-tuning. This prevents catastrophic forgetting of ImageNet features.

2. **Year-variant merging** — Instead of treating each model-year as a separate class, year variants are merged (e.g., all Civic years → one "Honda_Civic" class). This gives more images per class and a more practical output.

3. **Colab-safe checkpointing** — The training loop saves checkpoints to Google Drive after every epoch with automatic resume support, preventing progress loss from Colab disconnections.

4. **Multi-engine OCR** — PaddleOCR is the primary engine with EasyOCR as an optional fallback, ensuring broad language and platform compatibility.

5. **Modular pipeline** — Each component (detection, classification, OCR, country ID) is a standalone module that can be tested and improved independently.
