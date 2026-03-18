# CoreVision v2 — Video-Based Vehicle Recognition

Detects **car model/brand** and **license plate text + country** from uploaded videos or images. Entirely self-trained models, no external AI APIs.

## Architecture

```
Video Upload → Frame Extraction (OpenCV)
    ↓
YOLOv8-nano (fine-tuned) → Vehicle Crop + Plate Crop
    ↓                           ↓
EfficientNetV2-S          PaddleOCR
(~1,716 car models)       (80+ languages)
    ↓                           ↓
Car Make/Model            Plate Text → Country Code
```

## Project Structure
```
├── model/
│   ├── video_processor.py    # Frame extraction from video
│   ├── detector.py           # YOLOv8 vehicle + plate detection
│   ├── car_classifier.py     # EfficientNetV2-S car model classification
│   ├── plate_ocr.py          # PaddleOCR multi-language plate reader
│   ├── country_identifier.py # Country code matching (60+ countries)
│   └── pipeline.py           # End-to-end orchestrator
├── notebooks/
│   ├── train_classifier.ipynb  # ← Colab: train car model classifier
│   └── train_detector.ipynb    # ← Colab: train plate detector
├── backend/app.py              # FastAPI server
├── frontend/                   # Web UI
├── weights/                    # Trained model weights (download from Colab)
│   ├── car_classifier.pth
│   └── plate_detector.pt
└── data/
    └── compcars_classes.txt    # Car class names (download from Colab)
```

## Training on Google Colab

### Step 1 — Train the Plate Detector (~1-2 hrs)
1. Open `notebooks/train_detector.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU → T4**
3. Get a [free Roboflow API key](https://app.roboflow.com) and paste it in Cell 3
4. Run all cells
5. Download `plate_detector.pt` → place in `weights/`

### Step 2 — Train the Car Classifier (~3-5 hrs)
1. Open `notebooks/train_classifier.ipynb` in Google Colab
2. Set runtime to **GPU → T4**
3. Upload your CompCars dataset zip to Google Drive → `MyDrive/CoreVision/data/`
4. Run all cells
5. Download `car_classifier.pth` and `compcars_classes.txt` → place in `weights/` and `data/`

## Running the App

```bash
pip install -r requirements.txt
python backend/app.py
```

Open **http://localhost:8000**

## Key Components

| Component | Model | Dataset | Accuracy Target |
|-----------|-------|---------|----------------|
| Plate Detection | YOLOv8-nano | Roboflow (multi-country) | 90%+ mAP |
| Car Classification | EfficientNetV2-S | CompCars (~1,716 models) | 90%+ Top-5 |
| Plate OCR | PaddleOCR | Pretrained (80+ langs) | 90-95% |
| Country ID | Regex + lookup | Rule-based | 85%+ |

## Tech Stack
PyTorch • EfficientNetV2 • YOLOv8 • PaddleOCR • FastAPI • OpenCV
