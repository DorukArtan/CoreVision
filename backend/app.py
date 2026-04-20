"""
app.py - FastAPI Backend Server (CoreVision v2)

Endpoints:
    GET  /               → serves the frontend
    POST /predict        → image upload inference
    POST /predict-video  → video upload inference
    GET  /health         → health check
"""

import os
import sys
import io
import base64
import time
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.pipeline import VehicleRecognitionPipeline

# ---- App Configuration ----
app = FastAPI(
    title="CoreVision",
    description="Video-based vehicle model & license plate recognition",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---- Global Pipeline ----
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = VehicleRecognitionPipeline(
            car_weights=os.path.join(PROJECT_ROOT, 'weights', 'car_classifier.pth'),
            plate_det_weights=os.path.join(PROJECT_ROOT, 'weights', 'plate_detector.pt'),
            car_class_names=os.path.join(PROJECT_ROOT, 'data', 'compcars_classes.txt'),
        )
    return pipeline


def _encode_image(pil_image):
    """Encode a PIL image as base64 JPEG string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- Routes ----

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in /frontend/</h1>")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Run full pipeline on a single uploaded image.
    Accepts image files (jpg, png, etc.)
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        model = get_pipeline()
        result = model.process_image(image, return_annotated=True)

        # Encode annotated image
        annotated_b64 = None
        if 'annotated_image' in result and result['annotated_image']:
            annotated_b64 = _encode_image(result['annotated_image'])

        # Clean result for JSON (remove PIL Image objects)
        vehicles = []
        for v in result.get('vehicles', []):
            vehicle_clean = {k: val for k, val in v.items() if k != 'crop'}
            plates_clean = []
            for p in vehicle_clean.get('plates', []):
                plates_clean.append({k: val for k, val in p.items() if k != 'crop'})
            vehicle_clean['plates'] = plates_clean
            vehicles.append(vehicle_clean)

        # Clean standalone plates too
        standalone_plates = []
        for p in result.get('standalone_plates', []):
            standalone_plates.append({k: val for k, val in p.items() if k != 'crop'})

        return JSONResponse(content={
            "success": True,
            "vehicles": vehicles,
            "standalone_plates": standalone_plates,
            "total_vehicles": result.get('total_vehicles', 0),
            "total_plates": result.get('total_plates', 0),
            "annotated_image": annotated_b64,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """
    Run full pipeline on an uploaded video.
    Accepts video files (mp4, avi, mov, etc.)
    Extracts frames, runs detection per-frame, returns aggregated summary.
    """
    allowed = ('video/', 'application/octet-stream')
    if file.content_type and not any(file.content_type.startswith(a) for a in allowed):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        contents = await file.read()

        # Write to temp file for OpenCV
        import tempfile
        suffix = os.path.splitext(file.filename or 'video.mp4')[-1] or '.mp4'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            model = get_pipeline()
            result = model.process_video(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Build clean response
        summary = result.get('summary', {})

        return JSONResponse(content={
            "success": True,
            "video_info": result.get('video_info', {}),
            "summary": {
                "unique_vehicles": summary.get('unique_vehicles', []),
                "unique_plates": summary.get('unique_plates', []),
                "total_unique_vehicles": summary.get('total_unique_vehicles', 0),
                "total_unique_plates": summary.get('total_unique_plates', 0),
            },
            "frames_analyzed": len(result.get('frame_results', [])),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "pipeline_loaded": pipeline is not None,
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting CoreVision API v2...")
    print(f"Frontend: {FRONTEND_DIR}")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
