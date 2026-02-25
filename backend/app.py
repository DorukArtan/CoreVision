"""
app.py - FastAPI Backend Server

Serves the web interface and handles image upload + inference.
Endpoints:
    GET  /          → serves the frontend
    POST /predict   → runs model inference on uploaded image
"""

import os
import sys
import io
import base64
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.inference import VehicleInferencePipeline

# ---- App Configuration ----
app = FastAPI(
    title="Vehicle Recognition AI",
    description="Multi-task AI for vehicle model classification and license plate recognition",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---- Global Model Instance ----
pipeline = None


def get_pipeline():
    """Lazy-load the inference pipeline."""
    global pipeline
    if pipeline is None:
        # Check if trained weights exist
        weights_path = os.path.join(PROJECT_ROOT, 'weights', 'final_model.pth')
        if os.path.exists(weights_path):
            model_path = weights_path
            print(f"Loading trained model from: {weights_path}")
        else:
            model_path = None
            print("No trained weights found — using model with random weights (demo mode)")
        
        # Sample class names (would be loaded from the dataset in production)
        class_names = _load_class_names()
        
        pipeline = VehicleInferencePipeline(
            model_path=model_path,
            num_classes=196,
            class_names=class_names
        )
    return pipeline


def _load_class_names():
    """Load vehicle class names from file or use defaults."""
    class_file = os.path.join(PROJECT_ROOT, 'data', 'class_names.txt')
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    # Default Stanford Cars class names (representative subset)
    return [
        "AM General Hummer SUV 2000", "Acura RL Sedan 2012", "Acura TL Sedan 2012",
        "Acura TL Type-S 2008", "Acura TSX Sedan 2012", "Acura Integra Type R 2001",
        "Acura ZDX Hatchback 2012", "Aston Martin V8 Vantage Convertible 2012",
        "Aston Martin V8 Vantage Coupe 2012", "Aston Martin Virage Convertible 2012",
        "Aston Martin Virage Coupe 2012", "Audi R8 Coupe 2012", "Audi A5 Coupe 2012",
        "Audi TTS Coupe 2012", "Audi RS 4 Convertible 2008", "Audi S6 Sedan 2011",
        "Audi S5 Convertible 2012", "Audi S5 Coupe 2012", "Audi S4 Sedan 2012",
        "Audi S4 Sedan 2007", "Audi TT Hatchback 2011", "Audi TT RS Coupe 2012",
        "Audi V8 Sedan 1994", "Audi 100 Sedan 1994", "Audi 100 Wagon 1994",
        "BMW 1 Series Convertible 2012", "BMW 1 Series Coupe 2012",
        "BMW 3 Series Sedan 2012", "BMW 3 Series Wagon 2012",
        "BMW 6 Series Convertible 2007", "BMW ActiveHybrid 5 Sedan 2012",
        "BMW M3 Coupe 2012", "BMW M5 Sedan 2010", "BMW M6 Convertible 2010",
        "BMW X3 SUV 2012", "BMW X5 SUV 2007", "BMW X6 SUV 2012",
        "BMW Z4 Convertible 2012", "Bentley Continental Flying Spur Sedan 2007",
        "Bentley Continental GT Coupe 2007", "Bentley Continental GT Coupe 2012",
        "Bentley Continental Supersports Conv. Convertible 2012",
        "Bentley Mulsanne Sedan 2011", "Bugatti Veyron 16.4 Convertible 2009",
        "Bugatti Veyron 16.4 Coupe 2009", "Buick Enclave SUV 2012",
        "Buick Rainier SUV 2007", "Buick Regal GS 2012", "Buick Verano Sedan 2012",
        "Cadillac CTS-V Sedan 2012", "Cadillac Escalade EXT Crew Cab 2007",
        "Cadillac SRX SUV 2012", "Chevrolet Avalanche Crew Cab 2012",
        "Chevrolet Camaro Convertible 2012", "Chevrolet Cobalt SS 2010",
        "Chevrolet Corvette Convertible 2012", "Chevrolet Corvette Ron Fellows Edition Z06 2007",
        "Chevrolet Corvette ZR1 2012", "Chevrolet Express Cargo Van 2007",
        "Chevrolet Express Van 2007", "Chevrolet HHR SS 2010",
        "Chevrolet Impala Sedan 2007", "Chevrolet Malibu Hybrid Sedan 2010",
        "Chevrolet Malibu Sedan 2007", "Chevrolet Monte Carlo Coupe 2007",
        "Chevrolet Silverado 1500 Classic Extended Cab 2007",
        "Chevrolet Silverado 1500 Extended Cab 2012",
        "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
        "Chevrolet Silverado 1500 Regular Cab 2012",
        "Chevrolet Silverado 2500HD Regular Cab 2012",
        "Chevrolet Sonic Sedan 2012", "Chevrolet Tahoe Hybrid SUV 2012",
        "Chevrolet TrailBlazer SS 2009", "Chevrolet Traverse SUV 2012",
        "Chrysler 300 SRT-8 2010", "Chrysler Aspen SUV 2009",
        "Chrysler Crossfire Convertible 2008", "Chrysler PT Cruiser Convertible 2008",
        "Chrysler Sebring Convertible 2010", "Chrysler Town and Country Minivan 2012",
        "Daewoo Nubira Wagon 2002", "Dodge Caliber Wagon 2007",
        "Dodge Caliber Wagon 2012", "Dodge Caravan Minivan 1997",
        "Dodge Challenger SRT8 2011", "Dodge Charger SRT-8 2009",
        "Dodge Charger Sedan 2012", "Dodge Dakota Club Cab 2007",
        "Dodge Dakota Crew Cab 2010", "Dodge Durango SUV 2007",
        "Dodge Durango SUV 2012", "Dodge Journey SUV 2012",
        "Dodge Magnum Wagon 2008", "Dodge Ram Pickup 3500 Crew Cab 2010",
        "Dodge Ram Pickup 3500 Quad Cab 2009", "Dodge Sprinter Cargo Van 2009",
        "Eagle Talon Hatchback 1998", "FIAT 500 Abarth 2012",
        "FIAT 500 Convertible 2012", "Ferrari 458 Italia Convertible 2012",
        "Ferrari 458 Italia Coupe 2012", "Ferrari California Convertible 2012",
        "Ferrari FF Coupe 2012", "Fisker Karma Sedan 2012",
        "Ford E-Series Wagon Van 2012", "Ford Edge SUV 2012",
        "Ford Expedition EL SUV 2009", "Ford F-150 Regular Cab 2007",
        "Ford F-150 Regular Cab 2012", "Ford F-450 Super Duty Crew Cab 2012",
        "Ford Fiesta Sedan 2012", "Ford Focus Sedan 2007",
        "Ford Freestar Minivan 2007", "Ford GT Coupe 2006",
        "Ford Mustang Convertible 2007", "Ford Ranger SuperCab 2011",
        "GMC Acadia SUV 2012", "GMC Canyon Extended Cab 2012",
        "GMC Savana Van 2012", "GMC Terrain SUV 2012",
        "GMC Yukon Hybrid SUV 2012", "Geo Metro Convertible 1993",
        "HUMMER H2 SUT Crew Cab 2009", "HUMMER H3T Crew Cab 2010",
        "Honda Accord Coupe 2012", "Honda Accord Sedan 2012",
        "Honda Odyssey Minivan 2007", "Honda Odyssey Minivan 2012",
        "Hyundai Accent Sedan 2012", "Hyundai Azera Sedan 2012",
        "Hyundai Elantra Sedan 2007", "Hyundai Elantra Touring Hatchback 2012",
        "Hyundai Genesis Sedan 2012", "Hyundai Santa Fe SUV 2012",
        "Hyundai Sonata Hybrid Sedan 2012", "Hyundai Sonata Sedan 2012",
        "Hyundai Tucson SUV 2012", "Hyundai Veloster Hatchback 2012",
        "Hyundai Veracruz SUV 2012", "Infiniti G Coupe IPL 2012",
        "Infiniti QX56 SUV 2011", "Isuzu Ascender SUV 2008",
        "Jaguar XK XKR 2012", "Jeep Compass SUV 2012",
        "Jeep Grand Cherokee SUV 2012", "Jeep Liberty SUV 2012",
        "Jeep Patriot SUV 2012", "Jeep Wrangler SUV 2012",
        "Lamborghini Aventador Coupe 2012", "Lamborghini Diablo Coupe 2001",
        "Lamborghini Gallardo LP 570-4 Superleggera 2012",
        "Lamborghini Reventon Coupe 2008", "Land Rover LR2 SUV 2012",
        "Land Rover Range Rover SUV 2012", "Lincoln Town Car Sedan 2011",
        "MINI Cooper Roadster Convertible 2012", "Maybach Landaulet Convertible 2012",
        "Mazda Tribute SUV 2011", "McLaren MP4-12C Coupe 2012",
        "Mercedes-Benz 300-Class Convertible 1993", "Mercedes-Benz C-Class Sedan 2012",
        "Mercedes-Benz E-Class Sedan 2012", "Mercedes-Benz S-Class Sedan 2012",
        "Mercedes-Benz SL-Class Coupe 2009", "Mercedes-Benz Sprinter Van 2012",
        "Mitsubishi Lancer Sedan 2012", "Nissan 240SX Coupe 1998",
        "Nissan Juke Hatchback 2012", "Nissan Leaf Hatchback 2012",
        "Nissan NV Passenger Van 2012", "Plymouth Neon Coupe 1999",
        "Porsche Panamera Sedan 2012", "Ram C/V Cargo Van Minivan 2012",
        "Rolls-Royce Ghost Sedan 2012", "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
        "Rolls-Royce Phantom Sedan 2012", "Scion xD Hatchback 2012",
        "Spyker C8 Convertible 2009", "Spyker C8 Coupe 2009",
        "Suzuki Aerio Sedan 2007", "Suzuki Kizashi Sedan 2012",
        "Suzuki SX4 Hatchback 2012", "Suzuki SX4 Sedan 2012",
        "Tesla Model S Sedan 2012", "Toyota 4Runner SUV 2012",
        "Toyota Camry Sedan 2012", "Toyota Corolla Sedan 2012",
        "Toyota Sequoia SUV 2012", "Volkswagen Beetle Hatchback 2012",
        "Volkswagen Golf Hatchback 1991", "Volkswagen Golf Hatchback 2012",
        "Volvo 240 Sedan 1993", "Volvo C30 Hatchback 2012",
        "Volvo XC90 SUV 2007", "smart fortwo Convertible 2012",
    ]


# ---- Routes ----

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in /frontend/</h1>")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run multi-task inference on an uploaded image.
    
    Returns:
        JSON with vehicle_model, plate_bbox, plate_text, annotated_image
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get pipeline
        model = get_pipeline()
        
        # Run inference
        result = model.predict(image, return_annotated=True)
        
        # Encode annotated image to base64
        annotated_b64 = None
        if 'annotated_image' in result:
            buffered = io.BytesIO()
            result['annotated_image'].save(buffered, format="JPEG", quality=90)
            annotated_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Build response
        response = {
            "success": True,
            "vehicle_model": result.get('vehicle_model', 'Unknown'),
            "vehicle_confidence": result.get('vehicle_confidence', 0.0),
            "top5_predictions": result.get('top5', []),
            "plate_bbox": result.get('plate_bbox', None),
            "plate_text": result.get('plate_text', ''),
            "plate_confidence": result.get('plate_confidence', 0.0),
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "model_loaded": pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Vehicle Recognition API...")
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
