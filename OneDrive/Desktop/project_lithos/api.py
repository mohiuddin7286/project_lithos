from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
import copy
from PIL import Image
import io
from pathlib import Path
import re

# 1. Initialize API
app = FastAPI(title="Project Lithos API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load LightGBM Risk Model (The Telemetry AI)
print("Loading Lithos Core Model...")
try:
    model = joblib.load('lithos_core_model.pkl')
    EXPECTED_FEATURES = model.feature_name_
    print("✅ Lithos Core Model Loaded!")
except Exception as e:
    print(f"❌ Error loading LightGBM model: {e}")
    model = None

# 3. Load YOLOv8 Vision Model (The Drone AI)
def _find_latest_yolo_weights():
    candidate_roots = [
        Path(__file__).resolve().parent / 'runs' / 'detect',
        Path.home() / 'runs' / 'detect',
    ]
    candidates = []

    for root in candidate_roots:
        if not root.exists():
            continue
        for run_dir in root.glob('train*'):
            weight_path = run_dir / 'weights' / 'best.pt'
            if weight_path.exists():
                candidates.append(weight_path)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


try:
    from ultralytics import YOLO
    weights_path = _find_latest_yolo_weights()
    if weights_path is None:
        raise FileNotFoundError("No trained YOLO weights (best.pt) found.")
    vision_model = YOLO(str(weights_path))
    print(f"✅ YOLOv8 Vision Model Loaded from: {weights_path}")
except Exception as e:
    print(f"⚠️ YOLO Model not found yet. Finish training first! ({e})")
    vision_model = None

class SensorReading(BaseModel):
    telemetry: Dict[str, float] 


def _normalize_feature_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())

# ==========================================
# ENDPOINT 1: TELEMETRY DIAGNOSTICS
# ==========================================
@app.post("/api/predict")
def predict_risk(reading: SensorReading):
    if model is None:
        raise HTTPException(status_code=500, detail="AI Core Model is offline or missing.")

    try:
        incoming_data = reading.telemetry
        final_features = {feat: 0.0 for feat in EXPECTED_FEATURES}
        
        # Smart matcher (alphanumeric-only normalization).
        # This makes keys like `slope_angle` reliably match model names like
        # `Slope_Angle_(°)`.
        for react_key, val in incoming_data.items():
            clean_react = _normalize_feature_name(react_key)
            for model_feat in EXPECTED_FEATURES:
                clean_model = _normalize_feature_name(model_feat)
                if clean_react in clean_model or clean_model in clean_react:
                    final_features[model_feat] = float(val)
                    break
        
        # Calculate Risk
        df = pd.DataFrame([final_features])
        probability = model.predict_proba(df)[0][1]
        risk_score = round(probability * 100, 2)
        
        # Set Status
        if risk_score > 75:
            status = "CRITICAL: Evacuate Pit"
        elif risk_score > 30:
            status = "WARNING: Increase Monitoring"
        else:
            status = "SAFE: Nominal Operations"

        # Prescriptive Recommendation Engine
        recommendation = ""
        if risk_score > 30:
            test_features = copy.deepcopy(final_features)
            safe_found = False
            
            water_feat = next((f for f in EXPECTED_FEATURES if 'pore' in f.lower() or 'water' in f.lower() or 'ru' in f.lower()), None)
            angle_feat = next((f for f in EXPECTED_FEATURES if 'angle' in f.lower() and 'slope' in f.lower()), None)

            if water_feat and test_features[water_feat] > 0:
                test_val = test_features[water_feat]
                while test_val > 0.01:
                    test_val -= 0.05
                    test_features[water_feat] = test_val
                    test_df = pd.DataFrame([test_features])
                    if (model.predict_proba(test_df)[0][1] * 100) <= 30:
                        recommendation = f"💡 To reach a SAFE state, trigger drainage to reduce {water_feat} to {test_val:.2f}."
                        safe_found = True
                        break
            
            if not safe_found and angle_feat:
                if water_feat: test_features[water_feat] = 0.0 
                test_angle = test_features[angle_feat]
                while test_angle > 10:
                    test_angle -= 1.0
                    test_features[angle_feat] = test_angle
                    test_df = pd.DataFrame([test_features])
                    if (model.predict_proba(test_df)[0][1] * 100) <= 30:
                        recommendation = f"💡 To reach a SAFE state, excavate and reduce Slope Angle to {test_angle:.1f}°."
                        break
            
            if not recommendation:
                recommendation = "⚠️ Slope geometry is fundamentally unstable. Major structural redesign required."

        return {
            "risk_score": risk_score,
            "status": status,
            "recommendation": recommendation
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data Error: {str(e)}")

# ==========================================
# ENDPOINT 2: COMPUTER VISION
# ==========================================
@app.post("/api/vision")
async def analyze_drone_image(file: UploadFile = File(...)):
    if vision_model is None:
        raise HTTPException(status_code=503, detail="Vision AI is offline or still training.")
        
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        results = vision_model(img)
        
        hazards = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist() 
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = result.names[cls]
                
                hazards.append({
                    "type": name.upper(),
                    "confidence": round(conf * 100, 1),
                    "box": coords 
                })
                
        return {"status": "success", "hazards_detected": len(hazards), "data": hazards}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image Processing Error: {str(e)}")