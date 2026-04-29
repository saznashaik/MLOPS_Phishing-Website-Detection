from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import json
import os

# Create FastAPI app
app = FastAPI(
    title="Phishing Website Detection API",
    description="MLOps project - detects if a website URL is phishing or legitimate",
    version="1.0.0"
)

# Allow all origins (needed if you test from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model once when server starts ----
MODEL_PATH = "models/best_model.pkl"
model = None
feature_names = None

def load_artifacts():
    global model, feature_names
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        X_train = pd.read_csv("data/processed/X_train.csv")
        feature_names = list(X_train.columns)
        print(f"✅ Model loaded. Expects {len(feature_names)} features.")
    else:
        print("⚠️  Model file not found. Run training first.")

load_artifacts()


# ---- Define request body ----
class PredictionRequest(BaseModel):
    features: dict   # Dictionary of feature_name: value


# ---- Health check endpoint ----
@app.get("/")
def home():
    return {
        "message": "Phishing Detection API is running!",
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ---- Health check ----
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}


# ---- Model info ----
@app.get("/model-info")
def model_info():
    if not os.path.exists("config/best_model.json"):
        return {"error": "No model info found"}
    with open("config/best_model.json") as f:
        info = json.load(f)
    return info


# ---- Prediction endpoint ----
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        # Build DataFrame from request
        df = pd.DataFrame([request.features])

        # Fill missing columns with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        return {
            "prediction": int(prediction),
            "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
            "confidence_percent": round(float(max(probability)) * 100, 2),
            "phishing_probability": round(float(probability[1]) * 100, 2),
            "safe_probability": round(float(probability[0]) * 100, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))