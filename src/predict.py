import joblib
import pandas as pd
import numpy as np
import json
import os

def load_model():
    """Load the best trained model"""
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run training first: python src/train.py"
        )
    model = joblib.load(model_path)
    return model


def load_feature_names():
    """Get the feature names the model expects"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    return list(X_train.columns)


def predict_single(features: dict) -> dict:
    """
    Make a prediction for one sample.
    features: dictionary of feature_name -> value
    Returns: prediction result
    """
    model = load_model()
    feature_names = load_feature_names()

    # Create a DataFrame with one row, in the correct column order
    df = pd.DataFrame([features])

    # Ensure all expected columns are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0   # Fill missing features with 0

    # Keep only the columns the model knows about, in the right order
    df = df[feature_names]

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return {
        "prediction": int(prediction),
        "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "confidence": round(float(max(probability)) * 100, 2),
        "phishing_probability": round(float(probability[1]) * 100, 2),
        "safe_probability": round(float(probability[0]) * 100, 2),
    }