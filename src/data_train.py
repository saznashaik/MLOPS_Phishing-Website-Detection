import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import json
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocess import load_preprocessed_data

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ===================== CONFIGURATION =====================
MLFLOW_EXPERIMENT_NAME = "phishing-detection"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def evaluate_model(model, X_test, y_test):
    """Calculate all evaluation metrics"""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
    }
    return metrics, y_pred


def train_and_log_model(model, model_name, params, X_train, X_test, y_train, y_test):
    """
    Train one model, log everything to MLflow, save the model file.
    """
    print(f"\n🚀 Training: {model_name}")

    with mlflow.start_run(run_name=model_name):

        # ---- Train the model ----
        model.fit(X_train, y_train)

        # ---- Evaluate the model ----
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1 Score:  {metrics['f1_score']}")

        # ---- Log parameters to MLflow ----
        # Parameters = the settings we used (like how many trees, max depth)
        mlflow.log_params(params)

        # ---- Log metrics to MLflow ----
        mlflow.log_metrics(metrics)

        # ---- Log the trained model to MLflow ----
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ---- Save model locally as well ----
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # ---- Save classification report ----
        report = classification_report(y_test, y_pred)
        os.makedirs(f"{REPORTS_DIR}/model_eval", exist_ok=True)
        report_path = f"{REPORTS_DIR}/model_eval/{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Trained at: {datetime.now()}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        # Get the run ID (useful for loading model later)
        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow Run ID: {run_id}")

    return metrics, model, run_id


def save_best_model_info(results: list):
    """Find the best model and save its info"""
    # Sort by F1 score (best balance of precision and recall)
    best = max(results, key=lambda x: x["f1_score"])

    print(f"\n🏆 Best Model: {best['model_name']}")
    print(f"   F1 Score: {best['f1_score']}")

    # Save best model info to a JSON file
    os.makedirs("config", exist_ok=True)
    with open("config/best_model.json", "w") as f:
        json.dump(best, f, indent=4)

    # Copy best model to a standard name so FastAPI can easily load it
    best_model_path = f"{MODELS_DIR}/{best['model_name'].lower().replace(' ', '_')}.pkl"
    final_model_path = f"{MODELS_DIR}/best_model.pkl"
    
    import shutil
    shutil.copy(best_model_path, final_model_path)
    print(f"✅ Best model saved as: {final_model_path}")

    return best


def run_training():
    """Main training function — trains multiple models and picks the best"""

    # ---- Load preprocessed data ----
    print("Loading preprocessed data...")
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"Training with {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # ---- Setup MLflow ----
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")

    # ---- Define models to try ----
    # We will train 3 different models and pick the best one
    models_to_train = [
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 10, "model_type": "RandomForest"}
        },
        {
            "name": "Gradient Boosting",
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "model_type": "GradientBoosting"}
        },
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"max_iter": 1000, "model_type": "LogisticRegression"}
        },
    ]

    # ---- Train all models ----
    results = []
    for item in models_to_train:
        metrics, model, run_id = train_and_log_model(
            model=item["model"],
            model_name=item["name"],
            params=item["params"],
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        results.append({
            "model_name": item["name"],
            "run_id":     run_id,
            **metrics
        })

    # ---- Print comparison table ----
    print("\n📊 Model Comparison:")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['model_name']:<25} {r['accuracy']:>10} {r['precision']:>10} {r['recall']:>10} {r['f1_score']:>10}")

    # ---- Save best model ----
    best = save_best_model_info(results)

    print("\n🎉 Training Complete!")
    print("Run 'mlflow ui' in your terminal to see the experiment dashboard.")
    return best


if __name__ == "__main__":
    run_training()