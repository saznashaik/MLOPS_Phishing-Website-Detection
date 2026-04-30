"""
Auto-retraining script.
Called automatically when monitoring detects drift.
Re-runs the full pipeline and saves a new best model.
"""
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_retrain():
    print("=" * 60)
    print("🔄 AUTO-RETRAINING TRIGGERED")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load drift status to log why we are retraining
    drift_status_path = "reports/monitoring/drift_status.json"
    if os.path.exists(drift_status_path):
        with open(drift_status_path) as f:
            drift_status = json.load(f)
        print(f"\nRetrain reason:")
        print(f"  Drift detected: {drift_status.get('drift_detected')}")
        print(f"  Share drifted features: {drift_status.get('share_drifted_features', 0)*100:.1f}%")
        print(f"  Final decision: {drift_status.get('final_decision')}")

    # Step 1: Re-ingest fresh data from MongoDB
    print("\n[1/3] Re-ingesting data from MongoDB...")
    from src.data_ingestion import ingest_data_to_mongodb
    ingest_data_to_mongodb()

    # Step 2: Re-run preprocessing
    print("\n[2/3] Re-running preprocessing...")
    from src.data_ingestion import load_data_from_mongodb
    from src.data_preprocess import preprocess_data
    df = load_data_from_mongodb()
    preprocess_data(df)

    # Step 3: Re-train all models
    print("\n[3/3] Re-training models...")
    from src.data_train import run_training
    best = run_training()

    # Save retrain history log
    os.makedirs("reports/monitoring", exist_ok=True)
    history_path = "reports/monitoring/retrain_history.json"

    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trigger": "drift_detected",
        "best_model": best.get("model_name"),
        "f1_score": best.get("f1_score"),
    })

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\n✅ Auto-retraining complete!")
    print(f"   New best model: {best.get('model_name')}")
    print(f"   New F1 score:   {best.get('f1_score')}")
    print(f"   Retrain history saved: {history_path}")


if __name__ == "__main__":
    run_retrain()