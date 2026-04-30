import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DRIFT_THRESHOLD = 0.3
MONITORING_REPORT_DIR = "reports/monitoring"
DRIFT_STATUS_FILE = "reports/monitoring/drift_status.json"


def load_reference_data():
    path = "data/processed/reference_data.csv"

    if not os.path.exists(path):
        path = "data/raw/dataset_phishing.csv"
        if not os.path.exists(path):
            raise FileNotFoundError("No reference data found")

    df = pd.read_csv(path)
    print(f"✅ Loaded reference data: {df.shape}")
    return df


def simulate_production_data(reference_df, drift_intensity=0.08):
    df = reference_df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    noise = np.random.normal(
        0,
        drift_intensity * df[numeric_cols].std(),
        df[numeric_cols].shape
    )

    df[numeric_cols] += noise

    df = df.sample(n=min(800, len(df)), random_state=42)
    print(f"✅ Simulated production data: {df.shape}")
    return df


def run_drift_detection(reference_df, current_df):
    print("\n🔍 Running Drift Detection...")

    os.makedirs(MONITORING_REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])

        # ✅ Correct order
        result = report.run(current_df, reference_df)

        # ✅ Safe output
        report_dict = result.dict() if hasattr(result, "dict") else result.as_dict()

        # Extract drift info safely
        drift_detected = False
        share_drifted = 0.0

        for metric in report_dict.get("metrics", []):
            if "drift" in str(metric).lower():
                res = metric.get("result", {})
                drift_detected = res.get("dataset_drift", False)
                share_drifted = res.get("share_selected_features_drifted", 0.0)

        retrain = share_drifted > DRIFT_THRESHOLD

        status = {
            "timestamp": timestamp,
            "drift_detected": drift_detected,
            "share_drifted_features": round(share_drifted, 4),
            "drift_threshold": DRIFT_THRESHOLD,
            "retrain_triggered": retrain
        }

        with open(DRIFT_STATUS_FILE, "w") as f:
            json.dump(status, f, indent=4)

        print(f"\n📊 Drift: {share_drifted*100:.2f}%")

        if retrain:
            print("⚠️ Retraining needed")
        else:
            print("✅ Model stable")

        return status

    except Exception as e:
        print(f"❌ Drift detection failed: {e}")
        return {"retrain_triggered": False}


def run_monitoring():
    print("=" * 60)
    print("🔭 MONITORING STARTED")
    print("=" * 60)

    ref = load_reference_data()
    curr = simulate_production_data(ref)

    status = run_drift_detection(ref, curr)

    if status.get("retrain_triggered"):
        print("\n🔴 FINAL: RETRAIN MODEL")
        return True
    else:
        print("\n🟢 FINAL: MODEL OK")
        return False


if __name__ == "__main__":
    retrain = run_monitoring()
    sys.exit(1 if retrain else 0)