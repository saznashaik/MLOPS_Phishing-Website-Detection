import pandas as pd
import os
import json
from datetime import datetime
import sys
# Add the current project root to Python path so it can find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your previous file
from src.data_ingestion import load_data_from_mongodb

# ========================= CONFIGURATION =========================
# TODO: Change this to your actual target column name
TARGET_COLUMN = "status"          # Most common names: "status", "label", "phishing", "target"

def run_validation():
    print("🔍 Starting Simple Data Validation...\n")
    
    # Load data from MongoDB
    df = load_data_from_mongodb()
    
    print(f"✅ Loaded {len(df)} rows and {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(df),
        "columns_count": df.shape[1],
        "status": "PASSED",
        "issues": []
    }
    
    # === 1. Check Missing Values ===
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        report["issues"].append(f"Found {missing_values} missing values")
        print(f"⚠️  Missing values: {missing_values}")
    
    # === 2. Check Target Column ===
    if TARGET_COLUMN not in df.columns:
        report["issues"].append(f"Target column '{TARGET_COLUMN}' not found in dataset!")
        report["status"] = "FAILED"
        print(f"❌ Target column '{TARGET_COLUMN}' not found!")
    else:
        print(f"✅ Target column '{TARGET_COLUMN}' found")
        print("Target Distribution:")
        print(df[TARGET_COLUMN].value_counts())
        print()
    
    # === 3. Check Duplicate Rows ===
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        report["issues"].append(f"Found {duplicates} duplicate rows")
        print(f"⚠️  Duplicate rows: {duplicates}")
    
    # === 4. Basic Shape Check ===
    if df.shape[0] < 100:
        report["issues"].append("Dataset has very few rows (<100)")
    
    # Save validation report
    os.makedirs("reports/validation", exist_ok=True)
    report_path = f"reports/validation/validation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    # Final Result
    print("="*60)
    if report["status"] == "PASSED" and len(report["issues"]) == 0:
        print("🎉 DATA VALIDATION PASSED SUCCESSFULLY!")
    elif report["status"] == "PASSED":
        print("✅ DATA VALIDATION PASSED WITH WARNINGS")
    else:
        print("❌ DATA VALIDATION FAILED")
    print("="*60)
    
    if report["issues"]:
        print("Issues found:")
        for issue in report["issues"]:
            print(f"   • {issue}")
    
    print(f"\nReport saved: {report_path}")
    return report["status"] == "PASSED"


if __name__ == "__main__":
    success = run_validation()
    if not success:
        print("\nPlease fix the issues above before continuing.")