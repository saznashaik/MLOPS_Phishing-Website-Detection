import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import load_data_from_mongodb
# ===================== CONFIGURATION =====================
TARGET_COLUMN = "status"   # <<< CHANGE THIS if your column is different
# If your target column uses text like "phishing"/"legitimate", this code handles it.
# If it already uses 1/0, that is also fine.

def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare data for training.
    Returns: X_train, X_test, y_train, y_test
    """
    print("🔧 Starting Data Preprocessing...")
    print(f"Original shape: {df.shape}")

    # ---- Step 1: Drop columns we don't need ----
    # 'url' column is just the raw URL string — not useful as a feature
    # '_id' is MongoDB's internal ID
    columns_to_drop = ['url', '_id']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  Dropped column: {col}")

    # ---- Step 2: Handle the target column ----
    # Some datasets use "phishing"/"legitimate" as text
    # We need to convert to 1 (phishing) and 0 (legitimate)
    if df[TARGET_COLUMN].dtype == object:
        print(f"  Target column '{TARGET_COLUMN}' has text values. Converting...")
        le = LabelEncoder()
        df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN])
        print(f"  Classes found: {list(le.classes_)} → mapped to {list(range(len(le.classes_)))}")

    # ---- Step 3: Handle missing values ----
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values. Filling with column median...")
        df = df.fillna(df.median(numeric_only=True))

    # ---- Step 4: Handle non-numeric columns ----
    # Some columns might still be text. We encode them.
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if TARGET_COLUMN in non_numeric:
        non_numeric.remove(TARGET_COLUMN)
    
    for col in non_numeric:
        print(f"  Encoding text column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ---- Step 5: Separate features (X) and target (y) ----
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution:\n{y.value_counts()}")

    # ---- Step 6: Split into train and test sets ----
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # stratify=y means both train and test will have same ratio of 0s and 1s

    print(f"\n  Training set: {X_train.shape[0]} rows")
    print(f"  Testing set:  {X_test.shape[0]} rows")

    # ---- Step 7: Save preprocessed data ----
    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv",  index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv",  index=False)

    print("\n✅ Preprocessing complete! Files saved in data/processed/")
    return X_train, X_test, y_train, y_test


def load_preprocessed_data():
    """Load already-preprocessed data from disk"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # # Load data from MongoDB and preprocess it
    # from src.data_ingestion import load_data_from_mongodb
    df = load_data_from_mongodb()
    preprocess_data(df)