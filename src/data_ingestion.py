import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def ingest_data_to_mongodb():
    """Ingest CSV data into MongoDB"""
    if not MONGO_URI:
        print("❌ MONGO_URI not found in .env file!")
        return None
    
    client = MongoClient(MONGO_URI)
    db = client["phishing_db"]
    collection = db["raw_data"]
    
    csv_path = "data/raw/dataset_phishing.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        print("Please make sure the file is in data/raw/dataset_phishing.csv")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} rows from CSV file")
    
    # Clear previous data and insert new
    collection.delete_many({})
    collection.insert_many(df.to_dict("records"))
    
    print(f"✅ Successfully ingested {len(df)} rows into MongoDB collection 'raw_data'")
    return df


def load_data_from_mongodb():
    """Load data from MongoDB"""
    client = MongoClient(MONGO_URI)
    db = client["phishing_db"]
    collection = db["raw_data"]
    
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    
    return df


# Run ingestion when this file is executed directly
if __name__ == "__main__":
    print("Starting data ingestion...")
    ingest_data_to_mongodb()