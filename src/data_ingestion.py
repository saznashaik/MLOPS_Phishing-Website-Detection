import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()   # This reads the .env file

MONGO_URI = os.getenv("MONGO_URI")

def ingest_data_to_mongodb():
    """Step 1: Read CSV and put data into MongoDB"""
    client = MongoClient(MONGO_URI)
    db = client["saznashaik_db_user"]           # Database name
    collection = db["raw_data"]          # Collection (like table)
    
    # Read the CSV file
    df = pd.read_csv("data/raw/dataset_phishing.csv")
    
    # Convert to dictionary and insert into MongoDB
    records = df.to_dict("records")
    collection.delete_many({})           # Clear old data (only for practice)
    collection.insert_many(records)
    
    print(f"✅ Successfully inserted {len(records)} rows into MongoDB!")
    return df

def load_data_from_mongodb():
    """Load data from MongoDB when needed"""
    client = MongoClient(MONGO_URI)
    db = client["phishing_db"]
    collection = db["raw_data"]
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    return df

# Run this file once
if __name__ == "__main__":
    ingest_data_to_mongodb()