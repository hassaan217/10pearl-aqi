"""
Simple script to run the feature pipeline in GitHub Actions
"""
import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import sys

def run_feature_pipeline():
    print("="*50)
    print("🚀 Starting Feature Pipeline")
    print(f"Time: {datetime.now()}")
    print("="*50)
    
    # Get MongoDB URI from environment variable
    MONGO_URI = os.environ.get('MONGO_URI')
    if not MONGO_URI:
        print("❌ Error: MONGO_URI environment variable not set")
        sys.exit(1)
    
    try:
        # Connect to MongoDB
        print("📡 Connecting to MongoDB...")
        client = MongoClient(MONGO_URI)
        db = client['air_quality']
        
        # Load CSV file
        csv_path = "air qaulity.csv"  # Make sure this matches your filename
        print(f"📂 Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert time column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Convert to dictionary
        data = df.to_dict('records')
        
        # Save to MongoDB
        collection = db['raw_aqi']
        collection.delete_many({})  # Clear old data
        collection.insert_many(data)
        
        print(f"✅ Successfully saved {len(data)} records to MongoDB")
        print(f"📊 Data shape: {df.shape}")
        
        # Print sample
        print("\n📋 Sample data:")
        print(df.head(2).to_string())
        
        client.close()
        print("\n✅ Feature pipeline completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_feature_pipeline()