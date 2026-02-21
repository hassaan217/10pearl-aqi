import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Configuration
MONGO_URI = os.environ.get('MONGO_URI')

def calculate_aqi_from_pm25(pm25):
    if pd.isna(pm25): return 50
    if pm25 <= 12.0: return (pm25 / 12.0) * 50
    elif pm25 <= 35.4: return 51 + ((pm25 - 12.1) / 23.3) * 49
    elif pm25 <= 55.4: return 101 + ((pm25 - 35.5) / 19.9) * 49
    elif pm25 <= 150.4: return 151 + ((pm25 - 55.5) / 94.9) * 49
    elif pm25 <= 250.4: return 201 + ((pm25 - 150.5) / 99.9) * 99
    else: return 401

def main():
    if not MONGO_URI:
        print("❌ Missing MONGO_URI")
        sys.exit(1)

    print("🔄 Training Pipeline: Loading Data from MongoDB...")
    
    try:
        client = MongoClient(MONGO_URI)
        db = client['air_quality']
        
        # Load all data from MongoDB
        cursor = db['raw_aqi'].find().sort("time", -1).limit(10000)
        df = pd.DataFrame(list(cursor))
        
        if '_id' in df.columns: df.drop('_id', axis=1, inplace=True)
        
        if len(df) == 0:
            print("⚠️ No data in MongoDB. Cannot train.")
            sys.exit(1)

        print(f"✅ Loaded {len(df)} records")

        # Preprocessing (Matches main.py logic)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Rename columns
        rename_map = {
            'pm2_5 (µg/m³)': 'pm25', 'pm10 (µg/m³)': 'pm10',
            'carbon_monoxide (µg/m³)': 'co', 'carbon_dioxide (ppm)': 'co2',
            'nitrogen_dioxide (µg/m³)': 'no2', 'sulphur_dioxide (µg/m³)': 'so2',
            'ozone (µg/m³)': 'o3', 'dust (µg/m³)': 'dust',
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity',
            'wind_speed_10m (km/h)': 'wind_speed'
        }
        df.rename(columns=rename_map, inplace=True)

        # Calculate AQI
        df['aqi'] = df['pm25'].apply(calculate_aqi_from_pm25)
        df.dropna(subset=['aqi'], inplace=True)

        # Feature Engineering
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek

        # Define Features
        all_features = ['temperature', 'humidity', 'wind_speed', 'pm25', 'pm10', 
                        'no2', 'so2', 'o3', 'co', 'hour', 'month', 'day_of_week']
        available_features = [col for col in all_features if col in df.columns]
        
        X = df[available_features].fillna(df[available_features].median())
        y = df['aqi']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
        }

        results = {}
        best_model = None
        best_score = -np.inf
        best_name = ''

        print("🤖 Training models...")
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
            print(f"   {name}: R²={r2:.4f}, RMSE={rmse:.2f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        # Save Model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')

        # Save Metadata
        metadata = {
            'best_model': best_name,
            'r2_score': float(best_score),
            'metrics': {k: {'r2': float(v['r2']), 'rmse': float(v['rmse']), 'mae': float(v['mae'])} for k, v in results.items()},
            'timestamp': datetime.now().isoformat()
        }
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Best Model ({best_name}) saved.")

        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()