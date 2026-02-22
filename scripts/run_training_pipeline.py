#!/usr/bin/env python3
"""
Training Pipeline: Loads historical data from MongoDB, trains ML models,
and saves the best performer for AQI prediction.

✅ Fixed for GitHub Actions: TLS 1.2+, certifi CA bundle, robust timeouts
"""

import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import certifi
from pymongo import MongoClient, ServerSelectionTimeoutError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# --- Environment Configuration ---
def load_env_vars():
    """Load and validate required environment variables."""
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        print("❌ Missing required environment variable: MONGO_URI")
        sys.exit(1)
    
    return {
        'MONGO_URI': mongo_uri.strip(),
        'MODEL_PATH': os.getenv('MODEL_PATH', 'backend/models/best_model.pkl')
    }

def connect_to_mongodb(uri: str, timeout_ms: int = 30000):
    """Connect to MongoDB Atlas with TLS/SSL settings for CI/CD."""
    try:
        client = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
            retryWrites=True
        )
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        return client
    except Exception as e:
        print(f"❌ MongoDB connection error: {type(e).__name__}: {e}")
        raise

def calculate_aqi_from_pm25(pm25) -> float:
    """EPA-standard AQI calculation from PM2.5."""
    if pd.isna(pm25): return 50.0
    if pm25 <= 12.0: return (pm25 / 12.0) * 50
    elif pm25 <= 35.4: return 51 + ((pm25 - 12.1) / 23.3) * 49
    elif pm25 <= 55.4: return 101 + ((pm25 - 35.5) / 19.9) * 49
    elif pm25 <= 150.4: return 151 + ((pm25 - 55.5) / 94.9) * 49
    elif pm25 <= 250.4: return 201 + ((pm25 - 150.5) / 99.9) * 99
    elif pm25 <= 350.4: return 301 + ((pm25 - 250.5) / 99.9) * 99
    else: return min(500.0, 401 + ((pm25 - 350.5) / 149.9) * 99)

def main():
    print("🔄 Training Pipeline Started")
    config = load_env_vars()
    
    try:
        # 1. Connect to MongoDB & Load Data
        print("🔌 Connecting to MongoDB...")
        client = connect_to_mongodb(config['MONGO_URI'])
        db = client['air_quality']
        
        print("📥 Loading historical data from 'raw_aqi' collection...")
        cursor = db['raw_aqi'].find().sort("time", -1).limit(10000)
        df = pd.DataFrame(list(cursor))
        
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
        
        # Handle empty dataset gracefully
        if len(df) == 0:
            print("ℹ️  No training data available yet. Waiting for Feature Pipeline to collect data.")
            print("✅ Exiting with success - pipeline will retry on next schedule.")
            client.close()
            sys.exit(0)
        
        print(f"✅ Loaded {len(df)} records")

        # 2. Preprocessing
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Standardize column names
        rename_map = {
            'pm2_5 (µg/m³)': 'pm25', 'pm10 (µg/m³)': 'pm10',
            'carbon_monoxide (µg/m³)': 'co', 'carbon_dioxide (ppm)': 'co2',
            'nitrogen_dioxide (µg/m³)': 'no2', 'sulphur_dioxide (µg/m³)': 'so2',
            'ozone (µg/m³)': 'o3', 'dust (µg/m³)': 'dust',
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity',
            'wind_speed_10m (km/h)': 'wind_speed'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Calculate target variable
        df['aqi'] = df['pm25'].apply(calculate_aqi_from_pm25)
        df.dropna(subset=['aqi', 'pm25'], inplace=True)
        
        if len(df) < 100:
            print(f"⚠️  Only {len(df)} valid records. Need ≥100 for training.")
            client.close()
            sys.exit(0)

        # Feature engineering
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 20))

        # Select features
        feature_cols = [
            'temperature', 'humidity', 'wind_speed', 'pressure_hPa',
            'pm25', 'pm10', 'co', 'no2', 'so2', 'o3',
            'hour', 'month', 'day_of_week', 'is_rush_hour'
        ]
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].copy()
        y = df['aqi'].copy()
        X = X.fillna(X.median(numeric_only=True))

        # 3. Train/Test Split (time-series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 4. Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. Model Training
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=150, max_depth=7, learning_rate=0.08, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=150, max_depth=7, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1),
            'Ridge': Ridge(alpha=1.0)
        }

        results = {}
        best_model = None
        best_score = -np.inf
        best_name = ''

        print("🤖 Training & evaluating models...")
        for name, model in models.items():
            print(f"   → Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
            print(f"      {name}: R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        # 6. Save Artifacts
        os.makedirs(os.path.dirname(config['MODEL_PATH']), exist_ok=True)
        
        model_filename = config['MODEL_PATH']
        scaler_filename = config['MODEL_PATH'].replace('.pkl', '_scaler.pkl')
        metadata_filename = config['MODEL_PATH'].replace('.pkl', '_metadata.json')
        
        print(f"💾 Saving best model: {best_name}")
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        
        metadata = {
            'best_model': best_name,
            'best_r2_score': float(best_score),
            'all_metrics': {name: {k: float(v) for k, v in m.items()} for name, m in results.items()},
            'training_info': {
                'n_samples': len(df),
                'n_features': len(available_features),
                'feature_names': available_features,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pipeline_version': '1.3.0'
            }
        }
        
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Training Complete!")
        print(f"   🏆 Best Model: {best_name} (R² = {best_score:.4f})")
        
        client.close()
        sys.exit(0)
        
    except ServerSelectionTimeoutError:
        print("❌ MongoDB connection failed - check Atlas Network Access")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training Pipeline Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
