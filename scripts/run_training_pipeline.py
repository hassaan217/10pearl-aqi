"""
Simple script to run the training pipeline in GitHub Actions
"""
import os
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
import sys

def run_training_pipeline():
    print("="*50)
    print("🚀 Starting Training Pipeline")
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
        
        # Load data
        print("📂 Loading data from MongoDB...")
        cursor = db['raw_aqi'].find()
        df = pd.DataFrame(list(cursor))
        
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
        
        print(f"✅ Loaded {len(df)} records")
        
        if len(df) == 0:
            print("❌ No data found in database")
            sys.exit(1)
        
        # Prepare data
        print("\n🔄 Preparing data...")
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Add time features
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # Rename columns
        rename_map = {
            'pm2_5 (μg/m³)': 'pm25',
            'pm10 (μg/m³)': 'pm10',
            'carbon_monoxide (μg/m³)': 'co',
            'nitrogen_dioxide (μg/m³)': 'no2',
            'sulphur_dioxide (μg/m³)': 'so2',
            'ozone (μg/m³)': 'o3',
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity',
            'wind_speed_10m (km/h)': 'wind_speed'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Calculate AQI
        def calculate_aqi(pm25):
            if pd.isna(pm25):
                return np.nan
            if pm25 <= 30:
                return pm25 * (50/30)
            elif pm25 <= 60:
                return 50 + (pm25 - 30) * (50/30)
            elif pm25 <= 90:
                return 100 + (pm25 - 60) * (100/30)
            elif pm25 <= 120:
                return 200 + (pm25 - 90) * (100/30)
            elif pm25 <= 250:
                return 300 + (pm25 - 120) * (100/130)
            else:
                return 400 + (pm25 - 250) * (100/250)
        
        df['aqi'] = df['pm25'].apply(calculate_aqi)
        df.dropna(subset=['aqi'], inplace=True)
        
        # Features
        feature_cols = ['temperature', 'humidity', 'wind_speed', 'pm25', 'pm10', 
                        'no2', 'so2', 'o3', 'co', 'hour', 'month', 'day_of_week']
        available_features = [col for col in feature_cols if col in df.columns]
        
        print(f"📊 Using features: {available_features}")
        
        X = df[available_features].fillna(df[available_features].median())
        y = df['aqi']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        print("\n🤖 Training models...")
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1
            )
        }
        
        results = {}
        best_model = None
        best_score = -np.inf
        best_name = ''
        
        for name, model in models.items():
            print(f"\n📈 Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'r2': round(r2, 4),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2)
            }
            
            print(f"   R²: {r2:.4f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   MAE: {mae:.2f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
        
        # Save best model
        print("\n💾 Saving best model...")
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        joblib.dump(best_model, model_path)
        joblib.dump(best_model, 'models/best_model.pkl')  # Latest version
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Save metadata
        metadata = {
            'best_model': best_name,
            'metrics': results[best_name],
            'all_results': results,
            'timestamp': datetime.now().isoformat(),
            'features': available_features,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*50)
        print(f"✅ Best Model: {best_name}")
        print(f"   R² Score: {results[best_name]['r2']}")
        print(f"   RMSE: {results[best_name]['rmse']}")
        print(f"   MAE: {results[best_name]['mae']}")
        print("="*50)
        
        # Print comparison table
        print("\n📊 Model Comparison:")
        print("-"*50)
        print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
        print("-"*50)
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['r2']:<10} {metrics['rmse']:<10} {metrics['mae']:<10}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_training_pipeline()