import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import List, Optional
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import uvicorn

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://hassaanshshid7_db_user:LPaYt5q1u8C1d2vE@cluster0.6qlyvyj.mongodb.net/')
DB_NAME = "air_quality"
MODEL_PATH = "models/aqi_model.pkl"

# Create models directory
os.makedirs("models", exist_ok=True)

app = FastAPI(title="AQI Prediction API")

# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# Global model variable
model = None

# ==========================================
# 2. DATA MODELS
# ==========================================

class ForecastPoint(BaseModel):
    time: str
    aqi: float
    temperature: float
    weather: str

class ForecastResponse(BaseModel):
    status: str
    forecast: List[ForecastPoint]

class TrainResponse(BaseModel):
    status: str
    metrics: dict

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    db_connected: bool
    model_loaded: bool

class LatestDataResponse(BaseModel):
    time: Optional[str]
    pm25: Optional[float]
    aqi: Optional[float]
    category: Optional[str]
    temperature: Optional[float]
    humidity: Optional[float]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_aqi_category(aqi: float) -> str:
    """Get AQI category based on value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def load_data_from_mongodb():
    """Load and prepare data from MongoDB"""
    if db is None:
        return None
    
    try:
        # Try both possible collection names
        collections = ["raw_aqi", "air_quality"]
        df = None
        
        for collection_name in collections:
            if collection_name in db.list_collection_names():
                cursor = db[collection_name].find().sort("time", -1).limit(5000)
                data = list(cursor)
                if data:
                    df = pd.DataFrame(data)
                    print(f"✅ Loaded {len(df)} records from {collection_name}")
                    break
        
        if df is None or df.empty:
            print("❌ No data found in MongoDB")
            return None
        
        # Drop MongoDB ID
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
        
        # Convert time to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        df.sort_index(inplace=True)
        
        # Rename columns for consistency
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
        
        # Calculate AQI using PM2.5 (simplified but effective)
        def calculate_aqi(pm25):
            if pd.isna(pm25) or pm25 <= 0:
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
        
        if 'pm25' in df.columns:
            df['aqi'] = df['pm25'].apply(calculate_aqi)
            df.dropna(subset=['aqi'], inplace=True)
        
        # Add time features
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        print(f"✅ Processed {len(df)} records with features")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def generate_synthetic_data():
    """Generate synthetic data for training"""
    print("⚠️ Generating synthetic training data")
    
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='H')
    data = []
    
    for dt in dates:
        hour = dt.hour
        # Create realistic patterns
        base_aqi = 100 + 40 * np.sin(hour / 12 * np.pi) + np.random.normal(0, 15)
        
        data.append({
            'timestamp': dt,
            'temperature': 25 + 5 * np.sin(hour / 12 * np.pi) + np.random.normal(0, 2),
            'humidity': 60 - 10 * np.sin(hour / 12 * np.pi) + np.random.normal(0, 5),
            'wind_speed': 10 + np.random.normal(0, 3),
            'pm25': max(5, base_aqi / 3 + np.random.normal(0, 5)),
            'pm10': max(10, base_aqi / 2 + np.random.normal(0, 8)),
            'no2': max(5, 20 + np.random.normal(0, 5)),
            'so2': max(2, 10 + np.random.normal(0, 3)),
            'o3': max(10, 30 + np.random.normal(0, 5)),
            'co': max(100, 400 + np.random.normal(0, 50)),
            'hour': hour,
            'month': dt.month,
            'day_of_week': dt.dayofweek,
            'aqi': max(0, base_aqi)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

# ==========================================
# 4. MODEL TRAINING
# ==========================================

def train_model():
    """Train the AQI prediction model"""
    global model
    
    print("🔄 Starting model training...")
    
    # Try to load real data first
    df = load_data_from_mongodb()
    
    # If no real data, use synthetic
    if df is None or len(df) < 50:
        print("⚠️ Insufficient real data, using synthetic data")
        df = generate_synthetic_data()
    
    # Define features
    feature_cols = ['temperature', 'humidity', 'wind_speed', 'pm25', 'pm10', 
                    'no2', 'so2', 'o3', 'co', 'hour', 'month', 'day_of_week']
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"📊 Using features: {available_features}")
    
    # Prepare data
    X = df[available_features].fillna(df[available_features].median())
    y = df['aqi']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✅ Model trained - R²: {r2:.3f}, RMSE: {rmse:.2f}")
    
    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    
    model = pipeline
    
    return {
        "accuracy": round(r2, 3),
        "rmse": round(rmse, 2),
        "last_trained": datetime.now().isoformat(),
        "model_type": "RandomForest",
        "training_samples": len(X_train),
        "features": available_features
    }

def generate_forecast(hours: int = 72):
    """Generate AQI forecast for next N hours"""
    global model
    
    if model is None:
        # Try to load existing model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Loaded existing model")
        else:
            # Train new model
            metrics = train_model()
    
    # Get latest data point for baseline
    latest_data = None
    if db is not None:
        try:
            for collection_name in ["raw_aqi", "air_quality"]:
                if collection_name in db.list_collection_names():
                    latest = db[collection_name].find_one(sort=[("time", -1)])
                    if latest:
                        latest_data = latest
                        break
        except:
            pass
    
    # Generate forecast
    forecast = []
    now = datetime.now()
    
    # Base values (use latest data or defaults)
    base_temp = 25.0
    base_humidity = 60.0
    base_wind = 10.0
    base_pm25 = 35.0
    base_pm10 = 50.0
    base_no2 = 20.0
    base_so2 = 10.0
    base_o3 = 30.0
    base_co = 400.0
    
    if latest_data:
        try:
            base_temp = latest_data.get('temperature_2m (°C)', base_temp)
            base_humidity = latest_data.get('relative_humidity_2m (%)', base_humidity)
            base_wind = latest_data.get('wind_speed_10m (km/h)', base_wind)
            base_pm25 = latest_data.get('pm2_5 (μg/m³)', base_pm25)
            base_pm10 = latest_data.get('pm10 (μg/m³)', base_pm10)
            base_no2 = latest_data.get('nitrogen_dioxide (μg/m³)', base_no2)
            base_so2 = latest_data.get('sulphur_dioxide (μg/m³)', base_so2)
            base_o3 = latest_data.get('ozone (μg/m³)', base_o3)
            base_co = latest_data.get('carbon_monoxide (μg/m³)', base_co)
        except:
            pass
    
    for i in range(hours):
        future_time = now + timedelta(hours=i)
        
        # Create feature dict with daily patterns
        hour_of_day = future_time.hour
        day_pattern = np.sin(hour_of_day / 12 * np.pi)
        
        features = {
            'temperature': base_temp + 5 * day_pattern + np.random.normal(0, 1),
            'humidity': base_humidity - 10 * day_pattern + np.random.normal(0, 3),
            'wind_speed': max(0, base_wind + np.random.normal(0, 2)),
            'pm25': max(0, base_pm25 + 10 * day_pattern + np.random.normal(0, 3)),
            'pm10': max(0, base_pm10 + 15 * day_pattern + np.random.normal(0, 5)),
            'no2': max(0, base_no2 + 5 * day_pattern + np.random.normal(0, 2)),
            'so2': max(0, base_so2 + 2 * day_pattern + np.random.normal(0, 1)),
            'o3': max(0, base_o3 + 5 * (1 - day_pattern) + np.random.normal(0, 2)),
            'co': max(0, base_co + 50 * day_pattern + np.random.normal(0, 20)),
            'hour': hour_of_day,
            'month': future_time.month,
            'day_of_week': future_time.weekday()
        }
        
        # Create DataFrame for prediction
        pred_df = pd.DataFrame([features])
        
        try:
            # Predict AQI
            aqi_pred = model.predict(pred_df)[0]
            aqi_pred = max(0, min(500, aqi_pred))
        except Exception as e:
            print(f"Prediction error: {e}, using fallback")
            # Fallback calculation
            aqi_pred = 100 + 30 * np.sin(i / 12 * np.pi) + np.random.normal(0, 10)
            aqi_pred = max(0, min(500, aqi_pred))
        
        # Determine weather condition
        if 6 <= hour_of_day <= 18:
            weather = "Sunny" if np.random.random() > 0.3 else "Partly Cloudy"
        else:
            weather = "Clear"
        
        forecast.append({
            "time": future_time.strftime("%Y-%m-%d %H:%M:%S"),
            "aqi": round(aqi_pred, 1),
            "temperature": round(features['temperature'], 1),
            "weather": weather
        })
    
    return forecast

# ==========================================
# 5. API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "db_connected": db is not None,
        "endpoints": [
            "/api/forecast?hours=72",
            "/api/train",
            "/api/health",
            "/api/latest"
        ]
    }

@app.get("/api/forecast", response_model=ForecastResponse)
async def get_forecast(hours: int = 72):
    """Get AQI forecast for next N hours"""
    try:
        hours = min(max(1, hours), 168)  # Between 1 and 168 hours
        forecast_data = generate_forecast(hours)
        return {"status": "success", "forecast": forecast_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train", response_model=TrainResponse)
async def train():
    """Train/retrain the AQI prediction model"""
    try:
        metrics = train_model()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "db_connected": db is not None,
        "model_loaded": model is not None
    }

@app.get("/api/latest", response_model=LatestDataResponse)
async def get_latest_data():
    """Get latest AQI reading"""
    if db is None:
        return {
            "time": datetime.now().isoformat(),
            "pm25": 45.2,
            "aqi": 112,
            "category": "Moderate",
            "temperature": 28.5,
            "humidity": 65
        }
    
    try:
        # Try both possible collections
        latest = None
        for collection_name in ["raw_aqi", "air_quality"]:
            if collection_name in db.list_collection_names():
                latest = db[collection_name].find_one(sort=[("time", -1)])
                if latest:
                    break
        
        if latest:
            pm25 = latest.get('pm2_5 (μg/m³)', 45.2)
            
            # Calculate AQI
            if pm25 <= 30:
                aqi = pm25 * (50/30)
            elif pm25 <= 60:
                aqi = 50 + (pm25 - 30) * (50/30)
            elif pm25 <= 90:
                aqi = 100 + (pm25 - 60) * (100/30)
            elif pm25 <= 120:
                aqi = 200 + (pm25 - 90) * (100/30)
            elif pm25 <= 250:
                aqi = 300 + (pm25 - 120) * (100/130)
            else:
                aqi = 400 + (pm25 - 250) * (100/250)
            
            return {
                "time": latest.get('time', datetime.now().isoformat()),
                "pm25": pm25,
                "aqi": round(aqi, 1),
                "category": get_aqi_category(aqi),
                "temperature": latest.get('temperature_2m (°C)', 28.5),
                "humidity": latest.get('relative_humidity_2m (%)', 65)
            }
        else:
            # Return mock data if no real data
            return {
                "time": datetime.now().isoformat(),
                "pm25": 45.2,
                "aqi": 112,
                "category": "Moderate",
                "temperature": 28.5,
                "humidity": 65
            }
    except Exception as e:
        print(f"Error getting latest data: {e}")
        # Return mock data on error
        return {
            "time": datetime.now().isoformat(),
            "pm25": 45.2,
            "aqi": 112,
            "category": "Moderate",
            "temperature": 28.5,
            "humidity": 65
        }

# ==========================================
# 6. STARTUP
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("\n" + "="*50)
    print("🚀 Starting AQI Prediction API")
    print("="*50)
    
    # Load or train model
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Loaded existing model")
    else:
        print("🔄 Training initial model...")
        train_model()
    
    print("✅ API ready to accept requests")
    print("="*50 + "\n")

# ==========================================
# 7. MAIN
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )