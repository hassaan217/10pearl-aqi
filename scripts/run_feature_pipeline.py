#!/usr/bin/env python3
"""
Feature Pipeline: Fetches live weather data + simulates pollutants,
then stores records in MongoDB for AQI prediction.
"""

import os
import sys
import requests
from pymongo import MongoClient
from datetime import datetime, timezone
import numpy as np

# --- Environment Configuration ---
def load_env_vars():
    """Load and validate required environment variables."""
    required = ['MONGO_URI', 'OPENWEATHER_API_KEY']
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print(f"💡 Hint: Set these in GitHub Secrets or your .env file")
        for var in required:
            val = os.getenv(var)
            print(f"   {var}: {'✓ Set' if val else '✗ Missing'} (length: {len(val) if val else 0})")
        sys.exit(1)
    
    return {
        'MONGO_URI': os.getenv('MONGO_URI').strip(),
        'API_KEY': os.getenv('OPENWEATHER_API_KEY').strip(),
        'LAT': float(os.getenv('LATITUDE', '24.8607')),
        'LON': float(os.getenv('LONGITUDE', '67.0011')),
        'CITY': os.getenv('CITY', 'Karachi'),
        'COUNTRY': os.getenv('COUNTRY', 'PK')
    }

def calculate_aqi_from_pm25(pm25: float) -> float:
    """Calculate AQI from PM2.5 using EPA breakpoint method."""
    if pm25 <= 12.0: return (pm25 / 12.0) * 50
    elif pm25 <= 35.4: return 51 + ((pm25 - 12.1) / 23.3) * 49
    elif pm25 <= 55.4: return 101 + ((pm25 - 35.5) / 19.9) * 49
    elif pm25 <= 150.4: return 151 + ((pm25 - 55.5) / 94.9) * 49
    elif pm25 <= 250.4: return 201 + ((pm25 - 150.5) / 99.9) * 99
    elif pm25 <= 350.4: return 301 + ((pm25 - 250.5) / 99.9) * 99
    else: return min(500, 401 + ((pm25 - 350.5) / 149.9) * 99)

def fetch_weather_data(api_key: str, lat: float, lon: float) -> dict:
    """Fetch current weather from OpenWeatherMap API."""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat, 'lon': lon,
        'appid': api_key, 'units': 'metric'
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def simulate_pollutants(weather: dict, hour: int) -> dict:
    """Generate realistic pollutant estimates based on weather + time heuristics."""
    temp = weather['main']['temp']
    humidity = weather['main']['humidity']
    wind = weather['wind']['speed']  # m/s
    pressure = weather['main'].get('pressure', 1013)
    
    # Time-based factors
    rush_factor = 1.4 if (7 <= hour <= 9 or 17 <= hour <= 20) else 1.0
    night_factor = 0.85 if (22 <= hour or hour <= 5) else 1.0
    
    # Weather-based factors
    wind_factor = max(0.4, 1.0 - (wind / 25))  # Wind disperses pollution
    humidity_factor = 1.1 if humidity > 70 else 0.95  # High humidity can trap pollutants
    
    base_pm25 = 40 * rush_factor * night_factor * wind_factor * humidity_factor
    
    # Add realistic noise
    def noise(mean: float, std: float) -> float:
        return round(mean + np.random.normal(0, std), 2)
    
    return {
        'pm10 (µg/m³)': noise(base_pm25 * 1.9, 6),
        'pm2_5 (µg/m³)': noise(base_pm25, 3.5),
        'carbon_monoxide (µg/m³)': noise(420 * rush_factor * wind_factor, 55),
        'carbon_dioxide (ppm)': noise(415 + (temp - 20) * 2, 12),
        'nitrogen_dioxide (µg/m³)': noise(38 * rush_factor, 6),
        'sulphur_dioxide (µg/m³)': noise(16 + pressure * 0.01, 2.5),
        'ozone (µg/m³)': noise(32 * (1 + max(0, temp - 25)/40), 5.5),
        'dust (µg/m³)': noise(np.random.uniform(5, 15), 3),
    }

def main():
    print("🔄 Feature Pipeline Started")
    config = load_env_vars()
    
    try:
        # 1. Fetch Weather Data
        print(f"🌤️  Fetching weather for {config['CITY']}, {config['COUNTRY']}...")
        weather = fetch_weather_data(config['API_KEY'], config['LAT'], config['LON'])
        
        # 2. Prepare Record
        dt = datetime.now(timezone.utc)
        hour = dt.hour
        
        pollutants = simulate_pollutants(weather, hour)
        
        record = {
            'time': dt,
            'location': {
                'city': config['CITY'],
                'country': config['COUNTRY'],
                'coordinates': {'lat': config['LAT'], 'lon': config['LON']}
            },
            **pollutants,
            'temperature_2m (°C)': round(weather['main']['temp'], 1),
            'relative_humidity_2m (%)': round(weather['main']['humidity'], 1),
            'wind_speed_10m (km/h)': round(weather['wind']['speed'] * 3.6, 1),
            'pressure_hPa': weather['main'].get('pressure', 1013),
            'aqi_calculated': round(calculate_aqi_from_pm25(pollutants['pm2_5 (µg/m³)']), 1),
            'pipeline_version': '1.2.0',
            'data_source': 'openweathermap+simulation'
        }
        
        # 3. Store in MongoDB
        print("💾 Inserting record into MongoDB...")
        client = MongoClient(config['MONGO_URI'], serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Verify connection
        db = client['air_quality']
        collection = db['raw_aqi']
        
        result = collection.insert_one(record)
        
        print(f"✅ Success! Record ID: {result.inserted_id}")
        print(f"   📊 PM2.5: {record['pm2_5 (µg/m³)']} µg/m³ | AQI: {record['aqi_calculated']}")
        print(f"   🌡️  Temp: {record['temperature_2m (°C)']}°C | 💨 Wind: {record['wind_speed_10m (km/h)']} km/h")
        
        client.close()
        print("✨ Feature Pipeline Completed")
        sys.exit(0)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
