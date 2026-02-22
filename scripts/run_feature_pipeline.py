import os
import sys
import requests
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import certifi

# ----------------- Configuration -----------------
MONGO_URI = os.environ.get('MONGO_URI')
API_KEY = os.environ.get('OPENWEATHER_API_KEY')
LAT = 24.8607
LON = 67.0011

# ----------------- AQI Calculation -----------------
def calculate_aqi_from_pm25(pm25):
    """Calculate AQI from PM2.5 (EPA Standard)"""
    if pm25 <= 12.0: return (pm25 / 12.0) * 50
    elif pm25 <= 35.4: return 51 + ((pm25 - 12.1) / 23.3) * 49
    elif pm25 <= 55.4: return 101 + ((pm25 - 35.5) / 19.9) * 49
    elif pm25 <= 150.4: return 151 + ((pm25 - 55.5) / 94.9) * 49
    elif pm25 <= 250.4: return 201 + ((pm25 - 150.5) / 99.9) * 99
    elif pm25 <= 350.4: return 301 + ((pm25 - 250.5) / 99.9) * 99
    else: return 401 + ((pm25 - 350.5) / 149.9) * 99

# ----------------- Main Pipeline -----------------
def main():
    if not MONGO_URI or not API_KEY:
        print("❌ Missing MONGO_URI or OPENWEATHER_API_KEY")
        sys.exit(1)

    print("🔄 Feature Pipeline Started")
    print(f"🌤️  Fetching weather for {LAT}, {LON}...")

    try:
        # 1️⃣ Fetch Weather from OpenWeather API
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        dt = datetime.now()
        hour = dt.hour

        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind = data['wind']['speed']

        # 2️⃣ Simulate Pollutants
        rush_factor = 1.3 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.0
        wind_factor = max(0.5, 1.0 - (wind / 20))
        base_pm25 = 45 * rush_factor * wind_factor

        record = {
            'time': dt,
            'pm10 (µg/m³)': round(base_pm25 * 1.8 + np.random.normal(0, 5), 2),
            'pm2_5 (µg/m³)': round(base_pm25 + np.random.normal(0, 3), 2),
            'carbon_monoxide (µg/m³)': round(400 * rush_factor * wind_factor + np.random.normal(0, 50), 2),
            'carbon_dioxide (ppm)': round(420 + np.random.normal(0, 10), 2),
            'nitrogen_dioxide (µg/m³)': round(35 * rush_factor + np.random.normal(0, 5), 2),
            'sulphur_dioxide (µg/m³)': round(15 + np.random.normal(0, 2), 2),
            'ozone (µg/m³)': round(30 * (1 + (temp - 20)/50) + np.random.normal(0, 5), 2),
            'dust (µg/m³)': round(np.random.uniform(0, 10), 2),
            'temperature_2m (°C)': round(temp, 1),
            'relative_humidity_2m (%)': round(humidity, 1),
            'wind_speed_10m (km/h)': round(wind * 3.6, 1),
        }

        # 3️⃣ Connect to MongoDB with TLS
        client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000  # 30s timeout
        )

        # Verify connection
        client.admin.command('ping')

        print("💾 Inserting record into MongoDB...")
        db = client['air_quality']
        collection = db['raw_aqi']
        collection.insert_one(record)
        print(f"✅ Success: Inserted record for {dt}")
        print(f"   PM2.5: {record['pm2_5 (µg/m³)']} | Temp: {record['temperature_2m (°C)']}°C")
        client.close()

    except requests.HTTPError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ----------------- Run Script -----------------
if __name__ == "__main__":
    main()
