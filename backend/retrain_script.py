# retrain_script.py
import os
import sys
import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Connect to Mongo
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["aqi_dashboard"]

# 2. Load Data
cursor = db["weather_data"].find({}).sort("timestamp", -1).limit(5000)
df = pd.DataFrame(list(cursor))

if df.empty:
    print("No data found in DB to train.")
    sys.exit(1)

# 3. Preprocessing (Same as main.py)
# (Insert your exact preprocessing logic: Rename columns, create hour/month features, etc.)
# For brevity, assuming columns are already cleaned or clean them here:
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df['hour'] = df.index.hour
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek

# Define X and y
# Make sure these match your model's expected input
feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'pm25', 'pm10', 'no2', 'so2', 'o3', 'co', 'hour', 'month', 'day_of_week']
X = df[feature_cols]
y = df['aqi'] # Assuming you saved AQI in DB, otherwise calculate it

# 4. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

pipe.fit(X_train, y_train)

# 5. Evaluate
preds = pipe.predict(X_test)
r2 = r2_score(y_test, preds)
print(f"New Model R2 Score: {r2:.4f}")

# 6. Save Artifact
os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipe, "artifacts/model.pkl")
print("Model saved to artifacts/model.pkl")