import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Load & Preprocess Data
df = pd.read_csv("hybrid_energy_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Handle missing values via linear time interpolation
df = df.interpolate(method="linear")

# 2. Feature Engineering (Temporal + Wind Vector + Historical Lag Modalities)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"]))

# Lag features (1-hour prior generation modality)
df["prev_solar_mw"] = df["solar_power_mw"].shift(1).bfill()
df["prev_wind_mw"] = df["wind_power_mw"].shift(1).bfill()

# 3. Multimodal Feature Fusion
feature_cols = [
    "solar_irradiance", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "temperature", "humidity", "cloud_cover", "pressure", "rainfall",
    "hour_sin", "hour_cos", "month", "prev_solar_mw", "prev_wind_mw"
]

X = df[feature_cols]
y_solar = df["solar_power_mw"]
y_wind = df["wind_power_mw"]
y_hybrid = df["hybrid_power_mw"]

# 4. Chronological Data Split: 70% Train | 15% Val | 15% Test (Slide 7)
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train_s, y_val_s, y_test_s = y_solar[:train_end], y_solar[train_end:val_end], y_solar[val_end:]
y_train_w, y_val_w, y_test_w = y_wind[:train_end], y_wind[train_end:val_end], y_wind[val_end:]
y_train_h, y_val_h, y_test_h = y_hybrid[:train_end], y_hybrid[train_end:val_end], y_hybrid[val_end:]

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Train Baseline Model (Random Forest) vs Multimodal XGBoost
baseline_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
baseline_model.fit(X_train_scaled, y_train_h)
base_preds = baseline_model.predict(X_test_scaled)

# Multimodal XGBoost Models for Solar, Wind, and Hybrid
model_solar = XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, random_state=42)
model_wind = XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, random_state=42)
model_hybrid = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)

model_solar.fit(X_train_scaled, y_train_s, eval_set=[(X_val_scaled, y_val_s)], verbose=False)
model_wind.fit(X_train_scaled, y_train_w, eval_set=[(X_val_scaled, y_val_w)], verbose=False)
model_hybrid.fit(X_train_scaled, y_train_h, eval_set=[(X_val_scaled, y_val_h)], verbose=False)

xgb_preds = model_hybrid.predict(X_test_scaled)

# 6. Performance Evaluation (Slide 9 Metrics)
def get_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R2": round(r2_score(y_true, y_pred), 4)
    }

metrics_summary = {
    "Baseline (Random Forest)": get_metrics(y_test_h, base_preds),
    "Multimodal XGBoost": get_metrics(y_test_h, xgb_preds)
}

print("=== Test Set Evaluation (Hybrid Power MW) ===")
for name, m in metrics_summary.items():
    print(f"{name} -> MAE: {m['MAE']} MW | RMSE: {m['RMSE']} MW | R2: {m['R2']}")

# Save models, scaler, and test results for the web prototype
test_results = df.iloc[val_end:].copy()
test_results["pred_hybrid_mw"] = np.maximum(0, xgb_preds)
test_results.to_csv("test_predictions.csv", index=False)

joblib.dump({
    "scaler": scaler,
    "model_solar": model_solar,
    "model_wind": model_wind,
    "model_hybrid": model_hybrid,
    "feature_cols": feature_cols,
    "metrics": metrics_summary
}, "hybrid_forecast_artifacts.pkl")

print("Artifacts saved successfully!")