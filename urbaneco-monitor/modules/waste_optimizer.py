import os
import math
import shutil
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

DEPOT_LAT = 51.1801
DEPOT_LON = 71.4460

_REFERENCE_DATE = datetime.date(2024, 1, 1)

def _ensure_models_placed():
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    for filename in ["waste_gbm_model.pkl", "aqi_scaler.pkl", "lstm_aqi.h5"]:
        src = os.path.join(root_dir, filename)
        dst = os.path.join(models_dir, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")

_ensure_models_placed()

MODEL_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "waste_gbm_model.pkl")

def load_trained_waste_model():
    if os.path.exists(MODEL_BUNDLE_PATH):
        try:
            return joblib.load(MODEL_BUNDLE_PATH)
        except Exception as e:
            print(f"Error loading bundle: {e}")
    return None

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["last_collected"]):
        df["last_collected"] = pd.to_datetime(df["last_collected"])

    if "days_since_collected" not in df.columns:
        df["days_since_collected"] = (
            pd.Timestamp(_REFERENCE_DATE) - df["last_collected"]
        ).dt.days.clip(lower=1)

    if "fill_rate" not in df.columns:
        df["fill_rate"] = (df["fill_level"] / df["days_since_collected"]).round(4)

    return df

def preprocess_bin_data(df: pd.DataFrame):
    df = _add_derived_features(df)

    feature_cols = [
        'days_since_collected', 'fill_rate', 'latitude', 'longitude', 
        'district_Baikonur', 'district_Esil', 'district_Nura', 'district_Saryarka', 
        'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday', 
        'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday'
    ]

    X = _build_features_for_model(df, feature_cols)
    y = df["fill_level"].values

    return X, y, None, feature_cols

def _build_features_for_model(df: pd.DataFrame, feature_columns: list) -> np.ndarray:
    df = _add_derived_features(df)

    df['district'] = df['district'].replace({'Almaty': 'Almaty district'})

    cols = ['days_since_collected', 'fill_rate', 'latitude', 'longitude', 'district', 'day_of_week']
    df_feat = df[cols].copy()

    df_feat = pd.get_dummies(df_feat, columns=['district', 'day_of_week'], drop_first=True)

    df_feat = df_feat.reindex(columns=feature_columns, fill_value=0)

    return df_feat.values

def train_fill_predictor(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.07,
        random_state=42,
    )
    model.fit(X, y)
    return model

def predict_full_bins(
    model,
    df: pd.DataFrame,
    pipeline=None,
    feature_names=None,
    threshold: float = 75.0,
) -> list[str]:
    bundle = load_trained_waste_model()

    if bundle is not None and isinstance(bundle, dict):
        model_obj = bundle['model']
        scaler = bundle['scaler']
        feat_cols = bundle['feature_columns']

        X = _build_features_for_model(df, feat_cols)
        X_scaled = scaler.transform(X)
        preds = model_obj.predict(X_scaled)
    else:
        feat_cols = feature_names if feature_names is not None else [
            'days_since_collected', 'fill_rate', 'latitude', 'longitude', 
            'district_Baikonur', 'district_Esil', 'district_Nura', 'district_Saryarka', 
            'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday', 
            'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday'
        ]
        X = _build_features_for_model(df, feat_cols)
        preds = model.predict(X)

    mask = preds >= threshold
    return df.loc[mask, "bin_id"].tolist()

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def optimize_route(bins_df: pd.DataFrame, full_bin_ids: list[str]) -> list[tuple[float, float]]:
    subset = bins_df[bins_df["bin_id"].isin(full_bin_ids)].copy()
    if subset.empty:
        return [(DEPOT_LAT, DEPOT_LON)]

    stops = list(zip(subset["latitude"], subset["longitude"]))
    visited = [False] * len(stops)
    route = [(DEPOT_LAT, DEPOT_LON)]
    current = (DEPOT_LAT, DEPOT_LON)

    for _ in range(len(stops)):
        best_dist = float("inf")
        best_idx = -1
        for i, (lat, lon) in enumerate(stops):
            if not visited[i]:
                d = _haversine(current[0], current[1], lat, lon)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        if best_idx >= 0:
            visited[best_idx] = True
            current = stops[best_idx]
            route.append(current)

    route.append((DEPOT_LAT, DEPOT_LON))
    return route

def route_distance_km(route: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        total += _haversine(route[i][0], route[i][1], route[i + 1][0], route[i + 1][1])
    return round(total, 2)

def calculate_savings(original_route_km: float, optimized_route_km: float) -> dict:
    dist_saved = max(0.0, original_route_km - optimized_route_km)
    fuel_saved = dist_saved * 0.35
    co2_saved = fuel_saved * 2.68
    return {
        "distance_saved_km": round(dist_saved, 2),
        "fuel_saved_L": round(fuel_saved, 3),
        "co2_saved_kg": round(co2_saved, 3),
        "original_km": round(original_route_km, 2),
        "optimized_km": round(optimized_route_km, 2),
    }
