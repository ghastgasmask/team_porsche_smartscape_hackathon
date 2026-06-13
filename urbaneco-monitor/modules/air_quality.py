import os
import shutil
import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[air_quality] TensorFlow not found. LSTM features will be unavailable.")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

_MODULE_DIR  = os.path.dirname(__file__)
_MODELS_DIR  = os.path.join(_MODULE_DIR, "..", "models")
MODEL_PATH   = os.path.join(_MODELS_DIR, "lstm_aqi.h5")
SCALER_PATH  = os.path.join(_MODELS_DIR, "aqi_scaler.pkl")

LOOK_BACK   = 168          # week hourly lookback window
N_FEATURES  = 3            # [PM2.5, sin_time, cos_time]
FEATURE_IDX = 0            # 

# Columns used when training on the real Beijing dataset;
RAW_AQI_COL = "pm25"      

# constants 
FEATURE_COLS = ["pm25", "pm10", "no2", "o3", "temperature", "humidity", "wind_speed"]
TARGET_COL   = "aqi"
LAG_DAYS     = [1, 3, 7]


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.day_of_year

    df["sin_time"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_time"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df[[RAW_AQI_COL, "sin_time", "cos_time"]].values.astype("float32")


def preprocess_aqi_data(df: pd.DataFrame):
    matrix = _prepare_features(df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(matrix)

    X_seq, y_seq = [], []
    for i in range(len(scaled) - LOOK_BACK):
        X_seq.append(scaled[i : i + LOOK_BACK, :])
        y_seq.append(scaled[i + LOOK_BACK, FEATURE_IDX])

    return np.array(X_seq), np.array(y_seq), scaler


def build_lstm_model(input_shape: tuple = (LOOK_BACK, N_FEATURES)):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build the LSTM model.")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mae"])
    return model

def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 128):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to train the LSTM model.")

    # 80/20 split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    # Evaluate on test set
    preds = model.predict(X_test, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    metrics = {"rmse": round(rmse, 5), "mae": round(mae, 5), "test_samples": len(y_test)}

    os.makedirs(_MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    # Save the scaler so forecast_next_48h can use the exact same scaling
    if JOBLIB_AVAILABLE:
        joblib.dump(scaler, SCALER_PATH)

    return model, history, metrics


def load_trained_model():
    if not TF_AVAILABLE or not JOBLIB_AVAILABLE:
        return None, None

    model, scaler = None, None

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"[air_quality] Could not load model: {e}")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"[air_quality] Could not load scaler: {e}")
    return model, scaler


def forecast_next_48h(model, scaler: MinMaxScaler, recent_df: pd.DataFrame):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for forecasting.")

    # upsample from Daily to Hourly
    df = recent_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [
        col for col in df.columns
        if col != "date" and pd.api.types.is_numeric_dtype(df[col])
    ]
    df = df.set_index("date")[numeric_cols]

    # Resample daily data to hourly
    df_hourly = df.resample("h").interpolate(method="linear").reset_index()
    
    df_hourly["day_of_year"] = df_hourly["date"].dt.dayofyear
    df_hourly["sin_time"] = np.sin(2 * np.pi * df_hourly["day_of_year"] / 365.25)
    df_hourly["cos_time"] = np.cos(2 * np.pi * df_hourly["day_of_year"] / 365.25)
    
    features_matrix = df_hourly[["pm25", "sin_time", "cos_time"]].values.astype("float32")
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_matrix = scaler.fit_transform(features_matrix)
    else:
        scaled_matrix = scaler.transform(features_matrix)
        
    window = scaled_matrix[-LOOK_BACK:].copy()
    
    current_time = df_hourly["date"].iloc[-1]
    predictions_scaled = []
    
    for _ in range(48):
        x_input = window[np.newaxis, ...]  # shape (1, 168, 3)
        pred_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        predictions_scaled.append(pred_scaled)
        
        # Advance time by 1 hour
        current_time += pd.Timedelta(hours=1)
        doy = current_time.dayofyear
        next_sin = np.sin(2 * np.pi * doy / 365.25)
        next_cos = np.cos(2 * np.pi * doy / 365.25)
        
        dummy_row = np.array([[0.0, next_sin, next_cos]])
        dummy_scaled = scaler.transform(dummy_row)[0]
        
        new_step = np.array([pred_scaled, dummy_scaled[1], dummy_scaled[2]])
        
        window = np.vstack([window[1:], new_step])
        
    predictions_scaled = np.array(predictions_scaled)
    dummy_out = np.zeros((len(predictions_scaled), N_FEATURES), dtype="float32")
    dummy_out[:, FEATURE_IDX] = predictions_scaled
    predictions_raw = scaler.inverse_transform(dummy_out)[:, FEATURE_IDX]
    predictions_raw = np.clip(predictions_raw, 0, 500)
    
    seed_vals = scaled_matrix[-30:, FEATURE_IDX]
    std_band = float(np.std(seed_vals)) * (scaler.data_max_[FEATURE_IDX] - scaler.data_min_[FEATURE_IDX])
    
    return predictions_raw, std_band