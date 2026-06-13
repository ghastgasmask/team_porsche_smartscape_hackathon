"""
utils/data_loader.py
Shared data loading helpers for UrbanEco Monitor.
Generates synthetic CSV files on first run if they don't exist.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
DISTRICTS = ["Esil", "Almaty", "Saryarka", "Baikonur", "Nura"]

# Astana city center + per-district offsets
DISTRICT_CENTERS = {
    "Esil":     (51.1605, 71.4704),
    "Almaty":   (51.1879, 71.4460),
    "Saryarka": (51.1450, 71.3990),
    "Baikonur": (51.2100, 71.5200),
    "Nura":     (51.1300, 71.5600),
}

AQI_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_aqi.csv")
BINS_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_bins.csv")


# ── Synthetic data generators ──────────────────────────────────────────────────
def _generate_aqi_data() -> pd.DataFrame:
    """Generate 1825-row synthetic AQI dataset (365 days × 5 districts)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    rows = []

    for district in DISTRICTS:
        base_pm25 = rng.uniform(15, 45)
        for date in dates:
            day_of_year = date.day_of_year
            # Seasonal variation: worse in winter
            seasonal = 10 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
            temp = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + rng.normal(0, 3)
            humidity = rng.uniform(30, 80)
            wind_speed = rng.uniform(1, 12)

            pm25 = max(5, base_pm25 + seasonal + rng.normal(0, 5) - wind_speed * 0.5)
            pm10 = pm25 * rng.uniform(1.5, 2.0)
            no2 = rng.uniform(10, 60) + pm25 * 0.3
            o3 = max(5, rng.uniform(20, 80) - pm25 * 0.2 + temp * 0.5)

            # AQI correlated with PM2.5 and temperature
            aqi = min(300, max(0, pm25 * 2.1 + temp * 0.8 + rng.normal(0, 8)))

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "district": district,
                    "pm25": round(pm25, 2),
                    "pm10": round(pm10, 2),
                    "no2": round(no2, 2),
                    "o3": round(o3, 2),
                    "temperature": round(temp, 2),
                    "humidity": round(humidity, 2),
                    "wind_speed": round(wind_speed, 2),
                    "aqi": round(aqi, 1),
                }
            )

    return pd.DataFrame(rows)


def _generate_bins_data() -> pd.DataFrame:
    """Generate 50-row synthetic waste bin dataset across 5 Astana districts."""
    rng = np.random.default_rng(7)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows = []
    bin_id = 1

    for district in DISTRICTS:
        center_lat, center_lon = DISTRICT_CENTERS[district]
        for _ in range(10):  # 10 bins per district
            lat = center_lat + rng.uniform(-0.03, 0.03)
            lon = center_lon + rng.uniform(-0.04, 0.04)
            fill = round(rng.uniform(0, 100), 1)
            temp = round(rng.uniform(-5, 30), 1)
            last_days_ago = rng.integers(1, 8)
            last_collected = pd.Timestamp("2023-12-31") - pd.Timedelta(days=int(last_days_ago))

            rows.append(
                {
                    "bin_id": f"BIN-{bin_id:03d}",
                    "district": district,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "fill_level": fill,
                    "last_collected": last_collected.strftime("%Y-%m-%d"),
                    "day_of_week": days[last_collected.dayofweek],
                    "temperature": temp,
                }
            )
            bin_id += 1

    return pd.DataFrame(rows)


def _ensure_data_exists():
    """Create data directory and CSV files if they don't exist."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(AQI_DATA_PATH):
        df = _generate_aqi_data()
        df.to_csv(AQI_DATA_PATH, index=False)

    if not os.path.exists(BINS_DATA_PATH):
        df = _generate_bins_data()
        df.to_csv(BINS_DATA_PATH, index=False)


# ── Public loaders (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading AQI data…")
def load_aqi_data(path: str = None) -> pd.DataFrame:
    """Load and validate AQI CSV, generating it if missing."""
    _ensure_data_exists()
    path = path or AQI_DATA_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "district", "pm25", "pm10", "no2", "o3",
                "temperature", "humidity", "wind_speed", "aqi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AQI data missing columns: {missing}")
    return df


@st.cache_data(show_spinner="Loading bin data…")
def load_bin_data(path: str = None) -> pd.DataFrame:
    """Load and validate waste bin CSV, generating it if missing."""
    _ensure_data_exists()
    path = path or BINS_DATA_PATH
    df = pd.read_csv(path, parse_dates=["last_collected"])
    required = {"bin_id", "district", "latitude", "longitude",
                "fill_level", "last_collected", "day_of_week", "temperature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bin data missing columns: {missing}")
    return df


def get_district_list(df: pd.DataFrame) -> list[str]:
    """Return sorted unique district names from a dataframe."""
    return sorted(df["district"].unique().tolist())


# ── Standalone execution: pre-generate CSV files ───────────────────────────────
if __name__ == "__main__":
    _ensure_data_exists()
    print(f"AQI data:  {AQI_DATA_PATH}")
    print(f"Bins data: {BINS_DATA_PATH}")
    print("Sample data files generated successfully.")
