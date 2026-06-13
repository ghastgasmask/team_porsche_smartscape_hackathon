import sys, os, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_aqi_data, get_district_list
from modules.air_quality import (
    preprocess_aqi_data, build_lstm_model, train_model,
    load_trained_model, forecast_next_48h, TF_AVAILABLE,
)

st.set_page_config(page_title="Air Quality - UrbanEco Monitor", page_icon="Eco", layout="wide")

st.markdown("""
<style>
.aqi-good    { color: #2ecc71; font-weight: 700; }
.aqi-mod     { color: #f1c40f; font-weight: 700; }
.aqi-bad     { color: #e74c3c; font-weight: 700; }
.section-hdr { font-size: 1.1rem; font-weight: 600; color: #95a5a6;
               text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("Module 1: Air Quality forecast")
st.caption("LSTM-based 48-hour AQI predictions across Astana districts")


with st.sidebar:
    st.header("Controls")
    df_full = load_aqi_data()
    districts = get_district_list(df_full)
    district = st.selectbox("District", districts, index=0)
    date_min = df_full["date"].min().date()
    date_max = df_full["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=15, step=5)


df = df_full[df_full["district"] == district].copy()
if len(date_range) == 2:
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]
df = df.sort_values("date").reset_index(drop=True)


st.markdown('<div class="section-hdr">Current Conditions</div>', unsafe_allow_html=True)
latest = df.iloc[-1]

def aqi_label(v):
    if v <= 50:   return "Good", "aqi-good"
    if v <= 100:  return "Moderate", "aqi-mod"
    return "Unhealthy", "aqi-bad"

lbl, cls = aqi_label(latest["aqi"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AQI", f"{latest['aqi']:.0f}", lbl)
c2.metric("PM2.5 (μg/m³)", f"{latest['pm25']:.1f}")
c3.metric("PM10 (μg/m³)",  f"{latest['pm10']:.1f}")
c4.metric("NO₂ (μg/m³)",  f"{latest['no2']:.1f}")
c5.metric("Temperature °C", f"{latest['temperature']:.1f}")

st.markdown('<div class="section-hdr">AQI Trend (based on already existing data)</div>', unsafe_allow_html=True)

def aqi_zone_color(v):
    if v <= 50:  return "#2ecc71"
    if v <= 100: return "#f1c40f"
    return "#e74c3c"

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df["date"], y=df["aqi"],
    mode="lines",
    line=dict(color="#3498db", width=1.5),
    name="AQI",
    fill="tozeroy",
    fillcolor="rgba(52,152,219,0.1)",
))
# AQI zone bands
for threshold, color, label in [(50, "rgba(46,204,113,0.08)", "Good"),
                                  (100, "rgba(241,196,15,0.08)", "Moderate"),
                                  (300, "rgba(231,76,60,0.08)", "Unhealthy")]:
    fig_hist.add_hrect(y0=0, y1=50, fillcolor="rgba(46,204,113,0.08)", line_width=0)
    fig_hist.add_hrect(y0=50, y1=100, fillcolor="rgba(241,196,15,0.08)", line_width=0)
    fig_hist.add_hrect(y0=100, y1=300, fillcolor="rgba(231,76,60,0.08)", line_width=0)
    break  # add once

fig_hist.update_layout(
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    xaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d")),
    yaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d"), title="AQI"),
    legend=dict(font=dict(color="#ecf0f1")),
    margin=dict(l=0, r=0, t=10, b=0),
    height=260,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown('<div class="section-hdr">LSTM Forecasting</div>', unsafe_allow_html=True)

if not TF_AVAILABLE:
    st.error("TensorFlow is not installed")
else:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        train_btn = st.button("Train LSTM Model", type="primary", use_container_width=True)
    with col_info:
        model, scaler = load_trained_model()
        if model:
            st.success("Trained model loaded")
        else:
            st.info("Looks like this is your first time here? No saved model found rn. Click **Train LSTM Model**")

    if train_btn:
        if len(df) < 30:
            st.warning("not enough data")
        else:
            with st.spinner("Preprocessing"):
                try:
                    X, y, scaler = preprocess_aqi_data(df)
                    model, history, metrics = train_model(X, y, epochs=epochs)

                    # Save scaler right here so forecast always uses the matched one
                    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
                    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "..", "models", "aqi_scaler.pkl"))

                    # Store in session state so forecast below always uses the matching scaler
                    st.session_state["aqi_model"] = model
                    st.session_state["aqi_scaler"] = scaler
                
                    hist_df = pd.DataFrame({
                        "epoch": range(1, len(history.history["loss"]) + 1),
                        "train_loss": history.history["loss"],
                        "val_loss": history.history.get("val_loss", history.history["loss"]),
                    })
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=hist_df["epoch"], y=hist_df["train_loss"],
                        name="Train MAE", line=dict(color="#3498db", width=2)
                    ))
                    fig_loss.add_trace(go.Scatter(
                        x=hist_df["epoch"], y=hist_df["val_loss"],
                        name="Val MAE", line=dict(color="#e67e22", width=2, dash="dash")
                    ))
                    fig_loss.update_layout(
                        title="Training Loss Curve",
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        xaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d"), title="Epoch"),
                        yaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d"), title="MAE"),
                        legend=dict(font=dict(color="#ecf0f1")),
                        margin=dict(l=0, r=0, t=40, b=0), height=250,
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                    st.success(
                        f"Model trained for {len(hist_df)} epochs"
                        f"Test MAE: {metrics['mae']:.4f} | Test RMSE: {metrics['rmse']:.4f}"
                    )
                except Exception as e:
                    st.error(f"Training failed {e}")

    # Resolve model + scaler: prefer in-session (just trained) over disk
    active_model = st.session_state.get("aqi_model", model)
    active_scaler = st.session_state.get("aqi_scaler", scaler)

    # 48h forecast
    if active_model is not None and active_scaler is not None and len(df) >= 30:
        st.markdown("**48h AQI Forecast**")
        st.markdown("READ THIS!!! ZOOM IN TO SEE THE DETAILS!!! The graph is interactive, buttons on top-right.")
        try:
            df_district = df_full[df_full["district"] == district].sort_values("date").reset_index(drop=True)
            preds_scaled, std_val = forecast_next_48h(active_model, active_scaler, df_district)

            hours = pd.date_range(df["date"].max() + pd.Timedelta(hours=1), periods=48, freq="h")
            std = std_val if std_val > 0 else np.std(preds_scaled) * 0.3
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hours, y=preds_scaled + std,
                mode="lines", line=dict(width=0), showlegend=False,
                fillcolor="rgba(52,152,219,0.15)", fill="tonexty", name="Upper bound"
            ))
            fig_fc.add_trace(go.Scatter(
                x=hours, y=preds_scaled - std,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(52,152,219,0.15)",
                name="Confidence Band"
            ))
            fig_fc.add_trace(go.Scatter(
                x=hours, y=preds_scaled,
                mode="lines+markers", line=dict(color="#3498db", width=2),
                marker=dict(size=4), name="Forecast AQI"
            ))
            # Zone lines
            fig_fc.add_hline(y=50,  line=dict(color="#2ecc71", dash="dot", width=1), annotation_text="Good/Moderate", annotation_font_color="#2ecc71")
            fig_fc.add_hline(y=100, line=dict(color="#e74c3c", dash="dot", width=1), annotation_text="Unhealthy", annotation_font_color="#e74c3c")
            fig_fc.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                xaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d")),
                yaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d"), title="AQI", range=[0, 200]),
                legend=dict(font=dict(color="#ecf0f1")),
                margin=dict(l=0, r=0, t=10, b=0), height=280,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Top 3 worst predicted periods 
            st.markdown("**Top 3 Periods with Worst Predicted Air Quality**")
            fc_df = pd.DataFrame({"Time": hours, "Predicted AQI": preds_scaled})
            top_3 = fc_df.sort_values("Predicted AQI", ascending=False).head(3).reset_index(drop=True)
            top_3["Predicted AQI"] = top_3["Predicted AQI"].round(1)
            top_3["Status"] = top_3["Predicted AQI"].apply(
                lambda val: "Unhealthy" if val > 100 else ("Moderate" if val > 50 else "Good")
            )
            st.dataframe(top_3, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"idk {e}")
    elif active_model is None:
        st.info("Press above to see the 48h forecast.")