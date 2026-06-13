import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_bin_data, get_district_list
from modules.waste_optimizer import (
    preprocess_bin_data, train_fill_predictor, predict_full_bins,
    optimize_route, route_distance_km, calculate_savings,
)

st.set_page_config(page_title="Waste - UrbanEco Monitor", page_icon="Eco", layout="wide")
st.title("Module 2: Smart Waste Collection")
st.caption("ML fill-level prediction and greedy TSP route optimization")

df = load_bin_data()

with st.sidebar:
    st.header("Controls")
    selected_district = st.selectbox("Filter District", ["All"] + get_district_list(df))
    fill_threshold = st.slider("Full bin Threshold (%)", 50, 95, 75, 5)

df_view = df if selected_district == "All" else df[df["district"] == selected_district]

st.subheader("Bin Fill Levels Map")

def fill_color(level):
    if level < 40:  return "#2ecc71"
    if level < 75:  return "#f1c40f"
    return "#e74c3c"

df_view = df_view.copy()
df_view["color"] = df_view["fill_level"].apply(fill_color)
df_view["size"]  = df_view["fill_level"].apply(lambda x: max(8, x * 0.18))

fig_map = go.Figure()
for c, lbl in [("#2ecc71","Low (<40%)"),("#f1c40f","Medium 40–75%"),("#e74c3c","High >75%")]:
    sub = df_view[df_view["color"] == c]
    if not sub.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["latitude"], lon=sub["longitude"], mode="markers",
            marker=dict(size=sub["size"], color=c, opacity=0.85),
            text=sub.apply(lambda r: f"<b>{r['bin_id']}</b><br>{r['district']}<br>Fill: {r['fill_level']:.1f}%", axis=1),
            hoverinfo="text", name=lbl,
        ))

fig_map.update_layout(
    mapbox=dict(style="carto-darkmatter", center=dict(lat=51.18, lon=71.45), zoom=11),
    margin=dict(l=0,r=0,t=0,b=0),
    legend=dict(font=dict(color="#ecf0f1"), bgcolor="rgba(14,17,23,0.8)"),
    paper_bgcolor="#0e1117", height=400,
)
st.plotly_chart(fig_map, use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
st.subheader("Tomorrow's Full Bins Prediction")

if "pred_ids" not in st.session_state:
    st.session_state.pred_ids = []
    st.session_state.route    = []
    st.session_state.savings  = {}

if st.button("Predict Tomorrow's Full Bins", type="primary"):
    with st.spinner("Training model and running predictions…"):
        try:
            X, y, pipeline, feature_names = preprocess_bin_data(df)
            model = train_fill_predictor(X, y)
            ids   = predict_full_bins(model, df, pipeline, feature_names, threshold=fill_threshold)
            route = optimize_route(df, ids)
            naive_km = route_distance_km(
                [(r.latitude, r.longitude) for r in df[df["bin_id"].isin(ids)].itertuples()]
            ) * 1.3 if ids else 0
            opt_km = route_distance_km(route)
            st.session_state.pred_ids = ids
            st.session_state.route    = route
            st.session_state.savings  = calculate_savings(naive_km, opt_km)
            st.success(f"{len(ids)} bins predicted to be ≥{fill_threshold}% full tomorrow.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if st.session_state.pred_ids:
    ids   = st.session_state.pred_ids
    route = st.session_state.route
    svgs  = st.session_state.savings

    # Route map
    st.markdown("**Optimized Collection Route**")
    fig_r = go.Figure()
    fig_r.add_trace(go.Scattermapbox(
        lat=df["latitude"], lon=df["longitude"], mode="markers",
        marker=dict(size=7, color="#7f8c8d", opacity=0.45), name="All Bins",
        text=df["bin_id"], hoverinfo="text",
    ))
    df_fp = df[df["bin_id"].isin(ids)]
    fig_r.add_trace(go.Scattermapbox(
        lat=df_fp["latitude"], lon=df_fp["longitude"], mode="markers",
        marker=dict(size=14, color="#e74c3c"), name=f"Full Bins ({len(ids)})",
        text=df_fp["bin_id"], hoverinfo="text",
    ))
    if len(route) > 1:
        fig_r.add_trace(go.Scattermapbox(
            lat=[p[0] for p in route], lon=[p[1] for p in route],
            mode="lines+markers", line=dict(width=3, color="#f39c12"),
            marker=dict(size=5, color="#f39c12"), name="Route",
        ))
    fig_r.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=51.18, lon=71.45), zoom=11),
        margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(font=dict(color="#ecf0f1"), bgcolor="rgba(14,17,23,0.8)"),
        paper_bgcolor="#0e1117", height=360,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Savings
    st.subheader("Route Savings")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Original Route",  f"{svgs.get('original_km',0):.1f} km")
    c2.metric("Optimized Route", f"{svgs.get('optimized_km',0):.1f} km")
    c3.metric("Distance Saved",f"{svgs.get('distance_saved_km',0):.1f} km")
    c4.metric("CO₂ Avoided",  f"{svgs.get('co2_saved_kg',0):.2f} kg")
    st.metric("Fuel Saved", f"{svgs.get('fuel_saved_L',0):.3f} L")

    st.markdown("**Bins Scheduled for Collection**")
    tbl = df[df["bin_id"].isin(ids)][
        ["bin_id","district","latitude","longitude","fill_level","last_collected"]
    ].sort_values("fill_level", ascending=False).reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, height=220)
