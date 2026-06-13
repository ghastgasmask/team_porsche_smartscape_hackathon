import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import io

from utils.data_loader import load_aqi_data, load_bin_data, get_district_list
from modules.eco_score import (
    calculate_eco_score, score_to_grade, score_to_color,
    grade_interpretation, generate_radar_chart, generate_district_comparison,
)

st.set_page_config(page_title="EcoScore - UrbanEco Monitor", page_icon="Eco", layout="wide")
st.title("Module 4: Different District Ecological Health Score")
st.caption("AI weighted environmental health score")
#load data
df_aqi  = load_aqi_data()
df_bins = load_bin_data()
districts = get_district_list(df_aqi)

with st.sidebar:
    st.header("Controls")
    district = st.selectbox("Select District", districts)
    st.markdown("---")
    st.markdown("**Override** *u may leave at 0*")
    manual_aqi    = st.slider("Air Quality Score", 0, 100, 0)
    manual_waste  = st.slider("Waste Management Score",0, 100, 0)
    manual_vision = st.slider("Vision Score", 0, 100, 0)

def auto_aqi_score(district: str) -> float:
    #Convert AQI to an invereted  0 to 100 score
    sub = df_aqi[df_aqi["district"] == district]
    if sub.empty:
        return 50.0
    mean_aqi = sub["aqi"].mean()
    # AQI 0 -> score 100; 
    # AQI 200+ → score 0
    return round(max(0.0, min(100.0, 100 - mean_aqi / 2.0)), 1)

def auto_waste_score(district: str) -> float:
    # Convert  fill_level to an inverted 0 to 100 score
    sub = df_bins[df_bins["district"] == district]
    if sub.empty:
        return 50.0
    mean_fill = sub["fill_level"].mean()
    return round(max(0.0, min(100.0, 100 - mean_fill)), 1)

aqi_score = manual_aqi if manual_aqi > 0 else auto_aqi_score(district)
waste_score = manual_waste if manual_waste > 0 else auto_waste_score(district)
vision_score = manual_vision if manual_vision > 0 else 62.0 # example

eco = calculate_eco_score(aqi_score, waste_score, vision_score)
grade = score_to_grade(eco)
color = score_to_color(eco)

top_l, top_m, top_r = st.columns([2, 1, 2])
#claude suggested visuals
with top_m:
    st.markdown(
        f"""
        <div style="background:#1a1a2e;border:3px solid {color};border-radius:16px;
                    padding:2rem;text-align:center;">
          <div style="font-size:0.9rem;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.12em;">
            {district} EcoHealth
          </div>
          <div style="font-size:5rem;font-weight:900;color:{color};line-height:1.0;margin:0.4rem 0;">
            {eco:.0f}
          </div>
          <div style="font-size:2.5rem;font-weight:700;color:{color};">Grade {grade}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_l:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Air Quality Score",     f"{aqi_score:.1f}/100")
    st.metric("Waste Management Score",     f"{waste_score:.1f}/100")
    st.metric("Vision Score",          f"{vision_score:.1f}/100")

with top_r:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Interpretation**")
    interp = grade_interpretation(grade)
    if grade in ("A", "B"):
        st.success(interp)
    elif grade == "C":
        st.warning(interp)
    else:
        st.error(interp)

st.markdown("---")

col_radar, col_bar = st.columns([1, 1], gap="large")

with col_radar:
    st.subheader("Score Radar")
    scores_dict = {
        "Air Quality":aqi_score,
        "Waste Management": waste_score,
        "Green Space":vision_score,
        "Overall Health":eco,
    }
    try:
        fig_radar = generate_radar_chart(scores_dict)
        buf = io.BytesIO()
        fig_radar.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                          facecolor="#0e1117")
        buf.seek(0)
        st.image(buf, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(fig_radar)
    except Exception as e:
        st.error(f"Radar chart error: {e}")

with col_bar:
    st.subheader("District Comparison")
    # Compute eco score for all districts
    all_scores = {}
    for d in districts:
        a = auto_aqi_score(d)
        w = auto_waste_score(d)
        v = 62.0  #  vision for all
        all_scores[d] = calculate_eco_score(a, w, v)

    try:
        fig_bar = generate_district_comparison(all_scores)
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

st.markdown("---")
st.subheader("Summary")

import pandas as pd
rows = []
for d in districts:
    a = auto_aqi_score(d)
    w = auto_waste_score(d)
    v = 62.0
    s = calculate_eco_score(a, w, v)
    rows.append({
        "District": d,
        "Air Quality": f"{a:.1f}",
        "Waste Mgmt": f"{w:.1f}",
        "Vision": f"{v:.1f}",
        "EcoScore": f"{s:.1f}",
        "Grade": score_to_grade(s),
    })

df_summary = pd.DataFrame(rows).sort_values("EcoScore", ascending=False).reset_index(drop=True)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.caption("Vision scores are put as 62 until Module 1 is trained")