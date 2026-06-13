
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st

from modules.vision import analyze_image, CV2_AVAILABLE, YOLO_AVAILABLE

st.set_page_config(page_title="Vision - UrbanEco Monitor", page_icon="Eco", layout="wide")
st.title("Module 3: Urban Image Analysis")
st.caption("YOLOv8 + OpenCV vegetation scoring and urban cleanliness detection")

if not CV2_AVAILABLE:
    st.error("OpenCV not installed")
    st.stop()
if not YOLO_AVAILABLE:
    st.warning("Ultralytics not found")

uploaded = st.file_uploader(
    "Upload an urban scene image (street, park, building, etc.) You can, AND IS RECCOMENDED TO USE A GOOGLE EARTH PHOTO FROM ANYWHERE ON EARTH (well, city of course)",
    type=["jpg", "jpeg", "png"],
    help="Accepted: JPG, JPEG, PNG",
)

if uploaded is None:
    st.info("Upload an image to start analysis.")
    st.markdown("""
    **What this module analyses using Yolov8**
    - Vegetation coverage (green pixels. Uses HSV masking)
    - Litter / trash detection (YOLOv8)
    - Your good old Urban stuff: cars, people, benches, plants
    - Cleanliness score (BASED ON detected trash)
    - Using all INFO, Outputs an Overall Vision Score (0–100)
    """)
    st.stop()

from PIL import Image as PILImage

pil_img = PILImage.open(uploaded).convert("RGB")

with st.spinner("Running YOLO detection and OpenCV"):
    try:
        result = analyze_image(pil_img)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if "error" in result:
    st.error(result["error"])
    st.stop()

col_img, col_scores = st.columns([3, 2], gap="large")

with col_img:
    st.subheader("Detection Result")
    annotated = result.get("annotated_image")
    if annotated is not None and CV2_AVAILABLE:
        import cv2
        annotated_rgb = cv2.cvtColor(annotated.astype("uint8"), cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detected objects with bounding boxes", use_column_width=True)
    else:
        st.image(pil_img, caption="YOLO unavailable", use_column_width=True)

with col_scores:
    st.subheader("Scores")

    # Overall vision score
    vs = result["overall_vision_score"]
    if vs >= 70: score_color = "#2ecc71"
    elif vs >= 45: score_color = "#f1c40f"
    else: score_color = "#e74c3c"

    st.markdown(
        f"""
        <div style="background:#1a1a2e;border:2px solid {score_color};border-radius:12px;
                    padding:1.2rem;text-align:center;margin-bottom:1rem;">
          <div style="font-size:0.85rem;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.1em;">
            Overall Vision Score
          </div>
          <div style="font-size:3rem;font-weight:800;color:{score_color};line-height:1.1;">
            {vs:.0f}
          </div>
          <div style="font-size:0.8rem;color:#95a5a6;">out of 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Green Ratio",       f"{result['green_ratio']:.1f}%")
    c2.metric("Cleanliness Score", f"{result['cleanliness_score']:.0f}/100")

    st.markdown("---")
    st.markdown("**Interpretation**")
    if vs >= 70:
        st.success("This area shows good urban greenery and cleanliness. The council is satisfied with your photo")
    elif vs >= 45:
        st.warning("Moderate conditions. Some litter or limited green space detected")
    else:
        st.error("Poor urban environmental conditions detected in this image")







st.subheader("Detected Objects Breakdown (jojo ref)")
counts = result.get("object_counts", {})
if counts:
    import pandas as pd
    df_det = pd.DataFrame(
        [{"Object Class": k.capitalize(), "Count": v} for k, v in sorted(counts.items())]
    )
    col_tbl, col_bar = st.columns([1, 2])
    with col_tbl:
        st.dataframe(df_det, use_container_width=True, hide_index=True)
    with col_bar:
        try:
            import plotly.express as px
            fig = px.bar(
                df_det, x="Count", y="Object Class", orientation="h",
                color="Count", color_continuous_scale="Viridis",
            )
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                xaxis=dict(gridcolor="#2c3e50", tickfont=dict(color="#7f8c8d")),
                yaxis=dict(tickfont=dict(color="#ecf0f1")),
                coloraxis_showscale=False,
                margin=dict(l=0,r=0,t=10,b=0), height=max(150, len(counts)*45),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
else:
    st.info("No relevant urban objects detected....... How about going outside?")
