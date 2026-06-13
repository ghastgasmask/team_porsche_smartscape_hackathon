
import streamlit as st
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ICON_DIR = APP_DIR / "pages" / "icons"
URBAN_ICON = ICON_DIR / "urban_icon.png"
WASTE_ICON = ICON_DIR / "waste_icon.png"
WIND_ICON = ICON_DIR / "wind_icon.webp"
ECOLOGY_ICON = ICON_DIR / "ecology_icon.png"
URBAN_ECOLOGY_IMAGE = ICON_DIR / "urban_ecology_image.png"

st.set_page_config(
    page_title="UrbanEco Monitor",
    page_icon="Eco",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("# UrbanEco Monitor")
    st.markdown("**City (urban) environmental health analysis**")
    st.divider()
    st.markdown(
        """
        **Navigation**
        Please, use the pages in the sidebar to explore each module!!
        """
    )

st.markdown(
    """
    <style>
    .hero-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1.25rem;
        width: 100%;
    }
    .hero-image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto 1rem auto;
        width: 150px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2ecc71, #27ae60, #1abc9c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        width: 100%;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2.5rem;
        width: 100%;
    }
    .module-icon {
        width: 72px;
        height: 72px;
        object-fit: contain;
        margin: 0 auto 0.75rem auto;
        display: block;
    }
    .module-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2ecc7133;
        border-radius: 12px;
        padding: 1.5rem;
        width: 100%;
        min-height: 190px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: border-color 0.3s;
    }
    .module-card:hover { border-color: #2ecc71aa; }
    .module-name { font-size: 1.1rem; font-weight: 700; color: #ecf0f1; margin-bottom: 0.5rem; }
    .module-desc { font-size: 0.9rem; color: #95a5a6; line-height: 1.4; }
    .disclaimer {
        background: #1a1a2e;
        border-left: 4px solid #2ecc71;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #95a5a6;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero: use native Streamlit columns to guarantee true centering
_left, _center, _right = st.columns([1, 2, 1])
with _center:
    st.image(str(URBAN_ECOLOGY_IMAGE), width=150, use_container_width=False)

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">UrbanEco Monitor</div>
        <div class="hero-subtitle">Real time AI, YOLO, ML, LSTM analysis of urban environmental health in Astana, Kazakhstan.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

modules = [
    {
        "image": WIND_ICON,
        "name": "Air Quality forecast",
        "desc": "LSTM-based 48-hour AQI predictions across Astana districts",
        "page": "pages/1_Air_Quality.py",
    },
    {
        "image": WASTE_ICON,
        "name": "Smart Waste collection",
        "desc": "Fill level prediction through ML + greedy neighbour TSP route optimization",
        "page": "pages/2_Waste_Management.py",
    },
    {
        "image": URBAN_ICON,
        "name": "Urban Image analysis",
        "desc": "YOLO version 8 + OpenCV vegetation, cleanliness, green scoring & detection",
        "page": "pages/3_Vision_Analysis.py",
    },
    {
        "image": ECOLOGY_ICON,
        "name": "Different District EcoHealth scoring",
        "desc": "Weighted ecological health score with radar & charts for comparison",
        "page": "pages/4_Eco_Score.py",
    },
]

import base64 # TO FIX IMAGES

def _img_to_b64(path: Path) -> str: # TO FIX IMAGES
    suffix = path.suffix.lstrip(".") # TO FIX IMAGES
    mime = "image/webp" if suffix == "webp" else f"image/{suffix}" # TO FIX IMAGES
    with open(path, "rb") as f: # TO FIX IMAGES
        data = base64.b64encode(f.read()).decode() # TO FIX IMAGES
    return f"data:{mime};base64,{data}" # TO FIX IMAGES

cols = st.columns(4)
for col, mod in zip(cols, modules):
    with col:
        icon_src = _img_to_b64(mod["image"])
        st.markdown(
            f"""
            <div class="module-card" style="text-align:center;">
                <img class="module-icon" src="{icon_src}">
                <div class="module-name">{mod['name']}</div>
                <div class="module-desc">{mod['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.subheader("Overview of OUR PLATFORM!")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Different Districts Monitored", "5", "Astana")
m2.metric("Data Points", "1825", "AQI records")
m3.metric("Different Waste Bins Tracked", "50", "GPS tagged btw")
m4.metric("Different Artificial Intelligence Modules", "4", "Are Active")

st.markdown(
    '<div class="disclaimer">'
    "<strong>SmartScape. Track 2: Ecology &amp; Urban Environment</strong> -> -> -> "
    "MADE BY PORSCHE TEAM! -> -> -> SOME SYNTHETIC DATA"
    "</div>",
    unsafe_allow_html=True,
)