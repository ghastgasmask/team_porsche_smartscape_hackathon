
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not found")


def calculate_eco_score(
    aqi_score: float,
    waste_score: float,
    vision_score: float,
    weights: tuple = (0.4, 0.35, 0.25),
) -> float:
    w_aqi, w_waste, w_vision = weights
    score = w_aqi * aqi_score + w_waste * waste_score + w_vision * vision_score
    return round(min(100.0, max(0.0, score)), 1)


def score_to_grade(score: float) -> str:
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    return "F"


def score_to_color(score: float) -> str:
    if score >= 85:
        return "#2ecc71"
    elif score >= 70:
        return "#f1c40f"
    elif score >= 55:
        return "#e67e22"
    elif score >= 40:
        return "#e74c3c"
    return "#8e44ad"


def grade_interpretation(grade: str) -> str:
    interpretations = {
        "A": "Excellent environmental health. The district maintains: 1) high air quality, "
             "2) efficient waste management, and 3) abundant green space, and also 4) 0 percent corruption or crimson!! (DRYAD IS SATISFIED)",
        "B": "Good environmental conditions. Minor areas for improvement. "
             "Continue monitoring and maintaining current standards (insert terraria ref here plz )",
        "C": "Moderate environmental health. Targeted interventions in the weakest "
             "disctrict areas are recommended",
        "D": "Below-average conditions. Immediate action needed to improve air quality, "
             "waste management, & urban greenery.",
        "F": "Critical environmental conditions. Urgent multi-sector response required. Nevermind... It might be over for us",
    }
    return interpretations.get(grade, "No interpretation available. I also hate people who are rude to others btw")


def generate_radar_chart(scores_dict: dict) -> plt.Figure:
    categories = ["Air Quality", "Waste Management", "Green Space", "Overall Health"]
    values = [scores_dict.get(cat, 0.0) for cat in categories]
    values += values[:1]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#ecf0f1", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#7f8c8d", fontsize=7)
    ax.yaxis.grid(color="#2c3e50", linestyle="--", alpha=0.5)
    ax.xaxis.grid(color="#2c3e50", linestyle="--", alpha=0.5)

    ax.plot(angles, values, linewidth=2, linestyle="solid", color="#2ecc71")
    ax.fill(angles, values, alpha=0.25, color="#2ecc71")

    ax.scatter(angles[:-1], values[:-1], color="#2ecc71", s=60, zorder=5)

    plt.tight_layout()
    return fig

# it's going to be alright
# or all right?
# ALL LEFTTTTTT
# ALL HAIL
#

def generate_district_comparison(districts_data: dict) -> "go.Figure":
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required to generate charts")

    districts = list(districts_data.keys())
    scores = [districts_data[d] for d in districts]
    colors = [score_to_color(s) for s in scores]
    grades = [score_to_grade(s) for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=districts,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1a1a2e", width=1)),
            text=[f"{s:.1f} ({g})" for s, g in zip(scores, grades)],
            textposition="outside",
            textfont=dict(color="#ecf0f1", size=12),
        )
    )
    fig.update_layout(
        title=dict(text="District EcoHealth Scores", font=dict(color="#ecf0f1", size=16)),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis=dict(
            range=[0, 115],
            tickfont=dict(color="#7f8c8d"),
            gridcolor="#2c3e50",
            title=dict(text="Score (0–100)", font=dict(color="#7f8c8d")),
        ),
        yaxis=dict(tickfont=dict(color="#ecf0f1")),
        margin=dict(l=10, r=60, t=50, b=30),
        height=300,
    )
    return fig
