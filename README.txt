# UrbanEco Monitor

Real time urban environmental health analysis powered by AI

———————————————————————————————————————————————————————————————————————————————————————

## Track

Track 2 Ecology & Urban Environment
SmartScape Hackathon 2026
By Porsche Team
THERE ARE NOT MANY COMMITS, BECAUSE THE DEVELOPMENT FOLDER WAS COPIED, TRANSFERRED, AND MOVED FREQUENTLY
———————————————————————————————————————————————————————————————————————————————————————

## Problem 

Urban areas like Astana face growing challenges in air pollution, inefficient waste collection, and declining green space. 
It's made for only automobiles (seems like so). All of this directly impact resident's quality of life. 
City authorities (akimat in astana, for example) lack data-driven tools to monitor & predict environment. 
UrbanEco Monitor HELPS this gap by combining LSTMs, AI, MLs, satellite imagery, IoT sensor data, into one dashboard

———————————————————————————————————————————————————————————————————————————————————————

## Solution Overview

UrbanEco Monitor is a modular Streamlit dashboard with four modules (ALL WITH AI):
| 1 | MODULE                      | FUNCTION                                                |
| 1 | **Air Quality Forecasting** | LSTM-based 48-hour AQI predictions per district         |
| 2 | **Smart Waste Collection**  | ML fill-level prediction + greedy TSP route optimizer   |
| 3 | **Urban Image Analysis**    | YOLOv8 + OpenCV vegetation and cleanliness scoring      |
| 4 | **District EcoHealth Score**| Weighted score with radar + comparison charts           |

———————————————————————————————————————————————————————————————————————————————————————

## Tech Stack

| Layer           | Technology                               |
——————————————————————————————————————————————————————————————
| Dashboard       | Streamlit ≥ 1.32                         |
| Deep Learning   | TensorFlow/Keras (LSTM), PyTorch         |
| Computer Vision | YOLOv8 (Ultralytics), OpenCV             |
| Machine Learning| scikit-learn (GradientBoostingRegressor) |
| Visualization   | Plotly, Matplotlib, Folium               |
| Data            | Pandas, NumPy (data from Kaggle)         |
| Language        | Python 3.12                              |

———————————————————————————————————————————————————————————————————————————————————————
## How to Run
## LOCALLY 

```bash
# clone repo
git clone <repo-url>
cd urbaneco-monitor # -> mandatory to use

# you may create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard. Add "py -m" before "streamlit" if something doesn't work
streamlit run app.py
```

The app will open at `http://localhost:8501`
———————————————————————————————————————————————————————————————————————————————————————
## How to Run
## ONLINE 

!!! IT IS NOT HIGHLY RECCOMENDED TO USE THIS APP ONLINE, TRY TO USE IT LOCALLY !!!
But here's the link, with which you can open the app:
https://teamporschesmartscapehackathon.streamlit.app/

Streamlit heavily struggles with tensorflow, and is quite weak & slow when a lot of libraries are involved.
It is highly reccomended for you to clone/download the repo and run it locally on your PC.
Sometimes errors pop out just because they want and can pop up.
———————————————————————————————————————————————————————————————————————————————————————

## Dataset Info

Most datasets are FROM KAGGLE:

- https://www.kaggle.com/datasets/ferhats/sample-urban-waste-dataset: SAMPLE URBAN WASTE DATASET
- https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set: BEIJING AIR QUALITY DATASET
- https://universe.roboflow.com/otherside/trash-classification-qy72f/dataset/4/download/yolov8: A dataset with images of trash
YOLOv8 version was downloaded

District names used: **Esil, Almaty, Saryarka, Baikonur, Nura** (are all in Astana, Kazakhstan).

———————————————————————————————————————————————————————————————————————————————————————

## Project Structure - MADE BY CLAUDE

```
urbaneco-monitor/
│
├── app.py                         # hompage
├── requirements.txt               
├── .gitignore                     
│
├── models/
│   ├── .gitkeep                   
│   ├── aqi_scaler.pkl             # Scaler for LSTM
│   ├── lstm_aqi.h5                # Trained LSTM weights
│   └── waste_gbm_model.pkl        # Trained GBM Tabular weights
│
├── modules/                       # processing files
│   ├── air_quality.py             # air quality processor
│   ├── eco_score.py               # ecological health score processor & agggregration maths
│   ├── vision.py                  # image analyzer
│   └── waste_optimizer.py         # waste optimizer calculations
├── pages/
│   ├── icons/                     # Contains only the 5 graphic images (for app.py)
│   ├── 1_Air_Quality.py           
│   ├── 2_Waste_Management.py      
│   ├── 3_Vision_Analysis.py       
│   └── 4_Eco_Score.py             
│
├── runs/                          # Contains fine-tuned YOLO runs
├── utils/                         
└── train_yolo.py                  # read this to see our training logic
```

———————————————————————————————————————————————————————————————————————————————————————

## Architecture Diagram - MADE BY CLAUDE

```
                                ┌──────────────────────────┐
                                │     Streamlit UI Shell   │
                                │         (app.py)         │
                                └─────────────┬────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
        │   Air Quality Page    │ │ Waste Management Page │ │  Vision Analysis Page │
        │  (1_Air_Quality.py)   │ │ (2_Waste_Mgmt.py)     │ │ (3_Vision_Analysis.py)│
        └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                    │                         │                         │
                    ▼                         ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
        │   Air Quality Module  │ │    Waste Optimizer    │ │     Vision Module     │
        │   (air_quality.py)    │ │ (waste_optimizer.py)  │ │      (vision.py)      │
        │   ├── LSTM Model      │ │ ├── GBM Predictor     │ │  ├── YOLOv8 Detector  │
        │   └── MinMaxScaler    │ │ └── StandardScaler    │ │  └── HSV Green Mask   │
        └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                                ┌──────────────────────────┐
                                │   EcoHealth Score Page   │
                                │     (4_Eco_Score.py)     │
                                │   Composite Aggregation  │
                                └──────────────────────────┘
```
