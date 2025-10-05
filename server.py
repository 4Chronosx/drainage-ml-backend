from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow local frontend + railway domain
origins = [
    "http://localhost:3000",
    "https://drainage-ml-backend-production-87a9.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] for all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model + scaler + ranking
scaler_100 = joblib.load("models/100yr/scaler_100yr.pkl")
kmeans_100 = joblib.load("models/100yr/kmeans_100yr.pkl")
cluster_ranking_100 = joblib.load("models/100yr/cluster_ranking_100yr.pkl")
categories_100 = joblib.load("models/100yr/categories_100yr.pkl")

scaler_50 = joblib.load("models/50yr/scaler_50yr.pkl")
kmeans_50 = joblib.load("models/50yr/kmeans_50yr.pkl")
cluster_ranking_50 = joblib.load("models/50yr/cluster_ranking_50yr.pkl")
categories_50 = joblib.load("models/50yr/categories_50yr.pkl")

scaler_25 = joblib.load("models/25yr/scaler_25yr.pkl")
kmeans_25 = joblib.load("models/25yr/kmeans_25yr.pkl")
cluster_ranking_25 = joblib.load("models/25yr/cluster_ranking_25yr.pkl")
categories_25 = joblib.load("models/25yr/categories_25yr.pkl")

# Define request format
class DataPoint(BaseModel):
    point: list[float]

@app.post("/predict-100yr")
def predict_100yr(data: dict):
    # Example input: {"point": [0.5, 1000, 300.0, 0.000014]}
    X = np.array([data["point"]])
    X_scaled = scaler_100.transform(X)
    cluster_label = kmeans_100.predict(X_scaled)[0]
    ranked_label = cluster_ranking_100[cluster_label]
    category = categories_100[ranked_label]
    return {"category": category}

@app.post("/predict-50yr")
def predict_100yr(data: dict):
    # Example input: {"point": [0.5, 1000, 300.0, 0.000014]}
    X = np.array([data["point"]])
    X_scaled = scaler_100.transform(X)
    cluster_label = kmeans_100.predict(X_scaled)[0]
    ranked_label = cluster_ranking_100[cluster_label]
    category = categories_100[ranked_label]
    return {"category": category}

@app.post("/predict-25yr")
def predict_100yr(data: dict):
    # Example input: {"point": [0.5, 1000, 300.0, 0.000014]}
    X = np.array([data["point"]])
    X_scaled = scaler_100.transform(X)
    cluster_label = kmeans_100.predict(X_scaled)[0]
    ranked_label = cluster_ranking_100[cluster_label]
    category = categories_100[ranked_label]
    return {"category": category}