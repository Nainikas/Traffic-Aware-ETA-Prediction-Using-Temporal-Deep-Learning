import sys
from pathlib import Path

# Add src/ to sys.path for model imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List
from model.lstm_model import LSTMETAModel

# Load trained LSTM model
model = LSTMETAModel()
model.load_state_dict(torch.load("models/lstm_eta_model.pth", map_location=torch.device("cpu")))
model.eval()

# Create FastAPI app
app = FastAPI(
    title="Traffic-Aware ETA Prediction API",
    version="1.0",
    description="Predicts ETA using LSTM on historical traffic speed data"
)

# ==== Schemas ====

# For single-segment prediction
class SpeedSequence(BaseModel):
    speeds: conlist(float, min_length=12, max_length=12)

# For multi-segment route prediction
class MultiSegmentInput(BaseModel):
    segments: List[conlist(float, min_length=12, max_length=12)]


# ==== Routes ====

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ETA predictor is running."}


@app.post("/predict")
def predict_eta(data: SpeedSequence):
    try:
        x = np.array(data.speeds, dtype=np.float32).reshape(1, 12, 1)
        x_tensor = torch.tensor(x)

        with torch.no_grad():
            prediction = model(x_tensor).item()

        return {"predicted_eta_minutes": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_route")
def predict_route_eta(data: MultiSegmentInput):
    try:
        segment_etas = []

        for speeds in data.segments:
            x = np.array(speeds, dtype=np.float32).reshape(1, 12, 1)
            x_tensor = torch.tensor(x)

            with torch.no_grad():
                eta = model(x_tensor).item()
                segment_etas.append(round(eta, 2))

        total_eta = round(sum(segment_etas), 2)

        return {
            "segment_etas": segment_etas,
            "total_eta": total_eta
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route prediction failed: {str(e)}")
