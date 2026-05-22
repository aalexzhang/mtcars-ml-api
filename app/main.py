from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone

from pathlib import Path
import joblib

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class PredictionRequest(BaseModel):
    wt: float
    hp: float

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "the API is running :) ",
        "time": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/ready")
def ready() -> dict[str, bool]:
    if not model:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"ready": True}

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if not model:
        raise HTTPException(status_code=503, detail="model not loaded")
    prediction = model.predict([[request.wt, request.hp]])[0]
    return {"predicted_mpg": prediction}