import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MTCars MPG Prediction API",
    description=(
        "Predicts fuel efficiency (MPG) from vehicle weight and horsepower "
        "using a linear regression model trained on the classic mtcars dataset."
    ),
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

if model:
    logger.info("Model loaded from %s", MODEL_PATH)
else:
    logger.warning("Model not found at %s — /predict will return 503", MODEL_PATH)


class PredictionRequest(BaseModel):
    wt: float = Field(..., gt=0, description="Vehicle weight in thousands of lbs (e.g. 2.62)")
    hp: float = Field(..., gt=0, description="Gross horsepower (e.g. 110)")


@app.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    """Returns service status and current UTC timestamp."""
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", summary="Readiness probe")
def ready() -> dict[str, bool]:
    """Returns 200 when the model is loaded and ready; 503 otherwise."""
    if not model:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"ready": True}


@app.post("/predict", summary="Predict MPG")
def predict(request: PredictionRequest) -> dict[str, float]:
    """Accepts vehicle weight and horsepower; returns predicted miles per gallon."""
    if not model:
        raise HTTPException(status_code=503, detail="model not loaded")
    prediction = model.predict([[request.wt, request.hp]])[0]
    logger.info("predict wt=%.2f hp=%.0f → mpg=%.2f", request.wt, request.hp, prediction)
    return {"predicted_mpg": round(float(prediction), 2)}
