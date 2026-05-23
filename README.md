# MTCars MPG Prediction API

A REST API that predicts a vehicle's fuel efficiency (miles per gallon) given its weight and horsepower. Built with FastAPI, scikit-learn, and containerized with Podman/Docker.

---

## Project Overview

This project trains a linear regression model on the classic [mtcars dataset](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html) and exposes it as an HTTP API. The API accepts a vehicle's weight and horsepower, and returns a predicted MPG value.

---

## Model Description

| Property | Value |
|---|---|
| Algorithm | Linear Regression (scikit-learn) |
| Training data | mtcars dataset (32 observations) |
| Target variable | `mpg` — miles per gallon |
| Input features | `wt` (weight), `hp` (horsepower) |
| Model file | `models/model.pkl` |

### Variables Used for Prediction

| Field | Type | Description | Example |
|---|---|---|---|
| `wt` | float | Vehicle weight in thousands of lbs | `2.62` |
| `hp` | float | Gross horsepower | `110` |

Both values must be greater than zero. The model is trained and saved to `models/model.pkl` via the training notebook.

---

## Repo Structure

```
mtcars-ml-api/
├── app/
│   └── main.py              # FastAPI application — endpoints and model inference
├── models/
│   └── model.pkl            # Serialized trained model (joblib)
├── scripts/
│   └── train_model.ipynb    # Jupyter notebook to retrain and save the model
├── tests/
│   └── test_api.py          # Automated API tests (pytest)
├── mtcars.csv               # Raw training dataset
├── Dockerfile               # Container build instructions (Podman/Docker compatible)
├── .dockerignore            # Files excluded from the container image
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Dev/test dependencies (includes pytest, httpx)
└── pytest.ini               # Pytest configuration
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Podman (or Docker)
- Jupyter (optional, only needed to retrain the model)

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd mtcars-ml-api
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 3. (Optional) Retrain the model

The pre-trained model is already included at `models/model.pkl`. To retrain from scratch:

```bash
jupyter notebook scripts/train_model.ipynb
```

Run all cells. The notebook loads `mtcars.csv`, trains a `LinearRegression` model on `wt` and `hp`, and saves the result to `models/model.pkl`.

### 4. Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`.  
Interactive docs are auto-generated with FastAPI at `http://localhost:8080/docs`.

---

## Running Tests

```bash
pytest tests/
```

The test suite covers the health check, readiness probe, a successful prediction, and several invalid-input cases.

---

## Podman Build and Run

The `Dockerfile` is fully compatible with Podman (and Docker).

### Build the image

```bash
podman build -t mtcars-ml-api .
```

### Run the container

```bash
podman run --rm -p 8080:8080 mtcars-ml-api
```

The API is now available at `http://localhost:8080`.

---

## API Endpoint Documentation

### `GET /health` — Liveness probe

Returns service status and current UTC time. Always returns 200 while the process is running.

**Response**
```json
{
  "status": "ok",
  "time": "2026-05-23T14:32:00.000000+00:00"
}
```

---

### `GET /ready` — Readiness probe

Returns 200 when the model is loaded and the API is ready to serve predictions. Returns 503 if the model file is missing.

**Response (200)**
```json
{ "ready": true }
```

**Response (503)**
```json
{ "detail": "model not loaded" }
```

---

### `POST /predict` — Predict MPG

Accepts vehicle weight and horsepower; returns predicted miles per gallon.

**Request body**

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `wt` | float | yes | > 0 | Weight in thousands of lbs |
| `hp` | float | yes | > 0 | Gross horsepower |

**Response (200)**
```json
{ "predicted_mpg": 22.35 }
```

**Response (422)** — returned when input is invalid (missing field, wrong type, or value ≤ 0)
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "wt"],
      "msg": "Input should be greater than 0"
    }
  ]
}
```

---

## Example API Call

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"wt": 2.62, "hp": 110}'
```

**Example response**

```json
{
  "predicted_mpg": 22.35
}
```

More examples:

```bash
# Liveness check
curl http://localhost:8080/health

# Readiness check
curl http://localhost:8080/ready

# Heavy, high-power car (expect lower MPG)
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"wt": 5.42, "hp": 335}'
```

---

## Deployment Instructions

The container image can be pushed to any OCI-compatible registry (Docker Hub, GitHub Container Registry, etc.) and run on any container platform.

### Example: deploy to a Linux VM or cloud instance

```bash
# On the build machine — build and export the image
podman build -t mtcars-ml-api .
podman save mtcars-ml-api | gzip > mtcars-ml-api.tar.gz

# On the target server
podman load < mtcars-ml-api.tar.gz
podman run -d -p 8080:8080 --name mtcars-api mtcars-ml-api
```

### Example: push to Docker Hub

```bash
podman tag mtcars-ml-api docker.io/<your-username>/mtcars-ml-api:latest
podman push docker.io/<your-username>/mtcars-ml-api:latest
```

---

## Deployed API URL

`https://https://mtcars-ml-api-472381907479.us-central1.run.app/predict`

---

## Production Features Included

| Feature | Implementation |
|---|---|
| Health check | `GET /health` |
| Readiness probe | `GET /ready` — returns 503 if model isn't loaded |
| Input validation | Pydantic `Field(gt=0)` with automatic 422 error responses |
| Error handling | HTTPException 503 when model is unavailable |
| Auto-generated docs | FastAPI Swagger UI at `/docs`, ReDoc at `/redoc` |
| Logging | Python `logging` — logs startup state and each prediction |
| Containerization | Dockerfile compatible with Podman and Docker |
| Automated tests | pytest suite covering health, readiness, predict, and invalid inputs |
