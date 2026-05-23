import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --- /health ---

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "time" in data


# --- /ready ---

def test_ready_returns_valid_status():
    response = client.get("/ready")
    # 200 when model is loaded, 503 when it isn't — both are correct behaviour
    assert response.status_code in (200, 503)


def test_ready_200_has_ready_flag():
    response = client.get("/ready")
    if response.status_code == 200:
        assert response.json() == {"ready": True}


# --- /predict (happy path) ---

def test_predict_valid_input():
    response = client.post("/predict", json={"wt": 2.62, "hp": 110})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mpg" in data
    assert isinstance(data["predicted_mpg"], float)


def test_predict_value_in_plausible_range():
    data = client.post("/predict", json={"wt": 2.62, "hp": 110}).json()
    assert 5 < data["predicted_mpg"] < 60


# --- /predict (validation failures) ---

def test_predict_negative_wt_returns_422():
    response = client.post("/predict", json={"wt": -1.0, "hp": 110})
    assert response.status_code == 422


def test_predict_negative_hp_returns_422():
    response = client.post("/predict", json={"wt": 2.62, "hp": -50})
    assert response.status_code == 422


def test_predict_missing_hp_returns_422():
    response = client.post("/predict", json={"wt": 2.62})
    assert response.status_code == 422


def test_predict_missing_wt_returns_422():
    response = client.post("/predict", json={"hp": 110})
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    response = client.post("/predict", json={"wt": "heavy", "hp": 110})
    assert response.status_code == 422
