from fastapi.testclient import TestClient
from api.main import app
from api.db import engine, text

client = TestClient(app)


def test_validation_error():
    r = client.post("/predict", json={"lap_number": "oops"})
    assert r.status_code == 422


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_saves_to_db():
    payload = {
        "number": 22,
        "driver_number": 7,
        "lap_number": 12,
        "kph": 150.0,
        "top_speed": 300.0,
        "season": 2022,
        "round": 4,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    response = r.json()
    assert "lap_time_s" in response
    assert "model_version" in response

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
        )
        row = result.fetchone()
        assert row is not None
        assert row.prediction is not None
