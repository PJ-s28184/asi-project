import logging
from fastapi import FastAPI
from pydantic import BaseModel

from .feature_adapter import adapt_features
from .model import get_model
from .db import save_prediction

# Inicjalizacja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class Features(BaseModel):
    number: int
    driver_number: int
    lap_number: int
    kph: float
    top_speed: float
    season: int
    round: int


class Prediction(BaseModel):
    lap_time_s: float
    model_version: str


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Ładowanie modelu przy starcie aplikacji."""
    try:
        get_model()
        logger.info("Model został pomyślnie załadowany podczas startupu.")
    except Exception as e:
        logger.error(f"Błąd podczas ładowania modelu na starcie: {e}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    """Endpoint obsługujący predykcję czasu okrążenia."""
    model, model_version = get_model()

    # Konwersja danych (Pydantic -> dict -> DataFrame przez adapter)
    X = adapt_features(payload.dict())

    # Wywołanie predykcji modelu (model to teraz obiekt TabularPredictor)
    y_pred = model.predict(X)

    # Obsługa formatu zwracanego przez AutoGluon (zwykle Series/ndarray)
    if hasattr(y_pred, "iloc"):
        lap_time = float(y_pred.iloc[0])
    else:
        lap_time = float(y_pred[0])

    # Zapis predykcji do bazy danych
    try:
        save_prediction(
            payload=payload.dict(),
            prediction=lap_time,
            model_version=model_version,
        )
    except Exception as e:
        logger.warning(f"Nie udało się zapisać predykcji do bazy: {e}")

    return {
        "lap_time_s": lap_time,
        "model_version": model_version,
    }
