# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from .model import get_model
from .db import save_prediction

app = FastAPI()


class Features(BaseModel):
    """
    Wejście do modelu - trzeba dolozyc pola - nie zrobie bo nie mam 06_models i conda nie dziala
    tak aby zgadzały się z tym, na czym model był trenowany.
    """
    driver_number: int
    lap_number: int
    crossing_finish_line_in_pit: bool = False

    kph: float
    top_speed: float

    driver_name: str
    race_class: str
    team: str
    manufacturer: str

    season: int
    circuit: str
    round: int

    vehicle: str


class Prediction(BaseModel):
    """
    Wyjście z modelu – lap_time_s (sekundy).
    Pola:
    - lap_time_s: przewidywany czas okrążenia w sekundach,
    - model_version: identyfikator wersji modelu (np. z konfiguracji).
    """
    lap_time_s: float
    model_version: str


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """
    Ładowanie modelu przy starcie aplikacji.
    Dzięki temu model nie jest wczytywany przy każdym żądaniu /predict.
    """
    get_model()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    """
    Endpoint obsługujący predykcję czasu okrążenia.
    Kroki:
    1) Pobranie modelu oraz wersji modelu.
    2) Konwersja danych wejściowych (Pydantic → dict → DataFrame).
    3) Wywołanie metody predict modelu.
    4) Zapis predykcji do bazy danych.
    5) Zwrócenie przewidywanego lap_time_s wraz z wersją modelu.
    """
    model, model_version = get_model()

    # Konwersja danych wejściowych do formatu oczekiwanego przez model.
    X = pd.DataFrame([payload.dict()])

    # Wywołanie predykcji modelu.
    y_pred = model.predict(X)

    # Obsługa typowych formatów zwracanych przez modele (Series / ndarray).
    if hasattr(y_pred, "iloc"):
        lap_time = float(y_pred.iloc[0])
    else:
        lap_time = float(y_pred[0])

    # Zapis predykcji do bazy danych.
    save_prediction(
        payload=payload.dict(),
        prediction=lap_time,
        model_version=model_version,
    )

    return {
        "lap_time_s": lap_time,
        "model_version": model_version,
    }