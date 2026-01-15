# src/api/db.py
import json
import datetime as dt
from typing import Union

from sqlalchemy import create_engine, text

from asi_project.settings import settings as app_settings

# Konfiguracja w pliku (drugi poziom priorytetu po zmiennych środowiskowych).
# W przypadku braku zmiennych środowiskowych można uzupełnić poniższą wartość.
DATABASE_URL_CONFIG: str | None = None  # np. "sqlite:///local.db"

# Wartość domyślna (trzeci poziom priorytetu).
DATABASE_URL_DEFAULT = "sqlite:///local.db"


def _resolve_database_url() -> str:
    """
    Ustalanie URL bazy danych według priorytetu:
    1) konfiguracja z env/.env (DATABASE_URL)
    2) wartość skonfigurowana w pliku (DATABASE_URL_CONFIG)
    3) wartość domyślna (DATABASE_URL_DEFAULT)
    """
    if "DATABASE_URL" in getattr(app_settings, "model_fields_set", set()):
        return app_settings.DATABASE_URL

    return DATABASE_URL_CONFIG or DATABASE_URL_DEFAULT


# Utworzenie silnika bazy danych (singleton na poziomie modułu).
engine = create_engine(_resolve_database_url(), future=True)


def save_prediction(
    payload: dict,
    prediction: Union[float, int],
    model_version: str,
) -> None:
    """
    Zapis pojedynczej predykcji do tabeli predictions.
    Tabela jest tworzona automatycznie, jeśli nie istnieje.

    Kolumny:
    - id           : klucz główny (auto)
    - ts           : znacznik czasu (UTC, tekst lub TIMESTAMP)
    - payload      : dane wejściowe (JSON jako TEXT/JSONB)
    - prediction   : wynik predykcji (REAL/DOUBLE PRECISION)
    - model_version: wersja modelu użytego do predykcji
    """
    with engine.begin() as conn:
        backend = engine.url.get_backend_name()

        if backend == "sqlite":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT,
                        payload TEXT,
                        prediction REAL,
                        model_version TEXT
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        ts TIMESTAMP,
                        payload JSONB,
                        prediction DOUBLE PRECISION,
                        model_version TEXT
                    )
                    """
                )
            )

        conn.execute(
            text(
                """
                INSERT INTO predictions (ts, payload, prediction, model_version)
                VALUES (:ts, :payload, :pred, :ver)
                """
            ),
            {
                "ts": dt.datetime.utcnow().isoformat(),
                "payload": json.dumps(payload),
                "pred": float(prediction),
                "ver": model_version,
            },
        )
