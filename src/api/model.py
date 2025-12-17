# src/api/model.py
from typing import Any, Tuple

import joblib

from asi_project.settings import settings as app_settings

# Konfiguracja w pliku (drugi poziom priorytetu po zmiennych środowiskowych).
# W przypadku braku zmiennych środowiskowych można uzupełnić poniższe wartości.
MODEL_PATH_CONFIG: str | None = None  # np. "data/06_models/ag_production.pkl"
MODEL_VERSION_CONFIG: str | None = None  # np. "autogluon-v1"

# Wartości domyślne (trzeci poziom priorytetu).
MODEL_PATH_DEFAULT = "data/06_models/ag_production.pkl"
MODEL_VERSION_DEFAULT = "local-file"

# Cache modelu na poziomie modułu (singleton).
_MODEL: Any | None = None
_MODEL_VERSION: str = "local-dev"


def _resolve_model_path() -> str:
    """
    Ustalanie ścieżki do pliku modelu według priorytetu:
    1) konfiguracja z env/.env (MODEL_PATH)
    2) wartość skonfigurowana w pliku (MODEL_PATH_CONFIG)
    3) wartość domyślna (MODEL_PATH_DEFAULT)
    """
    return (
        app_settings.MODEL_PATH
        or MODEL_PATH_CONFIG
        or MODEL_PATH_DEFAULT
    )


def _resolve_model_version() -> str:
    """
    Ustalanie wersji modelu według priorytetu:
    1) konfiguracja z env/.env (MODEL_VERSION)
    2) wartość skonfigurowana w pliku (MODEL_VERSION_CONFIG)
    3) wartość domyślna (MODEL_VERSION_DEFAULT)
    """
    return (
        app_settings.MODEL_VERSION
        or MODEL_VERSION_CONFIG
        or MODEL_VERSION_DEFAULT
    )


def _load_local_model() -> Tuple[Any, str]:
    """
    Ładowanie modelu z lokalnego pliku przy użyciu ustalonej ścieżki.
    """
    model_path = _resolve_model_path()
    from pathlib import Path

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    version = _resolve_model_version()
    return model, version


def get_model() -> Tuple[Any, str]:
    """
    Zwrócenie krotki (model, model_version).
    Pierwsze wywołanie ładuje model z pliku i zapisuje w pamięci,
    kolejne korzystają z cache na poziomie modułu.
    """
    global _MODEL, _MODEL_VERSION

    if _MODEL is None:
        model, version = _load_local_model()
        _MODEL = model
        _MODEL_VERSION = version

    return _MODEL, _MODEL_VERSION
