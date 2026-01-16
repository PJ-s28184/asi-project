import pickle
import os
import logging
from typing import Any, Tuple
from pathlib import Path
from autogluon.tabular import TabularPredictor

from asi_project.settings import settings as app_settings

# Inicjalizacja loggera dla komunikatów o ładowaniu
logger = logging.getLogger(__name__)

MODEL_PATH_CONFIG: str | None = None
MODEL_VERSION_CONFIG: str | None = None

MODEL_PATH_DEFAULT = "data/06_models/ag_production.pkl"
MODEL_VERSION_DEFAULT = "local-file"

_MODEL: Any | None = None
_MODEL_VERSION: str = "local-dev"


def _resolve_model_path() -> str:
    """Ustalanie ścieżki do pliku modelu."""
    return app_settings.MODEL_PATH or MODEL_PATH_CONFIG or MODEL_PATH_DEFAULT


def _resolve_model_version() -> str:
    """Ustalanie wersji modelu."""
    return app_settings.MODEL_VERSION or MODEL_VERSION_CONFIG or MODEL_VERSION_DEFAULT


def _load_local_model() -> Tuple[Any, str]:
    """
    Ładowanie modelu AutoGluon.
    Obsługuje sytuację, w której .pkl zawiera tylko ścieżkę do katalogu modelu.
    """
    model_path = _resolve_model_path()
    path_obj = Path(model_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku metadanych modelu: {model_path}")

    # 1. Wczytujemy zawartość pliku .pkl (może to być obiekt modelu lub tylko string ze ścieżką)
    with open(model_path, "rb") as f:
        content = pickle.load(f)

    # 2. Logika dla AutoGluon: jeśli content to string, traktujemy go jako folder modelu
    if isinstance(content, str):
        # Sprawdzamy czy jesteśmy w Dockerze (obecność /.dockerenv)
        if os.path.exists("/.dockerenv"):
            model_dir = Path("/app") / content
        else:
            model_dir = Path.cwd() / content

        logger.info(
            f"Wykryto ścieżkę tekstową w pkl. Ładowanie AutoGluon z: {model_dir}"
        )

        # Weryfikacja czy katalog istnieje
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Katalog modelu AutoGluon nie istnieje: {model_dir}"
            )

        # Ładujemy model z folderu (tam gdzie znajduje się predictor.pkl)
        model = TabularPredictor.load(str(model_dir))
    else:
        # Jeśli obiektem w pkl był już załadowany predictor
        model = content

    version = _resolve_model_version()
    return model, version


def get_model() -> Tuple[Any, str]:
    """Zwrócenie modelu z cache (Singleton)."""
    global _MODEL, _MODEL_VERSION

    if _MODEL is None:
        model, version = _load_local_model()
        _MODEL = model
        _MODEL_VERSION = version

    return _MODEL, _MODEL_VERSION
