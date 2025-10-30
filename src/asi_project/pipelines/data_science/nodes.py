
"""
Proste nody do pipeline'u data_science (po studencku).
"""

from typing import Dict, Tuple, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Proste czyszczenie: usuń tylko duplikaty (NaN usuwa split)."""
    if df is None or df.empty:
        return df
    df_clean = df.drop_duplicates()
    # konwertuj season na int (Parquet lubi)
    if 'season' in df_clean.columns and df_clean['season'].dtype == 'object':
        df_clean['season'] = pd.to_numeric(df_clean['season'], errors='coerce')
    return df_clean


def split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Podział na train/test bez przecieku (wymaga kolumny celu)."""
    if target_column not in df.columns:
        raise ValueError("target_column not in dataframe")

    y = df[target_column]
    # użyj tylko cech numerycznych (LinearRegression tego potrzebuje)
    X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
    # jeśli brak cech numerycznych, dodaj prostą stałą (bias), żeby się w ogóle dało trenować
    if X.shape[1] == 0:
        X = pd.DataFrame({"bias": 1.0}, index=df.index)

    # odfiltruj wiersze z brakami w X lub y, żeby nie wyczyścić wszystkiego wcześniej
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    n = len(X)
    if n < 1:
        raise ValueError("Brak danych po filtracji. Sprawdź dane i cel.")
    if n == 1:
        # minimum: użyj tej samej próbki jako train i test
        X_train, X_test = X.copy(), X.copy()
        y_train, y_test = y.copy(), y.copy()
    elif n == 2:
        # minimalny split: 1 próbka train, 1 próbka test
        X_train, X_test = X.iloc[:1], X.iloc[1:]
        y_train, y_test = y.iloc[:1], y.iloc[1:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

    # convert y Series to DataFrame for parquet compatibility
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test


def train_baseline(
    X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series
) -> Tuple[LinearRegression, str]:
    """prosty baseline (LinearRegression) + zapis modelu na dysk."""
    model = LinearRegression()
    # unpack DataFrame to Series if needed
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    model.fit(X_train, y_train)

    from pathlib import Path
    import joblib  # type: ignore

    out_path = Path("data/06_models/model_baseline.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return model, str(out_path)


def evaluate(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
    log_to_wandb: bool = True,
    log_artifact_path: Optional[str] = None,
) -> pd.DataFrame:
    """Ewaluacja (MAE, RMSE, R², MAPE, MSE) i opcjonalne logowanie do W&B."""
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    
    # unpack DataFrame to Series if needed
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    y_pred = model.predict(X_test)
    
    # oblicz wszystkie metryki
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)
    
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

    if log_to_wandb:
        try:
            import wandb  # type: ignore

            wandb.init(project="asi-project", reinit=True)
            wandb.log(metrics)
            if log_artifact_path:
                try:
                    from pathlib import Path
                    import joblib  # type: ignore

                    tmp_path = Path("./model_baseline_tmp.pkl")
                    joblib.dump(model, tmp_path)
                    artifact = wandb.Artifact("model_baseline", type="model")
                    artifact.add_file(str(tmp_path))
                    wandb.log_artifact(artifact)
                except Exception:
                    pass
            wandb.finish()
        except Exception:
            # olej błędy logowania, ma się po prostu uruchomić
            pass

    return pd.DataFrame([metrics])
