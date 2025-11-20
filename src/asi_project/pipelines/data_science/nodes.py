from typing import Tuple, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyszczenie danych:
    - Usuwa duplikaty
    - Tworzy lap_time_seconds i pit_time_seconds (braki pit=0)
    - Usuwa wiersze z brakami w kolumnach numerycznych i target
    """
    if df is None or df.empty:
        return df

    df_clean = df.drop_duplicates()

    if "lap_time_seconds" not in df_clean.columns and "lap_time" in df_clean.columns:
        df_clean["lap_time_seconds"] = pd.to_timedelta(
            df_clean["lap_time"], errors="coerce"
        ).dt.total_seconds()

    if "pit_time_seconds" not in df_clean.columns and "pit_time" in df_clean.columns:
        df_clean["pit_time_seconds"] = pd.to_timedelta(
            df_clean["pit_time"], errors="coerce"
        ).dt.total_seconds()
    if "pit_time_seconds" in df_clean.columns:
        df_clean["pit_time_seconds"] = df_clean["pit_time_seconds"].fillna(0)

    if "season" in df_clean.columns and df_clean["season"].dtype == "object":
        df_clean["season"] = pd.to_numeric(df_clean["season"], errors="coerce")

    if "lap_time_seconds" in df_clean.columns:
        df_clean = df_clean[df_clean["lap_time_seconds"].notna()]

    numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
    if "pit_time_seconds" in numeric_cols:
        numeric_cols.remove("pit_time_seconds")

    if numeric_cols:
        mask = df_clean[numeric_cols].notna().all(axis=1)
        df_clean = df_clean[mask]

    return df_clean


def split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not in dataframe")

    y = df[target_column]
    X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
    if X.shape[1] == 0:
        X = pd.DataFrame({"bias": 1.0}, index=df.index)

    if y.isna().any() or X.isna().any().any():
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]

    n = len(X)
    if n < 1:
        raise ValueError("Brak danych po przygotowaniu. Sprawdź dane i cel.")
    if n == 1:
        X_train, X_test = X.copy(), X.copy()
        y_train, y_test = y.copy(), y.copy()
    elif n == 2:
        X_train, X_test = X.iloc[:1], X.iloc[1:]
        y_train, y_test = y.iloc[:1], y.iloc[1:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test


def train_baseline(
    X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series
) -> Tuple[RandomForestRegressor, str]:
    model = RandomForestRegressor(
        random_state=42, n_estimators=100, n_jobs=-1, verbose=0
    )
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    model.fit(X_train, y_train)

    from pathlib import Path
    import joblib

    out_path = Path("data/06_models/model_baseline.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return model, str(out_path)


def evaluate(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
    log_to_wandb: bool = True,
    log_artifact_path: Optional[str] = None,
) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

    if log_to_wandb:
        try:
            import wandb

            wandb.init(project="asi-project", reinit=True)
            wandb.log(metrics)
            if log_artifact_path:
                try:
                    from pathlib import Path
                    import joblib

                    tmp_path = Path("./model_baseline_tmp.pkl")
                    joblib.dump(model, tmp_path)
                    artifact = wandb.Artifact("model_baseline", type="model")
                    artifact.add_file(str(tmp_path))
                    wandb.log_artifact(artifact)
                except Exception:
                    pass
            wandb.finish()
        except Exception:
            pass

    return pd.DataFrame([metrics])
