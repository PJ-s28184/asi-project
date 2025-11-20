from typing import Tuple, Optional, Dict, Any
import time
import json
from pathlib import Path

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

    df_features = df.copy()

    if "driver_number" in df_features.columns and "lap_number" in df_features.columns:
        df_features = df_features.sort_values(["driver_number", "lap_number"])

        for col in ["s1", "s2", "s3", "kph", "top_speed"]:
            if col in df_features.columns:
                df_features[f"{col}_prev"] = df_features.groupby("driver_number")[
                    col
                ].shift(1)
                df_features[f"{col}_avg_3"] = (
                    df_features.groupby("driver_number")[col]
                    .shift(1)
                    .rolling(3, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        if target_column in df_features.columns:
            df_features["lap_time_prev"] = df_features.groupby("driver_number")[
                target_column
            ].shift(1)
            df_features["lap_time_avg_3"] = (
                df_features.groupby("driver_number")[target_column]
                .shift(1)
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    leakage_columns = [
        "s1",
        "s2",
        "s3",
        "s1_improvement",
        "s2_improvement",
        "s3_improvement",
        "lap_time_ms",
        "lap_time_s",
        "lap_time",
        "elapsed_ms",
        "elapsed_s",
        "elapsed",
        "interval_ms",
        "interval",
        "gap",
        "class_interval",
        "class_gap",
        "position",
        "class_position",
    ]

    y = df_features[target_column]

    cols_to_drop = [target_column] + [
        col
        for col in leakage_columns
        if col in df_features.columns and col != target_column
    ]
    X = df_features.drop(columns=cols_to_drop).select_dtypes(include=["number"])

    X = X.dropna()
    y = y[X.index]

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


def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    params: Dict[str, Any],
) -> Any:
    import wandb
    from autogluon.tabular import TabularPredictor

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    train_data = X_train.copy()
    train_data["target"] = y_train

    project_name = params.get("wandb_project", "asi-project")
    wandb.init(project=project_name, job_type="ag-train", config=params, reinit=True)

    time_limit = params.get("time_limit", 600)
    presets = params.get("presets", "best_quality")

    start_time = time.time()
    predictor = TabularPredictor(
        label="target",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets,
    )
    train_time_s = time.time() - start_time

    try:
        leaderboard = predictor.leaderboard(silent=True)
        if leaderboard is not None and len(leaderboard) > 0:
            best_score = leaderboard.iloc[0]["score_val"]
            validation_rmse = float(abs(best_score)) if not pd.isna(best_score) else 0.0
        else:
            validation_rmse = 0.0
    except Exception:
        validation_rmse = 0.0

    wandb.log({"validation_rmse": validation_rmse, "train_time_s": train_time_s})
    wandb.finish()

    return predictor


def evaluate_autogluon(
    ag_predictor: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
) -> Dict[str, float]:
    import wandb
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import numpy as np

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    y_pred = ag_predictor.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }

    try:
        wandb.init(project="asi-project", job_type="ag-evaluate", reinit=True)
        wandb.log(metrics)

        try:
            feature_importance = ag_predictor.feature_importance(X_test)
            if feature_importance is not None and len(feature_importance) > 0:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 6))
                feature_importance.head(20).plot(kind="barh")
                plt.title("AutoGluon Feature Importance (Top 20)")
                plt.xlabel("Importance Score")
                plt.ylabel("Features")
                plt.tight_layout()
                wandb.log({"feature_importance": wandb.Image(plt)})
                plt.close()
        except Exception:
            pass

        wandb.finish()
    except Exception:
        pass

    metrics_path = Path("data/09_tracking/ag_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def save_best_model(predictor: Any) -> str:
    import wandb
    import pickle

    model_path = Path("data/06_models/ag_production.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(predictor, f)

    try:
        wandb.init(project="asi-project", job_type="ag-model-save", reinit=True)

        art = wandb.Artifact(
            "ag_model",
            type="model",
            description="AutoGluon trained model for lap time prediction",
        )
        art.add_file(str(model_path))

        wandb.log_artifact(art, aliases=["candidate", "latest"])

        wandb.finish()
    except Exception:
        pass

    return str(model_path)
