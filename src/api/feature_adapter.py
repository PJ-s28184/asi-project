import pandas as pd

MODEL_FEATURES = [
    'Unnamed: 0',
    'number',
    'driver_number',
    'lap_number',
    'lap_improvement',
    'kph',
    'top_speed',
    'season',
    'round',
    'driver_stint_no',
    'team_stint_no',
    'pit_time_seconds',
    's1_prev',
    's1_avg_3',
    's2_prev',
    's2_avg_3',
    's3_prev',
    's3_avg_3',
    'kph_prev',
    'kph_avg_3',
    'top_speed_prev',
    'top_speed_avg_3',
    'lap_time_prev',
    'lap_time_avg_3',
]

DEFAULTS = {
    "Unnamed: 0": 0,
    "lap_improvement": 0,
    "driver_stint_no": 1,
    "team_stint_no": 1,
    "pit_time_seconds": 0,
    "s1_prev": 0, "s1_avg_3": 0,
    "s2_prev": 0, "s2_avg_3": 0,
    "s3_prev": 0, "s3_avg_3": 0,
    "kph_prev": 0, "kph_avg_3": 0,
    "top_speed_prev": 0, "top_speed_avg_3": 0,
    "lap_time_prev": 0, "lap_time_avg_3": 0,
}

def adapt_features(payload: dict) -> pd.DataFrame:
    row = {col: DEFAULTS.get(col, payload.get(col, 0)) for col in MODEL_FEATURES}
    return pd.DataFrame([row])