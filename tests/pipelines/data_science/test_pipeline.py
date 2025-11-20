import pandas as pd

from asi_project.pipelines.data_science.nodes import clean, split


def test_clean_drops_nans_and_dupes():
    df = pd.DataFrame(
        {
            "a": [1, 2, None, 2],
            "b": [3, None, 5, 3],
            "lap_time_seconds": [100.0, 110.0, 120.0, 100.0],
        }
    )
    cleaned = clean(df)

    assert len(cleaned) == 2
    assert len(cleaned) < len(df)
    assert cleaned.isnull().sum().sum() == 0


def test_split_proportions_and_no_leakage():
    n = 100
    df = pd.DataFrame(
        {
            "f1": range(n),
            "f2": range(n, 2 * n),
            "lap_time_seconds": [100.0 + i % 2 for i in range(n)],
        }
    )

    X_train, X_test, y_train, y_test = split(
        df, target_column="lap_time_seconds", test_size=0.2, random_state=42
    )

    assert abs(len(X_test) - int(0.2 * n)) <= 1
    assert len(X_train) + len(X_test) == n
    assert len(y_train) + len(y_test) == n

    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(X_train.index).union(set(X_test.index)) == set(df.index)
