"""
Proste spięcie pipeline'u data_science.
"""

from kedro.pipeline import Pipeline, node

from .nodes import clean, evaluate, split, train_baseline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean,
                inputs="raw_data",
                outputs="clean_data",
                name="clean_node",
            ),
            node(
                func=split,
                inputs=[
                    "clean_data",
                    "params:data_science.target_column",
                    "params:data_science.test_size",
                    "params:data_science.random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_node",
            ),
            node(
                func=train_baseline,
                inputs=["X_train", "y_train"],
                outputs=["model_baseline", "model_file_path"],
                name="train_baseline_node",
            ),
            node(
                func=evaluate,
                inputs=["model_baseline", "X_test", "y_test"],
                outputs="metrics_baseline",
                name="evaluate_node",
            ),
        ]
    )
