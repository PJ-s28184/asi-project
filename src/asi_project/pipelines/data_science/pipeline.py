from kedro.pipeline import Pipeline, node

from .nodes import (
    clean,
    evaluate,
    split,
    train_baseline,
    train_autogluon,
    evaluate_autogluon,
    save_best_model,
    # select_production_model,
)


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
            node(
                func=train_autogluon,
                inputs=["X_train", "y_train", "params:data_science.autogluon"],
                outputs="ag_predictor",
                name="train_autogluon_node",
            ),
            node(
                func=evaluate_autogluon,
                inputs=["ag_predictor", "X_test", "y_test"],
                outputs="ag_metrics",
                name="evaluate_autogluon_node",
            ),
            node(
                func=save_best_model,
                inputs="ag_predictor",
                outputs="ag_model",
                name="save_best_model_node",
            ),
            # node(
            #     func=select_production_model,
            #     inputs="params:best_alias",
            #     outputs="production_model_name",
            #     name="select_production_model_node",
            # ),
        ]
    )
