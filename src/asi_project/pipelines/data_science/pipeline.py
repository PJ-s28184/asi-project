"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import dummy


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=dummy,
                inputs=None,
                outputs="dummy_output",
                name="dummy_node",
            )
        ]
    )
