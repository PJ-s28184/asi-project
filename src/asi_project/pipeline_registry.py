
from __future__ import annotations

from kedro.pipeline import Pipeline
from asi_project.pipelines.data_science import create_pipeline as create_ds_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    ds = create_ds_pipeline()
    return {
        "data_science": ds,
        "__default__": ds,
    }
