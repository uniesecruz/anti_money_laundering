"""Notebook 08 converted to script: Spark ML training and OOT scoring."""

from loguru import logger

from source.config import get_stage_path, get_spark_session
from source.modeling.train import run_training_stage


def main() -> None:
    spark = get_spark_session(app_name="AML-Notebook-08-Train")
    try:
        outputs = run_training_stage(
            spark=spark,
            train_input_path=get_stage_path("train_vector"),
            oot_input_path=get_stage_path("oot_vector"),
            predictions_output_path=get_stage_path("predictions_oot"),
            algorithm="rf",
        )
        logger.success("Notebook 08 complete. Model: {}", outputs["model_path"])
        logger.success("Notebook 08 complete. Predictions: {}", outputs["predictions_path"])
        logger.success("Notebook 08 complete. Metrics: {}", outputs["metrics_path"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
