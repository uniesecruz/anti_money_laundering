"""Notebook 06 converted to script: Spark ML preprocessing pipeline."""

from loguru import logger

from source.config import get_stage_path, get_spark_session
from source.preprocessing import run_preprocessing_stage


def main() -> None:
    spark = get_spark_session(app_name="AML-Notebook-06-Preprocessing")
    try:
        outputs = run_preprocessing_stage(
            spark=spark,
            train_input_path=get_stage_path("train_fe"),
            oot_input_path=get_stage_path("oot_fe"),
            train_output_path=get_stage_path("train_vector"),
            oot_output_path=get_stage_path("oot_vector"),
        )
        logger.success("Notebook 06 complete. Train vector: {}", outputs["train_vector"])
        logger.success("Notebook 06 complete. OOT vector: {}", outputs["oot_vector"])
        logger.success("Notebook 06 complete. Pipeline model: {}", outputs["pipeline_model"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
