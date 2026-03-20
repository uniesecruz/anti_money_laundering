"""Notebook 05 converted to script: Spark window-based feature engineering."""

from loguru import logger

from source.config import get_stage_path, get_spark_session
from source.features import run_feature_engineering_stage


def main() -> None:
    spark = get_spark_session(app_name="AML-Notebook-05-FeatureEngineering")
    try:
        outputs = run_feature_engineering_stage(
            spark=spark,
            train_input_path=get_stage_path("train"),
            oot_input_path=get_stage_path("oot"),
            train_output_path=get_stage_path("train_fe"),
            oot_output_path=get_stage_path("oot_fe"),
        )
        logger.success("Notebook 05 complete. Train FE: {}", outputs["train_fe"])
        logger.success("Notebook 05 complete. OOT FE: {}", outputs["oot_fe"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
