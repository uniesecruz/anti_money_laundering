"""Notebook 03 converted to script: Spark data quality integrity checks."""

from loguru import logger

from source.config import get_stage_path, get_spark_session
from source.data_quality import run_data_quality_stage


def main() -> None:
    spark = get_spark_session(app_name="AML-Notebook-03-DataQuality")
    try:
        outputs = run_data_quality_stage(
            spark=spark,
            input_path=get_stage_path("dataprep"),
            output_data_path=get_stage_path("data_quality"),
        )
        logger.success("Notebook 03 complete. Output data: {}", outputs["data"])
        logger.success("Notebook 03 complete. Report: {}", outputs["report"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
