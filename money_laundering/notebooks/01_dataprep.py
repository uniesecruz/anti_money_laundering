"""Notebook 01 converted to script: distributed dataprep with Spark."""

from loguru import logger

from source.config import get_stage_path, get_spark_session
from source.dataset import run_dataprep


def main() -> None:
    spark = get_spark_session(app_name="AML-Notebook-01-Dataprep")
    try:
        output_path = run_dataprep(spark=spark, dataset_prefix="HI-Medium", output_path=get_stage_path("dataprep"))
        logger.success("Notebook 01 complete. Output: {}", output_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
