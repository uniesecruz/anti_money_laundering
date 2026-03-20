"""Notebook 04 converted to script: deterministic temporal split in Spark."""

from loguru import logger

from source.config import get_stage_path, get_spark_session
from source.split import run_split_stage


def main() -> None:
    spark = get_spark_session(app_name="AML-Notebook-04-Split")
    try:
        outputs = run_split_stage(
            spark=spark,
            input_path=get_stage_path("data_quality"),
            split_ratio=0.8,
            gap_days=7,
        )
        logger.success("Notebook 04 complete. Train: {}", outputs["train"])
        logger.success("Notebook 04 complete. Gap: {}", outputs["gap"])
        logger.success("Notebook 04 complete. OOT: {}", outputs["oot"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
