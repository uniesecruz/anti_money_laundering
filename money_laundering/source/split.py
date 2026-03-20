"""Deterministic temporal split in PySpark for train/OOT."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from source.config import get_spark_session, get_stage_path


def _ordered_with_row_number(df: DataFrame) -> DataFrame:
    order_cols = [
        F.col("Timestamp").asc(),
        F.col("From Account").asc_nulls_last(),
        F.col("To Account").asc_nulls_last(),
        F.col("Amount Received").asc_nulls_last(),
    ]
    win = Window.orderBy(*order_cols)
    return df.withColumn("_row_num", F.row_number().over(win))


def _compute_cut_timestamp(df: DataFrame, split_ratio: float) -> Tuple[int, object]:
    total_rows = df.count()
    split_index = max(int(total_rows * split_ratio), 1)
    cut_row = (
        df.filter(F.col("_row_num") == split_index)
        .select("Timestamp")
        .limit(1)
        .collect()
    )
    if not cut_row:
        raise ValueError("Unable to compute split cut timestamp from dataset.")
    return split_index, cut_row[0]["Timestamp"]


def split_train_oot_with_gap(
    df: DataFrame,
    split_ratio: float = 0.8,
    gap_days: int = 7,
) -> Dict[str, DataFrame]:
    """Split by deterministic temporal boundary with anti-leakage gap."""
    base = df.filter(F.col("Timestamp").isNotNull())
    ordered = _ordered_with_row_number(base)

    split_index, cut_ts = _compute_cut_timestamp(ordered, split_ratio=split_ratio)
    cut_ts_plus_gap = F.lit(cut_ts) + F.expr(f"INTERVAL {int(gap_days)} DAYS")

    logger.info("Split index: {}", split_index)
    logger.info("Train end timestamp: {}", cut_ts)
    logger.info("Gap days: {}", gap_days)

    train_df = ordered.filter(F.col("Timestamp") < F.lit(cut_ts)).drop("_row_num")
    gap_df = (
        ordered.filter((F.col("Timestamp") >= F.lit(cut_ts)) & (F.col("Timestamp") < cut_ts_plus_gap))
        .drop("_row_num")
    )
    oot_df = ordered.filter(F.col("Timestamp") >= cut_ts_plus_gap).drop("_row_num")

    return {"train": train_df, "gap": gap_df, "oot": oot_df}


def save_split_outputs(
    split_data: Dict[str, DataFrame],
    train_path: Path | None = None,
    gap_path: Path | None = None,
    oot_path: Path | None = None,
) -> Dict[str, Path]:
    """Persist split outputs in Parquet format."""
    final_train = train_path or get_stage_path("train")
    final_gap = gap_path or get_stage_path("gap")
    final_oot = oot_path or get_stage_path("oot")

    split_data["train"].write.mode("overwrite").parquet(str(final_train))
    split_data["gap"].write.mode("overwrite").parquet(str(final_gap))
    split_data["oot"].write.mode("overwrite").parquet(str(final_oot))

    return {"train": final_train, "gap": final_gap, "oot": final_oot}


def run_split_stage(
    spark: SparkSession,
    input_path: Path | None = None,
    split_ratio: float = 0.8,
    gap_days: int = 7,
) -> Dict[str, Path]:
    """Run stage 04 split using quality-approved dataset."""
    source_path = input_path or get_stage_path("data_quality")
    logger.info("Reading quality dataset from {}", source_path)

    df = spark.read.parquet(str(source_path)).repartition("Timestamp")
    split_data = split_train_oot_with_gap(df=df, split_ratio=split_ratio, gap_days=gap_days)

    logger.info("Train rows: {}", split_data["train"].count())
    logger.info("Gap rows: {}", split_data["gap"].count())
    logger.info("OOT rows: {}", split_data["oot"].count())

    return save_split_outputs(split_data)


def main() -> None:
    spark = get_spark_session(app_name="AML-04-Split")
    try:
        outputs = run_split_stage(spark=spark)
        logger.success("Split output train: {}", outputs["train"])
        logger.success("Split output gap: {}", outputs["gap"])
        logger.success("Split output oot: {}", outputs["oot"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
