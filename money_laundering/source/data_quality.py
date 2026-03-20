"""Data quality checks using native PySpark functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    TimestampType,
)

from source.config import ARTIFACTS_DIR, get_spark_session, get_stage_path


NUMERIC_TYPES = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)


def _null_profile(df: DataFrame) -> Dict[str, int]:
    row = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0]
    return {c: int(row[c] or 0) for c in df.columns}


def _type_profile(df: DataFrame) -> Dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in df.schema.fields}


def _duplicate_count(df: DataFrame) -> int:
    total = df.count()
    unique = df.distinct().count()
    return total - unique


def _timestamp_and_numeric_checks(df: DataFrame) -> Dict[str, Dict[str, float]]:
    checks: Dict[str, Dict[str, float]] = {}

    if "Timestamp" in df.columns:
        checks["Timestamp"] = {
            "nulls": float(df.filter(F.col("Timestamp").isNull()).count()),
            "min": str(df.agg(F.min("Timestamp")).first()[0]),
            "max": str(df.agg(F.max("Timestamp")).first()[0]),
        }

    for field in df.schema.fields:
        if isinstance(field.dataType, NUMERIC_TYPES):
            col_name = field.name
            stats = df.agg(
                F.min(F.col(col_name)).alias("min"),
                F.max(F.col(col_name)).alias("max"),
                F.avg(F.col(col_name)).alias("avg"),
            ).first()
            checks[col_name] = {
                "min": float(stats["min"]) if stats["min"] is not None else 0.0,
                "max": float(stats["max"]) if stats["max"] is not None else 0.0,
                "avg": float(stats["avg"]) if stats["avg"] is not None else 0.0,
            }

    return checks


def run_data_quality_checks(df: DataFrame) -> Dict[str, object]:
    """Compute data integrity metrics: nulls, duplicates and schema types."""
    summary = {
        "row_count": df.count(),
        "column_count": len(df.columns),
        "duplicate_rows": _duplicate_count(df),
        "null_profile": _null_profile(df),
        "type_profile": _type_profile(df),
        "checks": _timestamp_and_numeric_checks(df),
    }
    return summary


def save_data_quality_outputs(
    df: DataFrame,
    summary: Dict[str, object],
    output_data_path: Path | None = None,
    output_report_path: Path | None = None,
) -> Dict[str, Path]:
    """Write quality-approved dataframe and report artifacts."""
    data_path = output_data_path or get_stage_path("data_quality")
    report_path = output_report_path or (ARTIFACTS_DIR / "03_data_quality_report.json")

    logger.info("Writing data quality parquet to {}", data_path)
    df.write.mode("overwrite").parquet(str(data_path))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"data": data_path, "report": report_path}


def run_data_quality_stage(
    spark: SparkSession,
    input_path: Path | None = None,
    output_data_path: Path | None = None,
) -> Dict[str, Path]:
    """Run stage 03 quality checks and persist outputs."""
    source_path = input_path or get_stage_path("dataprep")
    logger.info("Reading dataprep data from {}", source_path)

    df = spark.read.parquet(str(source_path)).repartition("Timestamp")

    # Basic deterministic cleanup before checks.
    if "Timestamp" in df.columns:
        df = df.filter(F.col("Timestamp").isNotNull())

    summary = run_data_quality_checks(df)
    outputs = save_data_quality_outputs(df=df, summary=summary, output_data_path=output_data_path)

    logger.success("Data quality completed with {} rows", summary["row_count"])
    return outputs


def main() -> None:
    spark = get_spark_session(app_name="AML-03-DataQuality")
    try:
        outputs = run_data_quality_stage(spark=spark)
        logger.success("Data quality output data: {}", outputs["data"])
        logger.success("Data quality report: {}", outputs["report"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
