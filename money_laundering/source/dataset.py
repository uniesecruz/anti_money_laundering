"""PySpark data preparation for AML HI-Medium dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from source.config import EXTERNAL_DATA_DIR, get_spark_session, get_stage_path

TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def _read_csv(spark: SparkSession, path: Path) -> DataFrame:
    """Read CSV with inferred schema and header."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .csv(str(path))
    )


def load_hi_large_sources(
    spark: SparkSession,
    dataset_prefix: str = "HI-Medium",
) -> Dict[str, DataFrame]:
    """Load account and transaction sources from external zone."""
    accounts_path = EXTERNAL_DATA_DIR / f"{dataset_prefix}_accounts.csv"
    trans_path = EXTERNAL_DATA_DIR / f"{dataset_prefix}_Trans.csv"

    if not accounts_path.exists():
        raise FileNotFoundError(f"Missing account file: {accounts_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Missing transaction file: {trans_path}")

    logger.info("Reading accounts from {}", accounts_path)
    accounts_df = _read_csv(spark, accounts_path)

    logger.info("Reading transactions from {}", trans_path)
    trans_df = _read_csv(spark, trans_path)

    if len(trans_df.columns) == len(TRANSACTION_COLUMNS):
        trans_df = trans_df.toDF(*TRANSACTION_COLUMNS)

    trans_df = trans_df.withColumn(
        "Timestamp",
        F.coalesce(
            F.to_timestamp(F.col("Timestamp"), "yyyy/MM/dd HH:mm"),
            F.to_timestamp(F.col("Timestamp"), "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(F.col("Timestamp")),
        ),
    )

    return {"accounts": accounts_df, "transactions": trans_df}


def enrich_transactions_with_accounts(
    trans_df: DataFrame,
    accounts_df: DataFrame,
) -> DataFrame:
    """Join transactions with origin and destination account metadata."""
    from_acc = (
        accounts_df.select(
            F.col("Bank ID").alias("from_bank_id"),
            F.col("Account Number").alias("from_account_number"),
            F.col("Bank Name").alias("From Bank Name"),
            F.col("Entity ID").alias("From Entity ID"),
            F.col("Entity Name").alias("From Entity Name"),
        )
        .dropDuplicates(["from_bank_id", "from_account_number"])
    )

    to_acc = (
        accounts_df.select(
            F.col("Bank ID").alias("to_bank_id"),
            F.col("Account Number").alias("to_account_number"),
            F.col("Bank Name").alias("To Bank Name"),
            F.col("Entity ID").alias("To Entity ID"),
            F.col("Entity Name").alias("To Entity Name"),
        )
        .dropDuplicates(["to_bank_id", "to_account_number"])
    )

    enriched = (
        trans_df.alias("t")
        .join(
            from_acc.alias("f"),
            (F.col("t.From Bank") == F.col("f.from_bank_id"))
            & (F.col("t.From Account") == F.col("f.from_account_number")),
            "left",
        )
        .join(
            to_acc.alias("d"),
            (F.col("t.To Bank") == F.col("d.to_bank_id"))
            & (F.col("t.To Account") == F.col("d.to_account_number")),
            "left",
        )
        .select(
            "t.*",
            "From Bank Name",
            "From Entity ID",
            "From Entity Name",
            "To Bank Name",
            "To Entity ID",
            "To Entity Name",
        )
    )

    return enriched


def save_enriched_dataset(df: DataFrame, output_path: Path | None = None) -> Path:
    """Persist enriched full dataset in Parquet format."""
    target_path = output_path or get_stage_path("dataprep")
    logger.info("Writing enriched dataset to {}", target_path)
    df.write.mode("overwrite").parquet(str(target_path))
    return target_path


def run_dataprep(
    spark: SparkSession,
    dataset_prefix: str = "HI-Medium",
    output_path: Path | None = None,
) -> Path:
    """Execute full Spark dataprep stage."""
    sources = load_hi_large_sources(spark=spark, dataset_prefix=dataset_prefix)
    enriched = enrich_transactions_with_accounts(
        trans_df=sources["transactions"],
        accounts_df=sources["accounts"],
    )

    logger.info("Dataprep rows: {}", enriched.count())
    return save_enriched_dataset(enriched, output_path=output_path)


def main() -> None:
    """CLI entrypoint for stage 01 dataprep."""
    spark = get_spark_session(app_name="AML-01-Dataprep")
    try:
        output = run_dataprep(spark=spark)
        logger.success("Dataprep completed. Output: {}", output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
