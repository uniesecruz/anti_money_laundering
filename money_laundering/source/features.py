"""Spark feature engineering pipeline for AML (anti-leakage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from source.config import get_spark_session, get_stage_path


SECONDS_BY_WINDOW = {
    "1h": 3600,
    "24h": 86400,
    "7d": 604800,
    "30d": 2592000,
}


@dataclass
class FeatureEngineeringPipelineSpark:
    """Generate AML features using deterministic Spark window operations."""

    timestamp_col: str = "Timestamp"
    account_col: str = "From Account"
    amount_col: str = "Amount Received"
    bank_col: str = "Receiving Currency"
    country_col: str = "From Bank"
    velocity_windows: Dict[str, str] = field(
        default_factory=lambda: {"1h": "1h", "24h": "24h", "7d": "7d"}
    )
    ratio_window: str = "30d"
    smurf_threshold: float = 10000.0

    def fit(self, df: DataFrame) -> "FeatureEngineeringPipelineSpark":
        """No learned state; kept for API compatibility."""
        _ = df
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Apply full feature engineering pipeline in Spark."""
        base = (
            df.withColumn(self.timestamp_col, F.to_timestamp(F.col(self.timestamp_col)))
            .filter(F.col(self.timestamp_col).isNotNull())
            .withColumn("_ts_long", F.col(self.timestamp_col).cast("long"))
        )

        # Deterministic ordering for lag-based features.
        order_cols = [
            F.col(self.timestamp_col).asc(),
            F.col(self.amount_col).asc_nulls_last(),
            F.col("To Account").asc_nulls_last(),
        ]
        row_win = Window.partitionBy(self.account_col).orderBy(*order_cols)

        enriched = self._add_velocity_features(base)
        enriched = self._add_ratio_features(enriched)
        enriched = self._add_behavioral_features(enriched, row_win, order_cols)
        enriched = self._add_smurfing_features(enriched)

        enriched = enriched.drop("_ts_long")
        return enriched

    def fit_transform(self, df: DataFrame) -> DataFrame:
        return self.fit(df).transform(df)

    def _add_velocity_features(self, df: DataFrame) -> DataFrame:
        out = df
        for window_name in self.velocity_windows:
            seconds = SECONDS_BY_WINDOW[window_name]
            win = (
                Window.partitionBy(self.account_col)
                .orderBy(F.col("_ts_long"))
                .rangeBetween(-seconds, -1)
            )
            out = out.withColumn(
                f"txn_count_{window_name}_velocity",
                F.coalesce(F.count(F.lit(1)).over(win), F.lit(0)).cast("double"),
            )
            out = out.withColumn(
                f"amount_sum_{window_name}_velocity",
                F.coalesce(F.sum(F.col(self.amount_col)).over(win), F.lit(0.0)),
            )
            out = out.withColumn(
                f"amount_mean_{window_name}_velocity",
                F.coalesce(F.avg(F.col(self.amount_col)).over(win), F.lit(0.0)),
            )
            out = out.withColumn(
                f"amount_max_{window_name}_velocity",
                F.coalesce(F.max(F.col(self.amount_col)).over(win), F.lit(0.0)),
            )
            if window_name == "7d":
                out = out.withColumn(
                    f"amount_std_{window_name}_velocity",
                    F.coalesce(F.stddev(F.col(self.amount_col)).over(win), F.lit(0.0)),
                )

        return out

    def _add_ratio_features(self, df: DataFrame) -> DataFrame:
        seconds = SECONDS_BY_WINDOW[self.ratio_window]
        win = (
            Window.partitionBy(self.account_col)
            .orderBy(F.col("_ts_long"))
            .rangeBetween(-seconds, -1)
        )

        hist_mean = F.avg(F.col(self.amount_col)).over(win)
        hist_max = F.max(F.col(self.amount_col)).over(win)
        hist_std = F.stddev(F.col(self.amount_col)).over(win)

        out = df.withColumn("_hist_mean", hist_mean)
        out = out.withColumn("_hist_max", hist_max)
        out = out.withColumn("_hist_std", hist_std)

        out = out.withColumn(
            "amount_to_historical_mean_ratio",
            F.when(F.col("_hist_mean") > 0, F.col(self.amount_col) / F.col("_hist_mean")).otherwise(0.0),
        )
        out = out.withColumn(
            "amount_to_historical_max_ratio",
            F.when(F.col("_hist_max") > 0, F.col(self.amount_col) / F.col("_hist_max")).otherwise(0.0),
        )
        out = out.withColumn(
            "amount_zscore_historical",
            F.when(F.col("_hist_std") > 0, (F.col(self.amount_col) - F.col("_hist_mean")) / F.col("_hist_std")).otherwise(0.0),
        )

        return out.drop("_hist_mean", "_hist_max", "_hist_std")

    def _add_behavioral_features(
        self,
        df: DataFrame,
        row_win: Window,
        order_cols: list,
    ) -> DataFrame:
        out = df

        lag_ts = F.lag(F.col("_ts_long")).over(row_win)
        out = out.withColumn(
            "time_since_last_txn_seconds",
            F.coalesce((F.col("_ts_long") - lag_ts).cast("double"), F.lit(0.0)),
        )

        lag_bank = F.lag(F.col(self.bank_col)).over(row_win)
        out = out.withColumn(
            "bank_change_flag",
            F.when(lag_bank.isNull(), F.lit(0)).when(F.col(self.bank_col) != lag_bank, F.lit(1)).otherwise(F.lit(0)),
        )

        # 1 if this is the first time account->country pair appears.
        country_win = Window.partitionBy(self.account_col, self.country_col).orderBy(*order_cols)
        out = out.withColumn(
            "is_new_country",
            F.when(F.row_number().over(country_win) == 1, F.lit(1)).otherwise(F.lit(0)),
        )

        out = out.withColumn("hour_of_day", F.hour(F.col(self.timestamp_col)))
        out = out.withColumn(
            "is_unusual_hour",
            F.when((F.col("hour_of_day") < 6) | (F.col("hour_of_day") > 22), F.lit(1)).otherwise(F.lit(0)),
        )

        return out

    def _add_smurfing_features(self, df: DataFrame) -> DataFrame:
        win24 = (
            Window.partitionBy(self.account_col)
            .orderBy(F.col("_ts_long"))
            .rangeBetween(-SECONDS_BY_WINDOW["24h"], -1)
        )

        is_smurf = (F.col(self.amount_col) >= self.smurf_threshold * 0.8) & (
            F.col(self.amount_col) < self.smurf_threshold
        )
        smurf_amount = F.when(is_smurf, F.col(self.amount_col)).otherwise(F.lit(0.0))
        proximity = F.when(
            F.col(self.amount_col) < self.smurf_threshold,
            (F.lit(self.smurf_threshold) - F.col(self.amount_col)) / F.lit(self.smurf_threshold),
        ).otherwise(F.lit(0.0))

        out = df.withColumn(
            "smurf_txn_count_24h_behavioral",
            F.coalesce(F.sum(F.when(is_smurf, 1).otherwise(0)).over(win24).cast("double"), F.lit(0.0)),
        )
        out = out.withColumn(
            "smurf_amount_sum_24h_behavioral",
            F.coalesce(F.sum(smurf_amount).over(win24), F.lit(0.0)),
        )
        out = out.withColumn(
            "smurf_proximity_score_behavioral",
            F.coalesce(F.avg(proximity).over(win24), F.lit(0.0)),
        )

        return out


def run_feature_engineering_stage(
    spark: SparkSession,
    train_input_path: Path | None = None,
    oot_input_path: Path | None = None,
    train_output_path: Path | None = None,
    oot_output_path: Path | None = None,
) -> Dict[str, Path]:
    """Run stage 05 on train and OOT with same deterministic transform logic."""
    source_train = train_input_path or get_stage_path("train")
    source_oot = oot_input_path or get_stage_path("oot")

    target_train = train_output_path or get_stage_path("train_fe")
    target_oot = oot_output_path or get_stage_path("oot_fe")

    train_df = spark.read.parquet(str(source_train)).repartition("Timestamp")
    oot_df = spark.read.parquet(str(source_oot)).repartition("Timestamp")

    fe = FeatureEngineeringPipelineSpark()
    train_fe = fe.fit_transform(train_df)
    oot_fe = fe.transform(oot_df)

    train_fe.write.mode("overwrite").parquet(str(target_train))
    oot_fe.write.mode("overwrite").parquet(str(target_oot))

    logger.success("Feature engineering completed")
    logger.info("Train FE path: {}", target_train)
    logger.info("OOT FE path: {}", target_oot)

    return {"train_fe": target_train, "oot_fe": target_oot}


def main() -> None:
    spark = get_spark_session(app_name="AML-05-FeatureEngineering")
    try:
        outputs = run_feature_engineering_stage(spark=spark)
        logger.success("Feature engineering outputs: {}", outputs)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
