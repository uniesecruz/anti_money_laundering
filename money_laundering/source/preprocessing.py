"""PySpark ML preprocessing pipeline for AML."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
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
    StringType,
)

from source.config import get_model_path, get_spark_session, get_stage_path


NUMERIC_TYPES = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)

DEFAULT_DROP_COLS = [
    "Timestamp",
    "From Bank",
    "To Bank",
    "From Account",
    "To Account",
    "From Entity ID",
    "To Entity ID",
]


class AMLPreprocessorSpark:
    """Build and apply Spark preprocessing with anti-leakage fit/transform."""

    def __init__(self, label_col: str = "Is Laundering", drop_cols: List[str] | None = None):
        self.label_col = label_col
        self.drop_cols = drop_cols or DEFAULT_DROP_COLS
        self.pipeline_: Pipeline | None = None
        self.model_: PipelineModel | None = None
        self.feature_columns_: List[str] = []

    def _prepare_base(self, df: DataFrame) -> DataFrame:
        available_drop = [c for c in self.drop_cols if c in df.columns]
        return df.drop(*available_drop)

    def _infer_column_groups(self, df: DataFrame) -> Tuple[List[str], List[str]]:
        categorical_cols: List[str] = []
        numeric_cols: List[str] = []

        for field in df.schema.fields:
            if field.name == self.label_col:
                continue
            if isinstance(field.dataType, StringType):
                categorical_cols.append(field.name)
            elif isinstance(field.dataType, NUMERIC_TYPES):
                numeric_cols.append(field.name)

        return categorical_cols, numeric_cols

    def _split_categorical_by_cardinality(
        self,
        df: DataFrame,
        categorical_cols: List[str],
        max_onehot_cardinality: int = 30,
    ) -> Tuple[List[str], List[str]]:
        onehot_cols: List[str] = []
        indexed_only_cols: List[str] = []

        if not categorical_cols:
            return onehot_cols, indexed_only_cols

        agg_exprs = [F.approx_count_distinct(F.col(c)).alias(c) for c in categorical_cols]
        cardinality_row = df.agg(*agg_exprs).first()

        for col_name in categorical_cols:
            card = int(cardinality_row[col_name] or 0)
            if card <= max_onehot_cardinality:
                onehot_cols.append(col_name)
            else:
                indexed_only_cols.append(col_name)

        return onehot_cols, indexed_only_cols

    def fit(self, train_df: DataFrame) -> "AMLPreprocessorSpark":
        base = self._prepare_base(train_df)
        categorical_cols, numeric_cols = self._infer_column_groups(base)
        onehot_cols, indexed_only_cols = self._split_categorical_by_cardinality(
            base, categorical_cols
        )

        logger.info("Categorical cols: {}", len(categorical_cols))
        logger.info("OneHot cols (low-cardinality): {}", len(onehot_cols))
        logger.info("Indexed-only cols (high-cardinality): {}", len(indexed_only_cols))
        logger.info("Numeric cols: {}", len(numeric_cols))

        indexer_outputs = [f"{c}__idx" for c in categorical_cols]
        ohe_outputs = [f"{c}__ohe" for c in onehot_cols]
        imputed_numeric = [f"{c}__imp" for c in numeric_cols]
        indexed_only_outputs = [f"{c}__idx" for c in indexed_only_cols]

        stages = []

        if categorical_cols:
            stages.extend(
                [
                    StringIndexer(
                        inputCols=categorical_cols,
                        outputCols=indexer_outputs,
                        handleInvalid="keep",
                    ),
                ]
            )

        if onehot_cols:
            onehot_input_cols = [f"{c}__idx" for c in onehot_cols]
            stages.append(
                OneHotEncoder(
                    inputCols=onehot_input_cols,
                    outputCols=ohe_outputs,
                    handleInvalid="keep",
                )
            )

        if numeric_cols:
            stages.append(
                Imputer(
                    inputCols=numeric_cols,
                    outputCols=imputed_numeric,
                    strategy="median",
                )
            )

        assembled_inputs = imputed_numeric + indexed_only_outputs + ohe_outputs
        self.feature_columns_ = assembled_inputs

        stages.extend(
            [
                VectorAssembler(
                    inputCols=assembled_inputs,
                    outputCol="features_raw",
                    handleInvalid="keep",
                ),
                StandardScaler(
                    inputCol="features_raw",
                    outputCol="features",
                    withStd=True,
                    withMean=False,
                ),
            ]
        )

        self.pipeline_ = Pipeline(stages=stages)
        self.model_ = self.pipeline_.fit(base)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        if self.model_ is None:
            raise ValueError("Preprocessor must be fitted before transform.")

        base = self._prepare_base(df)
        transformed = self.model_.transform(base)

        output = transformed.select(F.col(self.label_col).cast("double").alias("label"), "features")
        return output

    def fit_transform(self, train_df: DataFrame) -> DataFrame:
        return self.fit(train_df).transform(train_df)

    def save(self, model_path: Path) -> None:
        if self.model_ is None:
            raise ValueError("No fitted model to save.")
        self.model_.write().overwrite().save(str(model_path))


def run_preprocessing_stage(
    spark: SparkSession,
    train_input_path: Path | None = None,
    oot_input_path: Path | None = None,
    train_output_path: Path | None = None,
    oot_output_path: Path | None = None,
    pipeline_output_path: Path | None = None,
) -> Dict[str, Path]:
    """Run stage 06 preprocessing pipeline in Spark ML."""
    source_train = train_input_path or get_stage_path("train_fe")
    source_oot = oot_input_path or get_stage_path("oot_fe")

    target_train = train_output_path or get_stage_path("train_vector")
    target_oot = oot_output_path or get_stage_path("oot_vector")
    target_pipeline = pipeline_output_path or get_model_path("preprocessing_pipeline_spark")

    df_train = spark.read.parquet(str(source_train)).repartition("Timestamp")
    df_oot = spark.read.parquet(str(source_oot)).repartition("Timestamp")

    preprocessor = AMLPreprocessorSpark(label_col="Is Laundering")
    X_train = preprocessor.fit_transform(df_train)
    X_oot = preprocessor.transform(df_oot)

    X_train.write.mode("overwrite").parquet(str(target_train))
    X_oot.write.mode("overwrite").parquet(str(target_oot))
    preprocessor.save(target_pipeline)

    logger.success("Preprocessing completed")
    logger.info("Train vector path: {}", target_train)
    logger.info("OOT vector path: {}", target_oot)
    logger.info("Pipeline model path: {}", target_pipeline)

    return {
        "train_vector": target_train,
        "oot_vector": target_oot,
        "pipeline_model": target_pipeline,
    }


def main() -> None:
    spark = get_spark_session(app_name="AML-06-Preprocessing")
    try:
        outputs = run_preprocessing_stage(spark=spark)
        logger.success("Preprocessing outputs: {}", outputs)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
