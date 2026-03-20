"""Spark ML model training for AML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from loguru import logger
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from source.config import ARTIFACTS_DIR, get_model_path, get_spark_session, get_stage_path


def _compute_confusion_metrics(pred_df: DataFrame) -> Dict[str, float]:
    metrics = pred_df.select(
        F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("tn"),
    ).first()

    tp = float(metrics["tp"] or 0.0)
    fp = float(metrics["fp"] or 0.0)
    fn = float(metrics["fn"] or 0.0)
    tn = float(metrics["tn"] or 0.0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _score_and_evaluate(pred_df: DataFrame) -> Dict[str, float]:
    scored = pred_df.withColumn("score", vector_to_array(F.col("probability")).getItem(1))

    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    metrics = _compute_confusion_metrics(scored)
    metrics["areaUnderPR"] = float(evaluator_pr.evaluate(scored))
    metrics["areaUnderROC"] = float(evaluator_roc.evaluate(scored))

    return metrics


def train_models(
    train_df: DataFrame,
    oot_df: DataFrame,
    algorithm: str = "rf",
) -> Dict[str, object]:
    """Train a Spark ML classifier and evaluate in OOT."""
    if algorithm == "gbt":
        estimator = GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=80,
            maxDepth=6,
            stepSize=0.05,
            seed=42,
        )
        model_name = "GBTClassifier"
    else:
        estimator = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=60,
            maxDepth=8,
            maxBins=64,
            seed=42,
            featureSubsetStrategy="sqrt",
        )
        model_name = "RandomForestClassifier"

    logger.info("Training model: {}", model_name)
    model = estimator.fit(train_df)

    logger.info("Scoring OOT dataset")
    pred_oot = model.transform(oot_df)
    metrics = _score_and_evaluate(pred_oot)

    return {"model_name": model_name, "model": model, "pred_oot": pred_oot, "metrics": metrics}


def run_training_stage(
    spark: SparkSession,
    train_input_path: Path | None = None,
    oot_input_path: Path | None = None,
    predictions_output_path: Path | None = None,
    model_output_path: Path | None = None,
    algorithm: str = "rf",
) -> Dict[str, object]:
    """Run stage 08 training pipeline with Spark ML."""
    source_train = train_input_path or get_stage_path("train_vector")
    source_oot = oot_input_path or get_stage_path("oot_vector")

    target_predictions = predictions_output_path or get_stage_path("predictions_oot")
    target_model = model_output_path or get_model_path("aml_classifier_spark")

    train_df = spark.read.parquet(str(source_train))
    oot_df = spark.read.parquet(str(source_oot))

    result = train_models(train_df=train_df, oot_df=oot_df, algorithm=algorithm)

    result["pred_oot"].write.mode("overwrite").parquet(str(target_predictions))
    result["model"].write().overwrite().save(str(target_model))

    metrics = result["metrics"]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ARTIFACTS_DIR / "08_training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.success("Training completed with {}", result["model_name"])
    logger.info("Metrics: {}", metrics)
    logger.info("Predictions path: {}", target_predictions)
    logger.info("Model path: {}", target_model)

    return {
        "model_name": result["model_name"],
        "metrics": metrics,
        "predictions_path": target_predictions,
        "model_path": target_model,
        "metrics_path": metrics_path,
    }


def main() -> None:
    spark = get_spark_session(app_name="AML-08-Train")
    try:
        outputs = run_training_stage(spark=spark, algorithm="rf")
        logger.success("Training outputs: {}", outputs)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
