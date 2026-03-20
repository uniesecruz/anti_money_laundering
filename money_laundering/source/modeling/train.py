"""Spark ML model training for AML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from loguru import logger
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from source.config import ARTIFACTS_DIR, get_model_path, get_spark_session, get_stage_path


def _compute_confusion_metrics(pred_df: DataFrame, prediction_col: str = "prediction") -> Dict[str, float]:
    metrics = pred_df.select(
        F.sum(F.when((F.col("label") == 1.0) & (F.col(prediction_col) == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0.0) & (F.col(prediction_col) == 1.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1.0) & (F.col(prediction_col) == 0.0), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((F.col("label") == 0.0) & (F.col(prediction_col) == 0.0), 1).otherwise(0)).alias("tn"),
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


def _build_scored_df(pred_df: DataFrame) -> DataFrame:
    return pred_df.withColumn("score", vector_to_array(F.col("probability")).getItem(1))


def _evaluate_at_threshold(scored_df: DataFrame, threshold: float) -> Dict[str, float]:
    thresholded = scored_df.withColumn(
        "prediction_threshold",
        F.when(F.col("score") >= F.lit(float(threshold)), F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    metrics = _compute_confusion_metrics(thresholded, prediction_col="prediction_threshold")
    metrics["threshold"] = float(threshold)
    return metrics


def _candidate_thresholds() -> Iterable[float]:
    # Balanced grid to keep runtime bounded on large OOT sets.
    base = [x / 100.0 for x in range(5, 100, 5)]
    extra = [0.005, 0.01, 0.02, 0.03, 0.04]
    values = sorted(set(extra + base))
    return values


def _optimize_threshold_for_f1(scored_df: DataFrame) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _evaluate_at_threshold(scored_df, threshold=0.5)

    for threshold in _candidate_thresholds():
        current = _evaluate_at_threshold(scored_df, threshold=threshold)
        if current["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = current

    return best_threshold, best_metrics


def _score_and_evaluate(pred_df: DataFrame) -> Dict[str, object]:
    scored = _build_scored_df(pred_df).persist(StorageLevel.DISK_ONLY)

    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    try:
        default_metrics = _evaluate_at_threshold(scored, threshold=0.5)
        best_threshold, optimized_metrics = _optimize_threshold_for_f1(scored)

        area_under_pr = float(evaluator_pr.evaluate(scored))
        area_under_roc = float(evaluator_roc.evaluate(scored))

        default_metrics["areaUnderPR"] = area_under_pr
        default_metrics["areaUnderROC"] = area_under_roc
        optimized_metrics["areaUnderPR"] = area_under_pr
        optimized_metrics["areaUnderROC"] = area_under_roc

        metrics = {
            "threshold_default_0_5": default_metrics,
            "threshold_optimized": optimized_metrics,
            "best_threshold": float(best_threshold),
        }

        return metrics
    finally:
        scored.unpersist()


def train_models(
    train_df: DataFrame,
    oot_df: DataFrame,
    algorithm: str = "rf",
) -> Dict[str, object]:
    """Train a Spark ML classifier on full train set and evaluate in same OOT."""
    if algorithm == "gbt":
        estimator = GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=3,
            maxDepth=2,
            maxBins=16,
            stepSize=0.1,
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

    logger.info("Training model: {} (using full training dataset, no internal split)", model_name)
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
    logger.info("Metrics (default threshold): {}", metrics["threshold_default_0_5"])
    logger.info("Metrics (optimized threshold): {}", metrics["threshold_optimized"])
    logger.info("Best threshold found: {}", metrics["best_threshold"])
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
