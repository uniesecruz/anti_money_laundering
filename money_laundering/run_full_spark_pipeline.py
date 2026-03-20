"""Run the full 100% PySpark AML pipeline end-to-end."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List

from loguru import logger

from source.config import get_spark_session
from source.data_quality import run_data_quality_stage
from source.dataset import run_dataprep
from source.features import run_feature_engineering_stage
from source.modeling.train import run_training_stage
from source.preprocessing import run_preprocessing_stage
from source.split import run_split_stage


StageRunner = Callable[..., object]


def _build_stage_map() -> Dict[str, StageRunner]:
    return {
        "dataprep": run_dataprep,
        "data_quality": run_data_quality_stage,
        "split": run_split_stage,
        "feature_engineering": run_feature_engineering_stage,
        "preprocessing": run_preprocessing_stage,
        "train": run_training_stage,
    }


def _slice_stages(stage_names: List[str], start_stage: str, end_stage: str) -> List[str]:
    start_idx = stage_names.index(start_stage)
    end_idx = stage_names.index(end_stage)
    if start_idx > end_idx:
        raise ValueError("start-stage must come before end-stage in pipeline order.")
    return stage_names[start_idx : end_idx + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AML Spark pipeline")
    parser.add_argument(
        "--start-stage",
        default="dataprep",
        choices=[
            "dataprep",
            "data_quality",
            "split",
            "feature_engineering",
            "preprocessing",
            "train",
        ],
        help="Stage to start from (inclusive)",
    )
    parser.add_argument(
        "--end-stage",
        default="train",
        choices=[
            "dataprep",
            "data_quality",
            "split",
            "feature_engineering",
            "preprocessing",
            "train",
        ],
        help="Stage to stop at (inclusive)",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="HI-Medium",
        help="Dataset prefix for stage dataprep (default: HI-Medium)",
    )
    parser.add_argument(
        "--algorithm",
        default="rf",
        choices=["rf", "gbt"],
        help="Training algorithm for stage train",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train ratio for deterministic temporal split",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=7,
        help="Gap days between train and OOT",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    stage_map = _build_stage_map()
    stage_names = list(stage_map.keys())
    selected_stages = _slice_stages(stage_names, args.start_stage, args.end_stage)

    spark = get_spark_session(app_name="AML-FullSpark-Orchestrator")

    try:
        for stage in selected_stages:
            logger.info("=" * 80)
            logger.info("Running stage: {}", stage)
            logger.info("=" * 80)

            if stage == "dataprep":
                output = stage_map[stage](spark=spark, dataset_prefix=args.dataset_prefix)
            elif stage == "split":
                output = stage_map[stage](
                    spark=spark,
                    split_ratio=args.split_ratio,
                    gap_days=args.gap_days,
                )
            elif stage == "train":
                output = stage_map[stage](spark=spark, algorithm=args.algorithm)
            else:
                output = stage_map[stage](spark=spark)

            logger.success("Stage '{}' finished. Output: {}", stage, output)

        logger.success("Full pipeline execution completed successfully.")

    except Exception as exc:
        logger.exception("Pipeline failed at runtime: {}", exc)
        raise

    finally:
        spark.stop()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
