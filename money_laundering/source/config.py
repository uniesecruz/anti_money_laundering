"""Central project configuration for a full PySpark AML pipeline."""

from __future__ import annotations

import os
import ctypes
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from loguru import logger
import pyspark
from pyspark.sql import SparkSession

load_dotenv()

# Root directories
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJ_ROOT / "docs"
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"

# Stage paths (Parquet-first)
STAGE_PATHS: Dict[str, Path] = {
    "dataprep": PROCESSED_DATA_DIR / "01_dataprep_full.parquet",
    "data_quality": PROCESSED_DATA_DIR / "03_data_quality_full.parquet",
    "train": PROCESSED_DATA_DIR / "04_df_treino.parquet",
    "gap": PROCESSED_DATA_DIR / "04_df_gap.parquet",
    "oot": PROCESSED_DATA_DIR / "04_df_oot.parquet",
    "train_fe": PROCESSED_DATA_DIR / "05_df_treino_fe.parquet",
    "oot_fe": PROCESSED_DATA_DIR / "05_df_oot_fe.parquet",
    "train_vector": PROCESSED_DATA_DIR / "06_X_train_vector.parquet",
    "oot_vector": PROCESSED_DATA_DIR / "06_X_oot_vector.parquet",
    "predictions_oot": PROCESSED_DATA_DIR / "08_predictions_oot.parquet",
}

# Spark defaults for HI-Large
SPARK_DEFAULTS: Dict[str, str] = {
    "spark.driver.memory": os.getenv("SPARK_DRIVER_MEMORY", "12g"),
    "spark.executor.memory": os.getenv("SPARK_EXECUTOR_MEMORY", "12g"),
    "spark.driver.maxResultSize": os.getenv("SPARK_DRIVER_MAX_RESULT_SIZE", "2g"),
    "spark.sql.shuffle.partitions": os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "400"),
    "spark.default.parallelism": os.getenv("SPARK_DEFAULT_PARALLELISM", "400"),
    "spark.sql.adaptive.enabled": os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "true"),
    "spark.sql.adaptive.coalescePartitions.enabled": os.getenv(
        "SPARK_SQL_ADAPTIVE_COALESCE", "true"
    ),
    "spark.sql.autoBroadcastJoinThreshold": os.getenv(
        "SPARK_SQL_AUTO_BROADCAST_THRESHOLD", "10485760"
    ),
}


def ensure_directories() -> None:
    """Create all required directories for pipeline artifacts."""
    essential_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    for dir_path in essential_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_data_path(filename: str, data_type: str = "processed") -> Path:
    """Return a data path for a given data zone."""
    dir_mapping = {
        "raw": RAW_DATA_DIR,
        "interim": INTERIM_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "external": EXTERNAL_DATA_DIR,
    }
    if data_type not in dir_mapping:
        raise ValueError(f"data_type must be one of: {list(dir_mapping.keys())}")
    return dir_mapping[data_type] / filename


def get_stage_path(stage: str) -> Path:
    """Return output path for a pipeline stage."""
    if stage not in STAGE_PATHS:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(STAGE_PATHS)}")
    return STAGE_PATHS[stage]


def get_model_path(filename: str) -> Path:
    """Return full path for a model artifact."""
    return MODELS_DIR / filename


def get_figure_path(filename: str) -> Path:
    """Return full path for report figure."""
    return FIGURES_DIR / filename


def get_spark_session(
    app_name: str = "AML-FullSpark",
    extra_configs: Optional[Dict[str, str]] = None,
) -> SparkSession:
    """Create SparkSession tuned for full HI-Large processing."""
    pyspark_home = Path(pyspark.__file__).resolve().parent

    # Force SPARK_HOME to the venv PySpark distribution to avoid invalid external paths.
    spark_home_str = str(pyspark_home)
    if os.name == "nt":
        # Spark batch launchers are sensitive to some Unicode paths on Windows.
        short_buf = ctypes.create_unicode_buffer(260)
        result = ctypes.windll.kernel32.GetShortPathNameW(spark_home_str, short_buf, 260)
        if result > 0:
            spark_home_str = short_buf.value

        # Required by Spark/Hadoop local filesystem permissions on Windows.
        if not os.environ.get("HADOOP_HOME"):
            os.environ["HADOOP_HOME"] = "C:\\hadoop"
        if not os.environ.get("hadoop.home.dir"):
            os.environ["hadoop.home.dir"] = os.environ["HADOOP_HOME"]

    os.environ["SPARK_HOME"] = spark_home_str

    configs = dict(SPARK_DEFAULTS)
    if extra_configs:
        configs.update(extra_configs)

    builder = SparkSession.builder.appName(app_name)
    for key, value in configs.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))

    logger.info(f"[CONFIG] PROJ_ROOT={PROJ_ROOT}")
    logger.info(
        "[CONFIG] Spark session created with driver={} executor={} shuffle={}",
        configs["spark.driver.memory"],
        configs["spark.executor.memory"],
        configs["spark.sql.shuffle.partitions"],
    )
    return spark


ensure_directories()


__all__ = [
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "ARTIFACTS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "DOCS_DIR",
    "NOTEBOOKS_DIR",
    "STAGE_PATHS",
    "SPARK_DEFAULTS",
    "ensure_directories",
    "get_data_path",
    "get_stage_path",
    "get_model_path",
    "get_figure_path",
    "get_spark_session",
]
