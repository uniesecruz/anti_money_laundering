"""Deprecated module: sampling removed in full PySpark AML architecture."""

from loguru import logger


def main() -> None:
    logger.error(
        "source.spark_sampler is deprecated. The project now runs 100% on full HI-Large with Spark and no sampling."
    )
    logger.info("Use: python notebooks/01_dataprep.py")


if __name__ == "__main__":
    main()
