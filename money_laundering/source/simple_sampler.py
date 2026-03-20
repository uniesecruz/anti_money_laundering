"""Deprecated module: pandas sampling removed in full PySpark AML architecture."""

from loguru import logger


def main() -> None:
    logger.error(
        "source.simple_sampler is deprecated. Sampling, p-value checks, and z-test flow were removed."
    )
    logger.info("Use: python notebooks/01_dataprep.py")


if __name__ == "__main__":
    main()
