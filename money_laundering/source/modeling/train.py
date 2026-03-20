from pathlib import Path
import argparse

from loguru import logger

from source.config import MODELS_DIR, PROCESSED_DATA_DIR
from source.modeling.train_pipeline import main as run_train_pipeline

def run(
    data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    use_rus: bool = True,
    random_state: int = 42,
    rus_neg_pos_ratio: float = 10.0,
    pos_weight_multiplier: float = 0.1,
    beta: float = 0.5,
):
    """Executa pipeline de treino/eval com threshold otimizado por F-beta."""
    logger.info("Iniciando pipeline de treinamento AML")
    logger.info(
        "Parâmetros: use_rus={}, rus_neg_pos_ratio=1:{}, pos_weight_multiplier={}, beta={}",
        use_rus,
        rus_neg_pos_ratio,
        pos_weight_multiplier,
        beta,
    )

    run_train_pipeline(
        data_dir=data_dir,
        models_dir=models_dir,
        use_rus=use_rus,
        random_state=random_state,
        rus_neg_pos_ratio=rus_neg_pos_ratio,
        pos_weight_multiplier=pos_weight_multiplier,
        beta=beta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint de treinamento AML com foco em precisão")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--use-rus", type=int, choices=[0, 1], default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rus-neg-pos-ratio", type=float, default=10.0)
    parser.add_argument("--pos-weight-multiplier", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        use_rus=bool(args.use_rus),
        random_state=args.random_state,
        rus_neg_pos_ratio=args.rus_neg_pos_ratio,
        pos_weight_multiplier=args.pos_weight_multiplier,
        beta=args.beta,
    )
