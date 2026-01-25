r"""
Configuração Centralizada do Projeto - Anti-Money Laundering Detection

Este módulo define TODOS os caminhos do projeto usando pathlib para garantir
portabilidade multiplataforma e eliminar hardcoding de paths.

CRITICAL: Todos os notebooks e scripts DEVEM importar caminhos daqui.
Nunca use caminhos absolutos ou strings hardcoded.

Autor: TCC - Anti-Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS DINÂMICOS - Funcionam em qualquer OS (Windows/Linux/Mac)
# ============================================================================

# Raiz do projeto (diretório contendo source/)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Diretórios de dados
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Diretórios de modelos e artefatos
MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"

# Diretórios de documentação e relatórios
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJ_ROOT / "docs"

# Diretório de notebooks
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"

# ============================================================================
# GARANTIR EXISTÊNCIA DE DIRETÓRIOS CRÍTICOS
# ============================================================================

def ensure_directories():
    """Cria diretórios essenciais se não existirem."""
    essential_dirs = [
        DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR
    ]
    
    for dir_path in essential_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

# Criar diretórios na importação
ensure_directories()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure loguru com tqdm se disponível
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Log do projeto root na importação
logger.info(f"[CONFIG] PROJ_ROOT: {PROJ_ROOT}")
logger.info(f"[CONFIG] Python executando de: {Path.cwd()}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data_path(filename: str, data_type: str = "processed") -> Path:
    """
    Retorna caminho completo para arquivo de dados.
    
    Args:
        filename: Nome do arquivo
        data_type: Tipo de dado ('raw', 'interim', 'processed', 'external')
    
    Returns:
        Path object para o arquivo
    
    Example:
        >>> from source.config import get_data_path
        >>> path = get_data_path('df_treino.csv', 'processed')
        >>> df = pd.read_csv(path)
    """
    dir_mapping = {
        'raw': RAW_DATA_DIR,
        'interim': INTERIM_DATA_DIR,
        'processed': PROCESSED_DATA_DIR,
        'external': EXTERNAL_DATA_DIR
    }
    
    if data_type not in dir_mapping:
        raise ValueError(f"data_type deve ser um de: {list(dir_mapping.keys())}")
    
    return dir_mapping[data_type] / filename


def get_model_path(filename: str) -> Path:
    """
    Retorna caminho completo para arquivo de modelo.
    
    Args:
        filename: Nome do arquivo de modelo
    
    Returns:
        Path object para o modelo
    
    Example:
        >>> from source.config import get_model_path
        >>> model = joblib.load(get_model_path('preprocessor.pkl'))
    """
    return MODELS_DIR / filename


def get_figure_path(filename: str) -> Path:
    """
    Retorna caminho completo para figura/plot.
    
    Args:
        filename: Nome do arquivo de figura
    
    Returns:
        Path object para a figura
    
    Example:
        >>> from source.config import get_figure_path
        >>> plt.savefig(get_figure_path('roc_curve.png'))
    """
    return FIGURES_DIR / filename


# ============================================================================
# EXPORT PÚBLICO
# ============================================================================

__all__ = [
    # Diretórios
    'PROJ_ROOT', 'DATA_DIR', 'RAW_DATA_DIR', 'INTERIM_DATA_DIR',
    'PROCESSED_DATA_DIR', 'EXTERNAL_DATA_DIR', 'MODELS_DIR', 'ARTIFACTS_DIR',
    'REPORTS_DIR', 'FIGURES_DIR', 'DOCS_DIR', 'NOTEBOOKS_DIR',
    # Funções
    'get_data_path', 'get_model_path', 'get_figure_path', 'ensure_directories'
]
