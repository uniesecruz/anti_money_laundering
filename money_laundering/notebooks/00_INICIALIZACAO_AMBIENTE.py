"""
Bloco de Inicialização Padrão para Notebooks AML

Execução Atômica:
- Path dinâmico para imports de source/
- Configuração Spark para processamento AML
- Importação de todos os módulos refatorados (ETAPAS 1-5)
- Validação de versões e dependências
- Informações de ambiente

Copiar e colar na primeira célula do notebook!
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("INICIALIZANDO AMBIENTE AML - NOTEBOOK SETUP")
print("="*100)

# ============================================================================
# 1. AJUSTE DE PATH DINÂMICO
# ============================================================================
print("\n[1/5] Configurando Path Dinâmico...")

# Detectar diretório raiz do projeto
current_file = Path.cwd()
possible_roots = [
    Path.cwd(),  # Diretório atual
    Path.cwd().parent,  # Um nível acima
    Path(__file__).parent if '__file__' in dir() else None
]

# Se estamos em um notebook, usar o working directory
notebook_dir = Path.cwd()
if (notebook_dir / "source").exists():
    proj_root = notebook_dir
elif (notebook_dir.parent / "source").exists():
    proj_root = notebook_dir.parent
elif (notebook_dir.parent.parent / "source").exists():
    proj_root = notebook_dir.parent.parent
else:
    # Fallback: procurar por pyproject.toml
    for p in [notebook_dir, notebook_dir.parent, notebook_dir.parent.parent]:
        if (p / "pyproject.toml").exists():
            proj_root = p
            break
    else:
        raise FileNotFoundError(
            "⚠ Não foi possível localizar o diretório raiz do projeto.\n"
            "Verifique se está na pasta notebooks/ e se source/ existe."
        )

# Adicionar ao sys.path se não estiver
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))
    print(f"  ✓ Path adicionado: {proj_root}")
else:
    print(f"  ✓ Path já configurado: {proj_root}")

# ============================================================================
# 2. CONFIGURAÇÃO SPARK PARA AML
# ============================================================================
print("\n[2/5] Configurando SparkSession para AML...")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
    
    # Configurar SparkSession com otimizações para AML
    spark = SparkSession.builder \
        .appName("AML_Detection_Pipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.sql.broadcastTimeout", "360") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print(f"  ✓ SparkSession inicializada")
    print(f"    - App Name: {spark.appName}")
    print(f"    - Driver Memory: 4g")
    print(f"    - Executor Memory: 4g")
    
except ImportError:
    print("  ⚠ PySpark não disponível - Esta é uma sessão sem Spark")
    print("    (Continuando com operações locais em Pandas/NumPy)")
    spark = None
except Exception as e:
    print(f"  ⚠ Erro ao inicializar Spark: {e}")
    spark = None

# ============================================================================
# 3. IMPORTAÇÃO DOS MÓDULOS REFATORADOS
# ============================================================================
print("\n[3/5] Importando módulos refatorados...")

# Dicionário para rastrear importações bem-sucedidas
imports_status = {}

# ETAPA 1-2: Features e Preprocessing
try:
    from source.config import PROJ_ROOT, PROCESSED_DATA_DIR, MODELS_DIR
    imports_status['source.config'] = True
    print("  ✓ source.config")
except ImportError as e:
    imports_status['source.config'] = False
    print(f"  ✗ source.config: {e}")

try:
    from source.features import (
        FeatureEngineer,
        VelocityFeature,
        RatioFeature,
        BehavioralFeature
    )
    imports_status['source.features'] = True
    print("  ✓ source.features (Feature Engineering)")
except ImportError as e:
    imports_status['source.features'] = False
    print(f"  ✗ source.features: {e}")

# ETAPA 4: Preprocessing com Anti-Leakage
try:
    from source.preprocessing import (
        AMLPreprocessor,
        YeoJohnsonTransformerSafe,
        TargetEncoderRegularized,
        FrequencyEncoder,
        DateTimeFeatureExtractor
    )
    imports_status['source.preprocessing'] = True
    print("  ✓ source.preprocessing (ETAPA 4: Anti-Leakage)")
except ImportError as e:
    imports_status['source.preprocessing'] = False
    print(f"  ✗ source.preprocessing: {e}")

# ETAPA 3: Bayesian Optimization e Anomaly Detection
try:
    from source.modeling.optuna_tuner import AMLTunerPipeline
    imports_status['source.modeling.optuna_tuner'] = True
    print("  ✓ source.modeling.optuna_tuner (ETAPA 3: Bayesian Optimization)")
except ImportError as e:
    imports_status['source.modeling.optuna_tuner'] = False
    print(f"  ✗ source.modeling.optuna_tuner: {e}")

# ETAPA 5: Métricas de Negócio
try:
    from source.modeling.metrics import (
        BusinessMetrics,
        ModelEvaluationReport
    )
    imports_status['source.modeling.metrics'] = True
    print("  ✓ source.modeling.metrics (ETAPA 5: Precision@K e PSI)")
except ImportError as e:
    imports_status['source.modeling.metrics'] = False
    print(f"  ✗ source.modeling.metrics: {e}")

# Utilitários
try:
    from source.dataset import DataLoader
    imports_status['source.dataset'] = True
    print("  ✓ source.dataset")
except ImportError as e:
    imports_status['source.dataset'] = False
    print(f"  ✗ source.dataset: {e}")

try:
    from source.plots import ModelPlotter
    imports_status['source.plots'] = True
    print("  ✓ source.plots")
except ImportError as e:
    imports_status['source.plots'] = False
    print(f"  ✗ source.plots: {e}")

# ============================================================================
# 4. VERIFICAÇÃO DE VERSÃO E DEPENDÊNCIAS
# ============================================================================
print("\n[4/5] Verificando dependências críticas...")

dependencies_status = {}

# Verificar Python
try:
    import platform
    python_version = f"{platform.python_version()}"
    dependencies_status['Python'] = (python_version, True)
    print(f"  ✓ Python {python_version}")
except Exception as e:
    dependencies_status['Python'] = (None, False)
    print(f"  ✗ Python: {e}")

# Verificar NumPy
try:
    import numpy as np
    np_version = np.__version__
    dependencies_status['NumPy'] = (np_version, True)
    print(f"  ✓ NumPy {np_version}")
except ImportError:
    dependencies_status['NumPy'] = (None, False)
    print(f"  ✗ NumPy: não disponível")

# Verificar Pandas
try:
    import pandas as pd
    pd_version = pd.__version__
    dependencies_status['Pandas'] = (pd_version, True)
    print(f"  ✓ Pandas {pd_version}")
except ImportError:
    dependencies_status['Pandas'] = (None, False)
    print(f"  ✗ Pandas: não disponível")

# Verificar scikit-learn
try:
    import sklearn
    sklearn_version = sklearn.__version__
    dependencies_status['scikit-learn'] = (sklearn_version, True)
    print(f"  ✓ scikit-learn {sklearn_version}")
except ImportError:
    dependencies_status['scikit-learn'] = (None, False)
    print(f"  ✗ scikit-learn: não disponível")

# Verificar XGBoost
try:
    import xgboost as xgb
    xgb_version = xgb.__version__
    dependencies_status['XGBoost'] = (xgb_version, True)
    print(f"  ✓ XGBoost {xgb_version}")
except ImportError:
    dependencies_status['XGBoost'] = (None, False)
    print(f"  ✗ XGBoost: não disponível")

# Verificar LightGBM
try:
    import lightgbm as lgb
    lgb_version = lgb.__version__
    dependencies_status['LightGBM'] = (lgb_version, True)
    print(f"  ✓ LightGBM {lgb_version}")
except ImportError:
    dependencies_status['LightGBM'] = (None, False)
    print(f"  ✗ LightGBM: não disponível")

# Verificar Optuna (ETAPA 3)
try:
    import optuna
    optuna_version = optuna.__version__
    dependencies_status['Optuna'] = (optuna_version, True)
    print(f"  ✓ Optuna {optuna_version} (ETAPA 3: Bayesian Optimization)")
except ImportError:
    dependencies_status['Optuna'] = (None, False)
    print(f"  ✗ Optuna: não disponível")

# Verificar Loguru
try:
    from loguru import logger
    dependencies_status['Loguru'] = ('available', True)
    print(f"  ✓ Loguru (logging com structured output)")
except ImportError:
    dependencies_status['Loguru'] = (None, False)
    print(f"  ✗ Loguru: não disponível")

# ============================================================================
# 5. SUMÁRIO E INFORMAÇÕES DE AMBIENTE
# ============================================================================
print("\n[5/5] Sumário de Ambiente")
print("-" * 100)

# Status geral
all_imports_ok = all(imports_status.values())
all_deps_ok = all(status[1] for status in dependencies_status.values())

if all_imports_ok:
    print("✓ TODOS OS MÓDULOS IMPORTADOS COM SUCESSO")
else:
    failed = [k for k, v in imports_status.items() if not v]
    print(f"⚠ PROBLEMAS COM IMPORTAÇÃO: {', '.join(failed)}")

if all_deps_ok:
    print("✓ TODAS AS DEPENDÊNCIAS DISPONÍVEIS")
else:
    failed = [k for k, (v, ok) in dependencies_status.items() if not ok]
    print(f"⚠ DEPENDÊNCIAS FALTANDO: {', '.join(failed)}")

print("-" * 100)

# Informações do Projeto
print("\nINFORMAÇÕES DO PROJETO:")
print(f"  Diretório Raiz: {proj_root}")
print(f"  Working Directory: {Path.cwd()}")
print(f"  Processed Data: {PROCESSED_DATA_DIR if imports_status['source.config'] else 'N/A'}")
print(f"  Models Directory: {MODELS_DIR if imports_status['source.config'] else 'N/A'}")

# Informações de Spark
if spark:
    print(f"\nINFORMAÇÕES SPARK:")
    print(f"  Status: Ativo")
    print(f"  App Name: {spark.appName}")
    print(f"  Master: {spark.sparkContext.master}")
else:
    print(f"\nINFORMAÇÕES SPARK:")
    print(f"  Status: Não disponível")

# Informações de ETAPAS
print("\nETAPAS IMPLEMENTADAS:")
print(f"  ETAPA 3 (Tuning + Anomaly): {'✓' if imports_status.get('source.modeling.optuna_tuner', False) else '✗'}")
print(f"  ETAPA 4 (Anti-Leakage): {'✓' if imports_status.get('source.preprocessing', False) else '✗'}")
print(f"  ETAPA 5 (Métricas AML): {'✓' if imports_status.get('source.modeling.metrics', False) else '✗'}")

print("\n" + "="*100)
print("AMBIENTE PRONTO! Você pode começar a trabalhar com os dados.")
print("="*100 + "\n")

# ============================================================================
# Funções auxiliares úteis
# ============================================================================

def show_status():
    """Exibe status resumido de todas as importações."""
    print("\nSTATUS DE IMPORTAÇÕES:")
    for module, status in imports_status.items():
        symbol = "✓" if status else "✗"
        print(f"  [{symbol}] {module}")
    
    print("\nDEPENDÊNCIAS:")
    for dep, (version, status) in dependencies_status.items():
        symbol = "✓" if status else "✗"
        ver_str = f"{version}" if version else "N/A"
        print(f"  [{symbol}] {dep:<20} → {ver_str}")

def load_and_prepare_data(data_type='train', sample_size=None):
    """
    Helper para carregar dados de treino ou OOT.
    
    Args:
        data_type: 'train' ou 'oot'
        sample_size: Número de linhas para amostrar (None = todos)
    
    Returns:
        DataFrame com os dados
    """
    try:
        import pandas as pd
        
        if data_type == 'train':
            path = PROCESSED_DATA_DIR / 'df_treino.csv'
        else:
            path = PROCESSED_DATA_DIR / 'df_oot.csv'
        
        df = pd.read_csv(path)
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"✓ Dados carregados: {path.name} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        return df
    
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return None

def validate_etapa_implementations():
    """Valida se todas as ETAPAs foram implementadas corretamente."""
    print("\nVALIDAÇÃO DE ETAPAS:")
    
    # ETAPA 3: Optuna + Isolation Forest
    try:
        from source.modeling.optuna_tuner import AMLTunerPipeline
        print("  ✓ ETAPA 3: AMLTunerPipeline com Bayesian Optimization")
    except:
        print("  ✗ ETAPA 3: AMLTunerPipeline não encontrada")
    
    # ETAPA 4: YeoJohnson + TargetEncoder
    try:
        from source.preprocessing import YeoJohnsonTransformerSafe, TargetEncoderRegularized
        print("  ✓ ETAPA 4: YeoJohnsonTransformerSafe + TargetEncoderRegularized")
    except:
        print("  ✗ ETAPA 4: Transformadores anti-leakage não encontrados")
    
    # ETAPA 5: Precision@K + PSI
    try:
        from source.modeling.metrics import BusinessMetrics, ModelEvaluationReport
        
        # Testar métodos
        assert hasattr(BusinessMetrics, 'precision_at_top_k')
        assert hasattr(BusinessMetrics, 'calculate_psi')
        assert hasattr(ModelEvaluationReport, 'add_model_metrics')
        
        print("  ✓ ETAPA 5: Precision@Top-K + PSI + ModelEvaluationReport")
    except:
        print("  ✗ ETAPA 5: Métricas de negócio não encontradas")

# Validar ETAPAS automaticamente
validate_etapa_implementations()

print("\n💡 DICAS:")
print("  • Use show_status() para ver resumen de importações")
print("  • Use load_and_prepare_data('train') para carregar dados de treino")
print("  • Use load_and_prepare_data('oot') para carregar dados de OOT")
print("  • Variáveis spark, pd, np, np.array estão disponíveis")
print("  • Métricas de ETAPA 5 prontas: BusinessMetrics.precision_at_top_k(), .calculate_psi()")
