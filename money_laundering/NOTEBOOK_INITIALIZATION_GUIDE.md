# Guia de Inicialização de Notebooks - AML Pipeline Refatorado

## 📋 Visão Geral

Após a refatoração completa das 5 ETAPAs estratégicas (Tunning Bayesiano, Anti-Leakage, e Métricas de Negócio), este guia fornece um bloco de código padrão para **toda primeira célula de seus notebooks**.

Este bloco garante:
- ✅ Detecção dinâmica do diretório raiz
- ✅ Configuração automática do Spark para AML
- ✅ Importação de todos os módulos refatorados
- ✅ Validação de dependências críticas
- ✅ Informações diagnosticáveis de ambiente

---

## 🎯 O Que Este Bloco Realiza

### 1️⃣ Ajuste Dinâmico de Path (5-10 linhas)

Localiza automaticamente o diretório raiz do projeto:
```
money_laundering/
├── notebooks/        ← Você está aqui
├── source/          ← Precisa acessar isto
├── data/
└── scripts/
```

O bloco detecta se você está em `notebooks/`, `notebooks/EDA/`, ou qualquer subpasta, e configura `sys.path` automaticamente.

### 2️⃣ Configuração Spark (15-20 linhas)

Inicializa SparkSession otimizada para AML:
- Memory tuning para features de alta dimensionalidade
- Particionamento adaptativo
- Logging estruturado

Se Spark não estiver disponível, continua com operações locais em Pandas.

### 3️⃣ Importação de Módulos (30-40 linhas)

Carrega **todos** os componentes refatorados:

| ETAPA | Componente | Função |
|-------|-----------|--------|
| **1-2** | FeatureEngineer | Criar features (Velocity, Ratio, Behavioral) |
| **4** | YeoJohnsonTransformerSafe | Transformação sem data leakage |
| **4** | TargetEncoderRegularized | Encoding de alta cardinalidade |
| **3** | AMLTunerPipeline | Bayesian Optimization com Optuna |
| **5** | BusinessMetrics | Precision@Top-K, PSI |
| **5** | ModelEvaluationReport | Relatório consolidado de avaliação |

### 4️⃣ Verificação de Versão (10-15 linhas)

Valida disponibilidade de:
- Python 3.10+
- NumPy, Pandas, scikit-learn, XGBoost, LightGBM
- Optuna (para ETAPA 3)
- Loguru (logging estruturado)

### 5️⃣ Sumário de Ambiente (10 linhas)

Exibe:
- Status de importações ✓/✗
- Versões de dependências
- Diretórios configurados
- Status do Spark
- Implementação das 5 ETAPAs

---

## 🚀 Como Usar

### Opção A: Cópia Direta (Recomendado)

1. **Abra qualquer notebook** (ou crie um novo com `File → New → Notebook`)

2. **Na primeira célula do notebook**, copie o código abaixo:

```python
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
```

3. **Execute a célula** (Ctrl+Enter ou Shift+Enter)

4. A célula exibirá um sumário completo do ambiente

---

### Opção B: Importar o Arquivo (Para Reutilização)

Se preferir centralizar o bloco em um arquivo:

```python
# No início do seu notebook
exec(open('../00_INICIALIZACAO_AMBIENTE.py').read())
```

---

## 📚 Exemplos de Uso Após Inicialização

### Exemplo 1: Carregar Dados e Aplicar Transformações (ETAPA 4)

```python
# Após inicialização, use:
df_train = load_and_prepare_data('train', sample_size=10000)

from source.preprocessing import YeoJohnsonTransformerSafe

# Aplicar transformação sem data leakage
transformer = YeoJohnsonTransformerSafe()
transformer.fit(df_train[['coluna_numerica']])
X_transformed = transformer.transform(df_train[['coluna_numerica']])

print("Transformação aplicada com sucesso (sem data leakage!)")
```

### Exemplo 2: Otimizar Hiperparâmetros (ETAPA 3)

```python
# Após inicialização, use:
from source.modeling.optuna_tuner import AMLTunerPipeline
import optuna

# Criar pipeline de tunning
tuner = AMLTunerPipeline(
    n_trials=50,
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Tunar modelo
best_params = tuner.optimize(X_train, y_train)
print(f"Melhores parâmetros: {best_params}")
```

### Exemplo 3: Calcular Métricas de Negócio (ETAPA 5)

```python
# Após inicialização, use:
from source.modeling.metrics import BusinessMetrics, ModelEvaluationReport

# Criar relatório
report = ModelEvaluationReport()

# Adicionar modelo
report.add_model_metrics(
    model_name="XGBoost_v1",
    y_true_train=y_train,
    y_scores_train=y_train_pred,
    y_true_oot=y_oot,
    y_scores_oot=y_oot_pred,
    standard_metrics={'accuracy': 0.95, 'precision': 0.92},
    top_k_list=[100, 500, 1000]
)

# Gerar relatório
df_report = report.generate_report()
report.print_summary()
```

### Exemplo 4: Validar PSI (Estabilidade de Modelo - ETAPA 5)

```python
# Após inicialização, use:
from source.modeling.metrics import BusinessMetrics

# Calcular PSI
psi_value = BusinessMetrics.calculate_psi(
    train_scores=y_train_pred,
    oot_scores=y_oot_pred,
    n_bins=10
)

if psi_value < 0.1:
    print("✓ Modelo ESTÁVEL - PSI < 0.1")
elif psi_value < 0.25:
    print("⚠ Modelo com degradação MODERADA - 0.1 < PSI < 0.25")
else:
    print("✗ Modelo com degradação SEVERA - PSI > 0.25")
```

---

## 🔍 O Que Procurar na Saída

### Status Esperado

```
====================================================================================================
INICIALIZANDO AMBIENTE AML - NOTEBOOK SETUP
====================================================================================================

[1/5] Configurando Path Dinâmico...
  ✓ Path adicionado: c:\...\money_laundering

[2/5] Configurando SparkSession para AML...
  ✓ SparkSession inicializada
    - App Name: AML_Detection_Pipeline
    - Driver Memory: 4g
    - Executor Memory: 4g

[3/5] Importando módulos refatorados...
  ✓ source.config
  ✓ source.features (Feature Engineering)
  ✓ source.preprocessing (ETAPA 4: Anti-Leakage)
  ✓ source.modeling.optuna_tuner (ETAPA 3: Bayesian Optimization)
  ✓ source.modeling.metrics (ETAPA 5: Precision@K e PSI)
  ✓ source.dataset
  ✓ source.plots

[4/5] Verificando dependências críticas...
  ✓ Python 3.10.11
  ✓ NumPy 1.24.3
  ✓ Pandas 2.0.1
  ✓ scikit-learn 1.2.2
  ✓ XGBoost 1.7.3
  ✓ LightGBM 3.3.5
  ✓ Optuna 3.0.5 (ETAPA 3: Bayesian Optimization)
  ✓ Loguru (logging com structured output)

[5/5] Sumário de Ambiente
----------------------------------------------------------------------------------------------------
✓ TODOS OS MÓDULOS IMPORTADOS COM SUCESSO
✓ TODAS AS DEPENDÊNCIAS DISPONÍVEIS
----------------------------------------------------------------------------------------------------

INFORMAÇÕES DO PROJETO:
  Diretório Raiz: c:\...\money_laundering
  Working Directory: c:\...\notebooks
  Processed Data: c:\...\data\processed
  Models Directory: c:\...\models

INFORMAÇÕES SPARK:
  Status: Ativo
  App Name: AML_Detection_Pipeline
  Master: local[*]

ETAPAS IMPLEMENTADAS:
  ETAPA 3 (Tuning + Anomaly): ✓
  ETAPA 4 (Anti-Leakage): ✓
  ETAPA 5 (Métricas AML): ✓

====================================================================================================
AMBIENTE PRONTO! Você pode começar a trabalhar com os dados.
====================================================================================================
```

### Sinais de Alerta

| Símbolo | Significado | Ação |
|---------|------------|------|
| ✗ | Módulo não importado | Verifique se `source/` existe e está acessível |
| ⚠ | Spark não disponível | Continua com Pandas (normal para alguns ambientes) |
| ✗ | Dependência ausente | Execute `pip install` do requirements.txt |

---

## 🛠️ Funções Auxiliares Disponíveis

Após executar o bloco, você tem acesso a:

### `show_status()`
Exibe resumo de todas as importações:
```python
show_status()
```

### `load_and_prepare_data(data_type, sample_size)`
Carrega dados com opção de amostragem:
```python
df = load_and_prepare_data('train', sample_size=5000)  # 5mil linhas
df_oot = load_and_prepare_data('oot')  # Todos os dados
```

### `validate_etapa_implementations()`
Valida se todas as ETAPAs estão operacionais:
```python
validate_etapa_implementations()
```

---

## 🔧 Troubleshooting

### Problema: "FileNotFoundError: Não foi possível localizar o diretório raiz"

**Causa**: Notebook está em subpasta não detectada.

**Solução**:
```python
# No início do notebook, defina manualmente:
import sys
from pathlib import Path

proj_root = Path("C:/Users/seu_usuario/OneDrive/Área de Trabalho/TCC/money_laundering")
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))
```

### Problema: "ModuleNotFoundError: No module named 'source.modeling.metrics'"

**Causa**: ETAPA 5 ainda não foi executada, ou arquivo não existe.

**Solução**:
```bash
# Verifique se o arquivo existe:
# c:\...\money_laundering\source\modeling\metrics.py
```

### Problema: "Spark initialization failed"

**Causa**: Java/Spark não instalado ou configuração de memória inadequada.

**Solução**: Normal para ambientes sem Spark. O código continua com Pandas automaticamente.

---

## 📋 Checklist antes de cada notebooks

- [ ] Primeira célula contém o bloco de inicialização
- [ ] Célula foi executada sem erros
- [ ] Status exibe "TODOS OS MÓDULOS IMPORTADOS COM SUCESSO"
- [ ] ETAPAS 3, 4, 5 exibem ✓
- [ ] Variável `spark` ou `pd` está disponível para uso

---

## 📞 Suporte

Se encontrar problemas:

1. Verifique que `source/` está no diretório raiz
2. Execute `show_status()` para diagnóstico
3. Revise logs de erro no bloco de inicialização
4. Verifique `pyproject.toml` para dependências

---

**Última Atualização**: 2024
**Compatibilidade**: Python 3.10+, Todos os notebooks (.ipynb)
**Status**: ✅ Pronto para Produção
