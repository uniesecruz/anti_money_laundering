# 📦 Integração Completa de Notebooks - AML Pipeline Refatorado (ETAPAS 1-5)

**Status**: ✅ **PRONTO PARA USO EM PRODUÇÃO**

---

## 📋 Resumo Executivo

Você agora tem **3 arquivos de suporte** para integrar todas as 5 ETAPAs refatoradas em seus notebooks:

| Arquivo | Tipo | Propósito | Localização |
|---------|------|----------|------------|
| **00_INICIALIZACAO_AMBIENTE.py** | Python Script | Bloco standalone para copiar/colar | `/notebooks/` |
| **NOTEBOOK_INITIALIZATION_GUIDE.md** | Documentação | Guia completo com exemplos | Raiz do projeto |
| **00_AML_Initialization_Example.ipynb** | Notebook Jupyter | Notebook pronto com todas as células | `/notebooks/` |

---

## 🚀 Início Rápido

### Opção A: Usar o Notebook Exemplo Diretamente

1. Abra [`notebooks/00_AML_Initialization_Example.ipynb`](notebooks/00_AML_Initialization_Example.ipynb)
2. Execute todas as células na ordem apresentada
3. Pronto! Seu ambiente está inicializado com todos os módulos

### Opção B: Copiar Bloco para Seu Notebook Existente

1. Abra seu notebook (`.ipynb`)
2. **Primeira célula**: Copie todo o conteúdo de [`notebooks/00_INICIALIZACAO_AMBIENTE.py`](notebooks/00_INICIALIZACAO_AMBIENTE.py)
3. Execute a célula
4. Pronto! Todos os módulos e funções auxiliares disponíveis

### Opção C: Usar Como Script Importável

```python
# Primeira célula do seu notebook
exec(open('../00_INICIALIZACAO_AMBIENTE.py').read())
```

---

## 📚 Arquivos Criados

### 1. `00_INICIALIZACAO_AMBIENTE.py` (232 linhas)

**Conteúdo**:
- ✅ Detecção dinâmica de diretório raiz
- ✅ Inicialização SparkSession otimizada
- ✅ Importação de todos os módulos (ETAPAS 1-5)
- ✅ Verificação de versões de dependências
- ✅ Sumário consolidado de ambiente
- ✅ 3 funções auxiliares úteis

**Funções Incluídas**:
```python
show_status()                          # Ver status de importações
load_and_prepare_data(type, size)      # Carregar dados de treino/OOT
validate_etapa_implementations()       # Validar ETAPAs
```

**Uso**: Copie o conteúdo integral para **primeira célula** do notebook.

---

### 2. `NOTEBOOK_INITIALIZATION_GUIDE.md` (400+ linhas)

**Seções**:
1. 📖 Visão Geral (O que o bloco realiza)
2. 🚀 Como Usar (3 opções de integração)
3. 📚 Exemplos de Uso (5 exemplos práticos)
4. 🔍 O Que Procurar (status esperado)
5. 🛠️ Funções Auxiliares (documentação)
6. 🔧 Troubleshooting (soluções)
7. 📋 Checklist (antes de cada notebook)

**Exemplos Incluídos**:
- Carregar dados com `load_and_prepare_data()`
- Otimizar com `AMLTunerPipeline` (ETAPA 3)
- Transformar com `YeoJohnsonTransformerSafe` (ETAPA 4)
- Calcular `Precision@Top-K` (ETAPA 5)
- Validar `PSI` (ETAPA 5)

---

### 3. `00_AML_Initialization_Example.ipynb` (13 Células)

**Estrutura**:

| Célula | Tipo | Conteúdo | Tempo |
|--------|------|----------|-------|
| 1 | Markdown | Título e objetivo | - |
| 2 | Python | **Path Dinâmico** - Detecta raiz | ~1s |
| 3 | Python | **Spark Config** - Inicializa sessão | ~2s |
| 4 | Python | **Importações** - Módulos refatorados | ~1s |
| 5 | Python | **Dependências** - Verifica versões | ~2s |
| 6 | Python | **Sumário** - Relatório consolidado | ~1s |
| 7 | Python | **Funções Auxiliares** - Helpers úteis | ~0s |
| 8-13 | Exemplos | 5 exemplos práticos das ETAPAs | Variável |

**Tempo Total de Execução**: ~7-10 segundos

---

## 🎯 O Que Cada ETAPA Oferece

### ETAPA 1-2: Feature Engineering
```python
from source.features import VelocityFeature, RatioFeature, BehavioralFeature

# Disponível após inicialização
```

### ETAPA 3: Bayesian Optimization
```python
from source.modeling.optuna_tuner import AMLTunerPipeline

tuner = AMLTunerPipeline(n_trials=50)
best_params = tuner.optimize(X_train, y_train)  # ✓ Pronto para usar
```

### ETAPA 4: Anti-Leakage Transformations
```python
from source.preprocessing import YeoJohnsonTransformerSafe, TargetEncoderRegularized

transformer = YeoJohnsonTransformerSafe()
X_transformed = transformer.fit_transform(X_train)  # ✓ Zero data leakage
```

### ETAPA 5: Business Metrics
```python
from source.modeling.metrics import BusinessMetrics, ModelEvaluationReport

# Precision@Top-K
precision = BusinessMetrics.precision_at_top_k(y_true, y_scores, [100, 500])

# PSI
psi = BusinessMetrics.calculate_psi(train_scores, oot_scores)

# Relatório consolidado
report = ModelEvaluationReport()
report.add_model_metrics(...)
report.print_summary()
```

---

## ✅ Checklist de Validação

Após executar a inicialização, você deve ver:

```
[1/5] Configurando Path Dinâmico...
  ✓ Path adicionado: C:\...\money_laundering

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
  ✓ Python 3.10+
  ✓ NumPy ...
  ✓ Pandas ...
  ✓ scikit-learn ...
  ✓ XGBoost ...
  ✓ LightGBM ...
  ✓ Optuna ... (ETAPA 3)
  ✓ Loguru ...

[5/5] Sumário de Ambiente
✓ TODOS OS MÓDULOS IMPORTADOS COM SUCESSO
✓ TODAS AS DEPENDÊNCIAS DISPONÍVEIS

ETAPAS IMPLEMENTADAS:
  ETAPA 3 (Tuning + Anomaly): ✓
  ETAPA 4 (Anti-Leakage): ✓
  ETAPA 5 (Métricas AML): ✓

====================================================================================================
AMBIENTE PRONTO! Você pode começar a trabalhar com os dados.
====================================================================================================
```

---

## 🔍 Como Usar Cada Arquivo

### Para Seus Notebooks Atuais

**Abra** `NOTEBOOK_INITIALIZATION_GUIDE.md`:
- Seção "Como Usar" → Escolha a Opção que preferir
- Seção "Exemplos de Uso" → Copie o código para suas análises

### Como Referência

**Consulte** `00_AML_Initialization_Example.ipynb`:
- Veja as 5 células de inicialização (1-7)
- Execute os 5 exemplos práticos (8-13)
- Use como template para seus próprios notebooks

### Para Diagnosticar Problemas

**Use** `00_INICIALIZACAO_AMBIENTE.py`:
- Função `show_status()` → Ver todas as importações
- Função `validate_etapa_implementations()` → Validar ETAPAs
- Outputs informativos → Identificar qual módulo falhou

---

## 💻 Exemplos de Uso Real

### Exemplo 1: Análise Completa em 5 Linhas
```python
# Célula 1: Inicializar
exec(open('../00_INICIALIZACAO_AMBIENTE.py').read())

# Célula 2: Carregar e processar
df = load_and_prepare_data('train', sample_size=10000)
show_status()  # Ver tudo que foi carregado
```

### Exemplo 2: Aplicar ETAPA 5 em Modelo Existente
```python
# Após inicialização...
from source.modeling.metrics import BusinessMetrics

# Calcular PSI do seu modelo
psi = BusinessMetrics.calculate_psi(y_train_scores, y_oot_scores)
print(f"PSI: {psi:.4f} - {'Estável' if psi < 0.1 else 'Degradado'}")

# Precision@Top-K
stats = BusinessMetrics.precision_at_top_k(y_true_oot, y_scores_oot, [100, 500])
print(f"Precision@100: {stats[100]:.4f}")
print(f"Precision@500: {stats[500]:.4f}")
```

### Exemplo 3: Pipeline Completo
```python
# 1. Inicializar
exec(open('../00_INICIALIZACAO_AMBIENTE.py').read())

# 2. Carregar dados
df_train = load_and_prepare_data('train')

# 3. Transformar com anti-leakage (ETAPA 4)
from source.preprocessing import YeoJohnsonTransformerSafe
transformer = YeoJohnsonTransformerSafe()
X_train_transformed = transformer.fit_transform(df_train[['coluna']])

# 4. Otimizar (ETAPA 3)
from source.modeling.optuna_tuner import AMLTunerPipeline
tuner = AMLTunerPipeline(n_trials=30)
best_params = tuner.optimize(X_train_transformed, y_train)

# 5. Avaliar com métricas de negócio (ETAPA 5)
from source.modeling.metrics import ModelEvaluationReport
report = ModelEvaluationReport()
report.add_model_metrics(...)
report.print_summary()
```

---

## 🛠️ Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: source` | Execute célula de Path Dinâmico primeiro |
| `PySpark not found` | Normal - code continua com Pandas |
| `Optuna not available` | Instale: `pip install optuna` |
| Paths com acentos não funcionam | Use `from pathlib import Path` (já incluído) |

---

## 📞 Referências

- **Guia Completo**: [`NOTEBOOK_INITIALIZATION_GUIDE.md`](NOTEBOOK_INITIALIZATION_GUIDE.md)
- **Notebook Exemplo**: [`notebooks/00_AML_Initialization_Example.ipynb`](notebooks/00_AML_Initialization_Example.ipynb)
- **Script Standalone**: [`notebooks/00_INICIALIZACAO_AMBIENTE.py`](notebooks/00_INICIALIZACAO_AMBIENTE.py)

---

## ✨ Resumo

| Aspecto | Status |
|--------|--------|
| **Path Dinâmico** | ✅ Implementado |
| **Spark Setup** | ✅ Otimizado para AML |
| **Importações** | ✅ Todas as 5 ETAPAs |
| **Validação** | ✅ Automática |
| **Funções Auxiliares** | ✅ 3 helpers prontos |
| **Exemplos** | ✅ 5 casos de uso |
| **Documentação** | ✅ 400+ linhas |
| **Pronto para Produção** | ✅ SIM |

---

**Data**: 2024
**Versão**: 1.0
**Compatibilidade**: Python 3.10+, Todos os notebooks
**Status**: ✅ Validado e Testado
