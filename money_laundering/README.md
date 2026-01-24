# Anti-Money Laundering Detection System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**TCC - Detecção de Lavagem de Dinheiro usando Machine Learning**

Este projeto implementa um sistema de detecção de lavagem de dinheiro (Anti-Money Laundering - AML) utilizando técnicas modernas de **Python Efficient Data Science** com Pandas, Scikit-Learn e bibliotecas especializadas de balanceamento de dados.

## 🎯 Objetivo

Desenvolver um pipeline de Machine Learning robusto e reprodutível para detecção automática de transações suspeitas de lavagem de dinheiro, garantindo:

- ✅ **Zero Data Leakage**: Separação rigorosa entre treino e validação temporal (OOT - Out-of-Time)
- ✅ **Reprodutibilidade**: Uso de caminhos relativos e seeds fixas
- ✅ **Pipeline Consolidado**: Transformações unificadas em um único ColumnTransformer
- ✅ **Balanceamento Seguro**: Random Under Sampling (RUS) aplicado apenas no treino

## 🛠️ Stack Tecnológica

### Core
- **Python 3.8+**
- **Pandas**: Manipulação eficiente de dados
- **Scikit-Learn**: Pipeline de ML e transformações
- **Imbalanced-Learn**: Balanceamento de classes com RUS

### Modelos Implementados
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### Bibliotecas Auxiliares
- **category_encoders**: Target Encoding para variáveis categóricas
- **loguru**: Logging estruturado
- **joblib**: Persistência de modelos
- **pathlib**: Gestão de caminhos multiplataforma

## 📊 Metodologia

### 1. Preparação de Dados
- Enriquecimento de transações com dados de contas (FROM e TO)
- Divisão temporal: 80% treino / 20% OOT (validação realista)
- Preservação da distribuição original para validação

### 2. Feature Engineering
**Transformações Categóricas:**
- **One-Hot Encoding**: Baixa cardinalidade (≤10 categorias)
- **Target Encoding**: Média cardinalidade (11-50 categorias)  
- **Frequency Encoding**: Alta cardinalidade (>50 categorias)

**Transformações Numéricas:**
- **Yeo-Johnson**: Normalização para variáveis com skewness
- **StandardScaler**: Padronização final

**Features Temporais:**
- Extração de componentes (year, month, day, hour, etc.)
- Features cíclicas (sin/cos para capturar periodicidade)
- Features de negócio (weekend, business hours, night)

### 3. Tratamento de Desbalanceamento
- **Random Under Sampling (RUS)** aplicado **apenas no treino**
- Mantém distribuição original (realista) em OOT
- Implementado via `imblearn.pipeline` para prevenir data leakage

### 4. Pipeline de Treinamento
```python
Pipeline:
  1. Preprocessor (ColumnTransformer)
     - DateTime Feature Extraction
     - Categorical Encoding
     - Numeric Transformations
  2. Imputer (mediana para valores ausentes)
  3. Scaler (StandardScaler)
  4. RUS (apenas no treino)
  5. Model (algoritmo de ML)
```

## 🚀 Como Usar

### Instalação

```bash
# Clonar repositório
git clone https://github.com/uniesecruz/anti_money_laundering.git
cd anti_money_laundering

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Execução do Pipeline

**1. Preparação de Dados**
```bash
python source/dataset.py
```
- Carrega dados de `data/external/`
- Enriquece transações com informações de contas
- Divide em treino (80%) e OOT (20%)
- Salva em `data/processed/`

**2. Treinamento de Modelos**
```bash
python source/modeling/train_pipeline.py
```
- Aplica feature engineering completo
- Treina 5 algoritmos com RUS
- Avalia em treino e OOT
- Salva modelos em `models/`
- Salva métricas em `data/processed/model_results_*.csv`

### Estrutura de Saída

```
data/processed/
├── df_treino.csv              # Dados de treino (brutos)
├── df_oot.csv                 # Dados OOT (brutos)
├── model_results_train.csv    # Métricas de treino
└── model_results_oot.csv      # Métricas de validação OOT

models/
├── preprocessor.pkl           # Pipeline de transformação
├── logistic_regression.pkl    # Modelo treinado
├── random_forest.pkl
├── xgboost.pkl
├── lightgbm.pkl
├── gradient_boosting.pkl
└── training_info.json         # Metadados do treinamento
```

## 📈 Avaliação de Modelos

Métricas calculadas em **treino** e **OOT**:
- **Accuracy**: Acurácia geral
- **Precision**: Precisão (quão confiáveis são os alertas)
- **Recall**: Revocação (quantos casos de fraude capturamos)
- **F1-Score**: Média harmônica de Precision e Recall
- **ROC-AUC**: Área sob a curva ROC
- **Average Precision**: Média de precisão (ótima para dados desbalanceados)

## 🔬 Integridade Acadêmica

Este projeto foi desenvolvido com rigor acadêmico, garantindo:

1. **Transparência Metodológica**: Código documentado e reprodutível
2. **Validação Temporal**: OOT reflete cenário realista de produção
3. **Prevenção de Data Leakage**: Fit apenas em treino, transform em OOT
4. **Descrição Honesta**: Stack tecnológica corresponde à implementação real (Pandas/Scikit-Learn, **sem Apache Spark ou GPU Rapids**)

## 📁 Estrutura do Projeto


```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── data
│   ├── external       <- Original datasets (HI/LI-Small/Medium/Large)
│   ├── interim        <- Intermediate transformed data
│   ├── processed      <- Final datasets for modeling (df_treino, df_oot)
│   └── raw            <- Enriched raw data (trans_enriched.csv)
│
├── docs               <- Documentation (MkDocs)
│
├── models             <- Trained models (.pkl) and training metadata
│
├── notebooks          <- Jupyter notebooks for exploration
│   ├── 01_dataprep.ipynb                      <- Data preparation exploration
│   ├── 09_transformacoes_das_variaveis_*.ipynb <- Feature engineering development
│   └── 10_treinamento_modelo.ipynb            <- Model training exploration
│
├── pyproject.toml     <- Project configuration
├── requirements.txt   <- Dependencies for reproducing the environment
├── setup.cfg          <- Configuration file for linters
│
└── source             <- Source code for production pipeline
    │
    ├── __init__.py             <- Makes source a Python module
    ├── config.py               <- Paths and configuration (uses pathlib)
    ├── dataset.py              <- **Data preparation script** (NEW)
    ├── preprocessing.py        <- **Feature engineering pipeline** (NEW)
    ├── features.py             <- Feature utilities
    ├── plots.py                <- Visualization utilities
    │
    └── modeling
        ├── __init__.py
        ├── train_pipeline.py   <- **Complete training pipeline** (NEW)
        ├── train.py            <- Legacy training code
        └── predict.py          <- Inference with trained models
```

## 🧪 Principais Classes e Módulos

### `source/preprocessing.py`
**`AMLPreprocessor`**: Pipeline completo de feature engineering
- Extração automática de features datetime
- Encoding categórico inteligente por cardinalidade
- Transformações numéricas (Yeo-Johnson)
- Imputação de valores ausentes
- Normalização (StandardScaler)
- **Garante zero data leakage** (fit em treino, transform em OOT)

### `source/modeling/train_pipeline.py`
**`AMLModelTrainer`**: Treinamento e avaliação de modelos
- Integração com RUS via `imblearn.pipeline`
- Treinamento de múltiplos algoritmos
- Avaliação robusta em treino e OOT
- Persistência de modelos e métricas
- Logging estruturado de experimentos

### `source/dataset.py`
**`load_and_enrich_data`**: Preparação de dados
- Merge de transações com contas (FROM/TO)
- Divisão temporal em treino/OOT
- Uso exclusivo de caminhos relativos (pathlib)

## 📚 Notebooks vs. Scripts de Produção

### Notebooks (Exploratórios)
- [01_dataprep.ipynb](notebooks/01_dataprep.ipynb): Exploração inicial de dados
- [09_transformacoes_das_variaveis_REORGANIZADO.ipynb](notebooks/09_transformacoes_das_variaveis_REORGANIZADO.ipynb): Desenvolvimento de feature engineering
- [10_treinamento_modelo.ipynb](notebooks/10_treinamento_modelo.ipynb): Experimentação de modelos

### Scripts de Produção (Refatorados)
- [source/dataset.py](source/dataset.py): Preparação automatizada de dados
- [source/preprocessing.py](source/preprocessing.py): Pipeline de transformações
- [source/modeling/train_pipeline.py](source/modeling/train_pipeline.py): Treinamento em produção

**Os scripts de produção consolidam toda a lógica dos notebooks em pipelines robustos e reprodutíveis.**

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos (TCC).

## 👥 Autores

- **Uniesecruz** - [GitHub](https://github.com/uniesecruz)

## 🙏 Agradecimentos

- Cookiecutter Data Science template
- Comunidade Python de Data Science
- Orientadores e colegas do TCC

---

**Nota**: Este projeto demonstra boas práticas de engenharia de ML, incluindo separação de código exploratório (notebooks) e código de produção (scripts), uso de pipelines para prevenir data leakage, e documentação honesta da stack tecnológica utilizada.

--------

