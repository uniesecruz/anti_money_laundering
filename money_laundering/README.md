# Anti-Money Laundering Detection System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**TCC - Detecção de Lavagem de Dinheiro usando Scikit-Learn Production Pipelines**

Este projeto implementa um sistema de detecção de lavagem de dinheiro (Anti-Money Laundering - AML) utilizando **técnicas profissionais de MLOps** com Scikit-Learn, Imbalanced-Learn e boas práticas de engenharia de software.

---

## 🎯 Filosofia do Projeto

> **Reprodutibilidade > Complexidade**  
> **Metodologia > Infraestrutura**  
> **Honestidade Acadêmica > Marketing**

Este projeto prioriza:
- ✅ **Scikit-Learn Pipelines** (modular, reprodutível, testável)
- ✅ **Imbalanced-Learn Strategy** (RUS aplicado corretamente)
- ✅ **Pathlib Dinâmico** (funciona em qualquer máquina/OS)
- ✅ **Zero Data Leakage** (fit apenas em treino)
- ✅ **Documentação Honesta** (sem "buzzwords" vazios)

---

## 🛠️ Stack Tecnológica REAL

### Core Processing
- **Python 3.8+**
- **Pandas** - Manipulação eficiente de dados
- **NumPy** - Computação numérica
- **Scikit-Learn** - ML Pipelines e transformações

### Modelagem
- **Imbalanced-Learn** - Random Under Sampling (RUS)
- **Category Encoders** - Target Encoding
- **XGBoost** - Gradient Boosting otimizado
- **LightGBM** - Gradient Boosting leve

### Infraestrutura
- **Pathlib** - Gestão de caminhos multiplataforma
- **Loguru** - Logging estruturado
- **Joblib** - Persistência de pipelines

### Visualização
- **Matplotlib** - Gráficos base
- **Seaborn** - Visualizações estatísticas
- **Plotly** - Gráficos interativos

---

## ⚠️ Honestidade Acadêmica

**O que este projeto NÃO utiliza:**
- ❌ Apache Spark (não necessário para o volume de dados)
- ❌ GPU Rapids (não implementado)
- ❌ Big Data Cluster (infraestrutura desnecessária)
- ❌ Kubernetes/Docker (fora do escopo do TCC)

**Por quê?**  
Este projeto foca em **metodologia correta** e **reprodutibilidade**, não em infraestrutura complexa. Pandas + Scikit-Learn são suficientes e profissionais para datasets de até milhões de registros.

---

## 📊 Metodologia

### 1. Preparação de Dados
```python
# source/dataset.py
# - Carrega dados de data/external/
# - Enriquece transações (FROM + TO accounts)
# - Divide temporalmente: 80% treino / 20% OOT
# - Usa APENAS pathlib (zero hardcoding)
```

### 2. Feature Engineering (Pipeline)
```python
# sklearn.compose.ColumnTransformer
# - Numéricas: Imputer → Log+1 → StandardScaler
# - Categóricas (≤10): OneHotEncoder
# - Categóricas (>10): TargetEncoder
# - FIT apenas no treino!
```

### 3. Balanceamento (RUS)
```python
# imblearn.pipeline.Pipeline
# - Random Under Sampling aplicado APENAS no fit
# - OOT mantém distribuição original (realista)
# - Previne data leakage
```

### 4. Treinamento
```python
# Múltiplos algoritmos:
# - Logistic Regression
# - Random Forest
# - Gradient Boosting
# - XGBoost
# - LightGBM
```

### 5. Avaliação
```python
# Métricas em treino E OOT:
# - Accuracy, Precision, Recall, F1
# - ROC-AUC, Average Precision
# - Confusion Matrix
```

---

## 🚀 Uso do Pipeline

### Instalação
```bash
git clone https://github.com/uniesecruz/anti_money_laundering.git
cd anti_money_laundering

python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Validação
```bash
python validate_pipeline.py
```

### Pipeline Completo
```bash
# 1. Preparar dados
python source/dataset.py

# 2. Treinar modelos
python source/modeling/train_pipeline.py
```

### Notebook Unificado
Abra [`notebooks/pipeline_final.ipynb`](notebooks/pipeline_final.ipynb) no Jupyter/VS Code.

---

## 📁 Estrutura do Projeto (Refatorada)


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

