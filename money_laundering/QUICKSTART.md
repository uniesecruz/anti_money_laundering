# Guia Rápido de Execução - Pipeline de Produção

Este guia demonstra como executar o pipeline completo de ML refatorado.

## 🎯 Objetivo

Transformar dados brutos em modelos treinados usando o pipeline consolidado, garantindo zero data leakage.

## 📋 Pré-requisitos

1. Python 3.8 ou superior instalado
2. Dados brutos em `data/external/` (ex: LI-Medium_accounts.csv, LI-Medium_Trans.csv)

## 🚀 Execução Passo a Passo

### Passo 1: Configurar Ambiente

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Passo 2: Preparar Dados

Este script carrega os dados brutos, enriquece transações com informações de contas, e divide em treino/OOT.

```bash
python source/dataset.py
```

**Saída esperada:**
```
data/processed/
├── df_treino.csv    # 80% dos dados (treino)
└── df_oot.csv       # 20% dos dados (validação temporal)
```

**O que acontece:**
1. ✅ Carrega `LI-Medium_accounts.csv` e `LI-Medium_Trans.csv`
2. ✅ Enriquece transações com dados FROM e TO
3. ✅ Divide temporalmente: 80% treino / 20% OOT
4. ✅ Salva em `data/processed/`

### Passo 3: Treinar Modelos

Este script aplica feature engineering, treina múltiplos modelos com RUS, e avalia em OOT.

```bash
python source/modeling/train_pipeline.py
```

**Saída esperada:**
```
models/
├── preprocessor.pkl           # Pipeline de transformações
├── logistic_regression.pkl    # Modelo Logistic Regression
├── random_forest.pkl          # Modelo Random Forest
├── xgboost.pkl                # Modelo XGBoost
├── lightgbm.pkl               # Modelo LightGBM
├── gradient_boosting.pkl      # Modelo Gradient Boosting
└── training_info.json         # Metadados

data/processed/
├── model_results_train.csv    # Métricas de treino
└── model_results_oot.csv      # Métricas de validação
```

**O que acontece:**
1. ✅ Carrega df_treino.csv e df_oot.csv
2. ✅ **Fit do preprocessador APENAS no treino**
3. ✅ Transform em treino e OOT (zero data leakage)
4. ✅ Aplica RUS apenas no treino
5. ✅ Treina 5 algoritmos
6. ✅ Avalia em treino e OOT
7. ✅ Salva modelos e métricas

### Passo 4: Analisar Resultados

```bash
# Ver métricas de validação OOT
python -c "import pandas as pd; print(pd.read_csv('data/processed/model_results_oot.csv'))"
```

## 🔍 Detalhamento do Pipeline

### Feature Engineering Automático

**Transformações Categóricas:**
- One-Hot Encoding: `Payment Format`
- Target Encoding: `Receiving Currency`, `Payment Currency`
- Frequency Encoding: `Timestamp`, `From Account`, `To Account`, etc.

**Transformações Numéricas:**
- Yeo-Johnson: `Amount Paid`, `Amount Received`, `Bank ID`, `Bank ID_To`, `From Bank`, `To Bank`
- StandardScaler: Todas as features numéricas

**Features Temporais (se houver):**
- Componentes: year, month, day, dayofweek, hour, etc.
- Features cíclicas: sin/cos para periodicidade
- Features de negócio: weekend, business hours, night

### Garantias de Integridade

✅ **Zero Data Leakage**
- Preprocessador faz `.fit()` apenas em `df_treino`
- Preprocessador faz `.transform()` em `df_treino` e `df_oot`
- Médias, frequências, etc. são aprendidas APENAS do treino

✅ **Balanceamento Seguro**
- RUS é aplicado via `imblearn.pipeline`
- RUS afeta apenas o treino durante `.fit()`
- Distribuição original é mantida em OOT

✅ **Caminhos Relativos**
- Todo o código usa `pathlib`
- Caminhos são relativos à raiz do projeto (`PROJ_ROOT`)
- Funciona em qualquer sistema operacional

## 📊 Interpretando Métricas

### Métricas de Treino (`model_results_train.csv`)
- Indicam se o modelo está aprendendo
- Naturalmente mais altas (modelo viu esses dados)

### Métricas de OOT (`model_results_oot.csv`)
- **Estas são as métricas REAIS do modelo**
- Simulam performance em dados futuros
- Use estas para comparar modelos

**Métricas Importantes:**
- **Precision**: Quantos alertas são verdadeiros positivos?
- **Recall**: Quantos casos de fraude capturamos?
- **F1-Score**: Balanço entre Precision e Recall
- **ROC-AUC**: Capacidade de discriminação geral

## 🐛 Troubleshooting

### Erro: "Arquivo não encontrado"
- Verifique se os dados estão em `data/external/`
- Confirme os nomes: `LI-Medium_accounts.csv` e `LI-Medium_Trans.csv`

### Erro: "ModuleNotFoundError"
- Execute `pip install -r requirements.txt`
- Ative o ambiente virtual primeiro

### Erro: "MemoryError"
- Use dataset menor (LI-Small ao invés de LI-Large)
- Reduza número de modelos treinados (edite `train_pipeline.py`)

## 🎓 Próximos Passos

1. **Análise de Features**: Verifique feature importance nos modelos tree-based
2. **Otimização de Hiperparâmetros**: Use GridSearchCV ou Optuna
3. **Validação Cruzada**: Implemente k-fold temporal
4. **Deployment**: Crie API Flask/FastAPI para servir o modelo
5. **Monitoring**: Implemente drift detection para produção

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique a documentação em `README.md`
2. Revise os logs de execução
3. Abra uma issue no GitHub

---

**Lembre-se**: Este pipeline foi projetado para **integridade acadêmica** e **reprodutibilidade**. Todas as transformações são documentadas e aplicadas de forma consistente.
