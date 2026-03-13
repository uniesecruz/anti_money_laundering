# ETAPA 3: CÓDIGO PRONTO PARA INTEGRAÇÃO

Este arquivo mostra exemplos práticos de como integrar ETAPA 3 no pipeline principal.

## 1. IMPORTAÇÃO BÁSICA

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from source.modeling.optuna_tuner import AMLTunerPipeline, AnomalyDetector, OptunaHyperparametersOptimizer
```

## 2. EXEMPLO 1: PIPELINE COMPLETO (Recomendado)

```python
# Setup
tscv = TimeSeriesSplit(n_splits=5)

# Criar e treinar pipeline
pipeline = AMLTunerPipeline(
    model_type='xgboost',          # XGBoost com scale_pos_weight otimizado
    n_optuna_trials=50,            # 50 tentativas de otimização
    anomaly_contamination=0.05     # 5% de anomalias
)

# Executar 3 passos: IF -> Features -> Optuna
pipeline.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    tscv=tscv
)

# Predições
y_pred_train = pipeline.predict(X_train_augmented)
y_pred_val = pipeline.predict(X_val_augmented)
y_pred_test = pipeline.predict(X_test)

# Probabilidades
y_proba_test = pipeline.predict_proba(X_test)

# Acessar resultados
print(f"Best params: {pipeline.optimizer.best_params}")
print(f"Best F1-Score: {pipeline.optimizer.best_trial.value:.4f}")
print(f"scale_pos_weight otimizado: {pipeline.optimizer.best_params['scale_pos_weight']:.4f}")
print(f"Features: {pipeline.X_train_augmented.columns.tolist()}")
```

## 3. EXEMPLO 2: COMPONENTES INDIVIDUAIS

### 3.1 Apenas Anomaly Detection

```python
# Treinar Isolation Forest
detector = AnomalyDetector(contamination=0.05)
detector.fit(X_train)

# Detectar anomalias
outliers = detector.detect(X_test)  # -1 ou 1
anomaly_scores = detector.get_anomaly_score(X_test)  # [0, 1]

# Usar scores como feature
X_with_anomaly = X_test.copy()
X_with_anomaly['anomaly_score'] = anomaly_scores

print(f"Anomaly ratio: {detector.get_anomaly_ratio(X_test):.2%}")
```

### 3.2 Apenas Optuna Optimization

```python
# Sem Isolation Forest, só optimize hiperparâmetros
optimizer = OptunaHyperparametersOptimizer(
    model_type='xgboost',
    n_trials=50,
    timeout=300
)

optimizer.optimize(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    tscv=tscv
)

# Usar modelo otimizado
best_model = optimizer.get_best_model()
y_pred = best_model.predict(X_test)

print(f"Best params: {optimizer.best_params}")
print(f"Best F1-Score: {optimizer.best_trial.value:.4f}")
```

## 4. EXEMPLO 3: INTEGRAÇÃO COM PIPELINE EXISTENTE

### Na ETAPA 2 ou posterior, adicione:

```python
from source.modeling.optuna_tuner import AMLTunerPipeline

# ... código anterior (feature engineering, etc) ...

# ETAPA 3: Tuning com anomaly detection
pipeline_etapa3 = AMLTunerPipeline(
    model_type='xgboost',
    n_optuna_trials=100,
    anomaly_contamination=0.05
)

pipeline_etapa3.fit(X_train, y_train, X_val, y_val, tscv)

# Usar modelo final
y_pred_final = pipeline_etapa3.predict(X_test)
y_proba_final = pipeline_etapa3.predict_proba(X_test)

# Guardar pipeline para uso posterior
import pickle
pickle.dump(pipeline_etapa3, open('models/pipeline_etapa3.pkl', 'wb'))
```

## 5. EXEMPLO 4: CONFIGURAÇÕES AVANÇADAS

```python
# Otimização com timeout
optimizer = OptunaHyperparametersOptimizer(
    model_type='lightgbm',  # Use LightGBM ao invés de XGBoost
    n_trials=100,
    timeout=600  # 10 minutos máximo
)

# Otimização com seed para reprodutibilidade
optimizer.sampler = optuna.samplers.TPESampler(seed=42)

# Usar diferentes métricas (padrão é F1)
# Nota: Modifique a função _objective_xgboost() para suas métricas
```

## 6. EXEMPLO 5: VERIFICAÇÃO DE QUALIDADE

```python
# Verificar se anomaly scores têm separação adequada
normal_mask = y_train == 0
anomaly_mask = y_train == 1

normal_scores = pipeline.X_train_augmented.loc[normal_mask, 'anomaly_score'].mean()
anomaly_scores = pipeline.X_train_augmented.loc[anomaly_mask, 'anomaly_score'].mean()

ratio = anomaly_scores / normal_scores if normal_scores > 0 else float('inf')
print(f"Mean anomaly score (normal): {normal_scores:.4f}")
print(f"Mean anomaly score (anomaly): {anomaly_scores:.4f}")
print(f"Razão de discriminação: {ratio:.2f}x")

# Boa separação quando ratio > 5x
assert ratio > 5, f"Anomaly scores com baixa discriminação: {ratio:.2f}x"
```

## 7. EXEMPLO 6: SALVANDO E CARREGANDO

```python
import pickle

# Salvar pipeline
with open('models/etapa3_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Carregar pipeline
with open('models/etapa3_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Usar pipeline carregado
y_pred_new = loaded_pipeline.predict(X_new)
```

## 8. EXEMPLO 7: COMPARAÇÃO ANTES/DEPOIS

```python
from sklearn.metrics import f1_score, precision_score, recall_score

# Treinar modelo SEM anomaly detection (baseline)
baseline_model = xgboost.XGBClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

# Com ETAPA 3
pipeline_etapa3.fit(X_train, y_train, X_val, y_val, tscv)

# Comparar
y_pred_baseline = baseline_model.predict(X_val)
y_pred_etapa3 = pipeline_etapa3.predict(X_val)

print("=" * 50)
print("COMPARAÇÃO: BASELINE vs ETAPA 3")
print("=" * 50)
print(f"F1-Score (Baseline):  {f1_score(y_val, y_pred_baseline):.4f}")
print(f"F1-Score (ETAPA3):    {f1_score(y_val, y_pred_etapa3):.4f}")
print()
print(f"Precision (Baseline): {precision_score(y_val, y_pred_baseline):.4f}")
print(f"Precision (ETAPA3):   {precision_score(y_val, y_pred_etapa3):.4f}")
print()
print(f"Recall (Baseline):    {recall_score(y_val, y_pred_baseline):.4f}")
print(f"Recall (ETAPA3):      {recall_score(y_val, y_pred_etapa3):.4f}")
```

## 9. EXEMPLO 8: DEBUGGING E LOGGING

```python
from loguru import logger

# Habilitar logging detalhado
logger.enable("source.modeling.optuna_tuner")

# Executar com logs completos
pipeline.fit(X_train, y_train, X_val, y_val, tscv)

# Logs mostram:
# - Training progress de Isolation Forest
# - Trials de Optuna com F1-Score de cada trial
# - Best params encontrados
# - Modelo final treinado
```

## 10. EXEMPLO 9: AJUSTANDO PARA SEUS DADOS

```python
# Para dados MUY desbalanceados (>500:1)
pipeline = AMLTunerPipeline(
    model_type='xgboost',
    n_optuna_trials=100,        # Mais trials
    anomaly_contamination=0.02  # Menos anomalias detectadas
)

# Para dados bem balanceados (~1:1)
pipeline = AMLTunerPipeline(
    model_type='xgboost',
    n_optuna_trials=30,         # Menos trials suficientes
    anomaly_contamination=0.10  # Mais anomalias detectadas
)

# Para dados com tendências sazonais
tscv = TimeSeriesSplit(n_splits=10)  # Mais splits temporais
```

## 11. EXEMPLO 10: INTEGRAÇÃO COM BANCO DE DADOS

```python
# Ler dados do banco
import sqlalchemy as sa

engine = sa.create_engine("postgresql://user:pass@localhost/db")

X_train = pd.read_sql("SELECT * FROM features_train", engine)
X_val = pd.read_sql("SELECT * FROM features_val", engine)
X_test = pd.read_sql("SELECT * FROM features_test", engine)

# Treinar pipeline
pipeline.fit(X_train, y_train, X_val, y_val, tscv)

# Salvar predições de volta ao banco
predictions_df = pd.DataFrame({
    'id': X_test.index,
    'prediction': pipeline.predict(X_test),
    'probability': pipeline.predict_proba(X_test)[:, 1]
})

predictions_df.to_sql('predictions_etapa3', engine, if_exists='replace')
```

---

## ⚠️ OBSERVAÇÕES IMPORTANTES

1. **TimeSeriesSplit:** Sempre use para respeitar ordem temporal dos dados
2. **Anti-leakage:** Anomalias são detectadas APENAS no set de treino
3. **scale_pos_weight:** Automatically tuned no espaço [1, 100]
4. **Features:** Nova coluna 'anomaly_score' será adicionada automaticamente
5. **Reprodutibilidade:** Use `random_state=42` em tudo para resultados consistentes

---

## 🔗 REFERÊNCIAS

- **Arquivos:** `source/modeling/optuna_tuner.py`, `scripts/test_stage_3_tuning_anomaly.py`
- **Documentação:** `ETAPA_3_COMPLETADA.md`
- **Testes:** `scripts/test_stage_3_tuning_anomaly.py` (5/5 passando)

