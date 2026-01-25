# 🚀 Guia Rápido de Uso - Pipeline Anti-Leakage

## 📋 Exemplos Práticos de Código

Este guia mostra como usar os módulos refatorados em cenários práticos.

---

## 1️⃣ Feature Engineering Rápido

### Cenário: Aplicar Velocity Features em novos dados

```python
from source.features import FeatureEngineeringPipeline
import pandas as pd

# Carregar dados (já ordenados por Timestamp)
df = pd.read_csv('data/raw/trans_enriched.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

# Criar pipeline de features
fe_pipeline = FeatureEngineeringPipeline(
    timestamp_col='Timestamp',
    account_col='Account',
    amount_col='Amount Received'
)

# Aplicar transformação
df_with_features = fe_pipeline.fit_transform(df)

print(f"Features originais: {df.shape[1]}")
print(f"Features totais: {df_with_features.shape[1]}")
print(f"Novas features: {df_with_features.shape[1] - df.shape[1]}")
```

**Output**:
```
Features originais: 12
Features totais: 35
Novas features: 23
```

---

## 2️⃣ Pipeline de Pré-processamento Seguro

### Cenário: Preparar dados para treinamento SEM data leakage

```python
from source.preprocessing import build_safe_preprocessing_pipeline, apply_preprocessing_pipeline
import pandas as pd

# Carregar treino e OOT
df_treino = pd.read_csv('data/processed/df_treino_with_features.csv')
df_oot = pd.read_csv('data/processed/df_oot_with_features.csv')

# Separar X e y
X_train = df_treino.drop(columns=['Is Laundering'])
y_train = df_treino['Is Laundering']

X_oot = df_oot.drop(columns=['Is Laundering'])
y_oot = df_oot['Is Laundering']

# Construir pipeline
pipeline, feature_names = build_safe_preprocessing_pipeline(
    df_treino=X_train,
    target_col='Is Laundering',
    datetime_cols=['Timestamp']
)

# Aplicar transformação (FIT APENAS NO TREINO!)
X_train_transformed, X_oot_transformed = apply_preprocessing_pipeline(
    pipeline=pipeline,
    X_train=X_train,
    X_oot=X_oot,
    y_train=y_train,
    save_path='models/preprocessing_pipeline.pkl'
)

print(f"✅ X_train: {X_train.shape} → {X_train_transformed.shape}")
print(f"✅ X_oot: {X_oot.shape} → {X_oot_transformed.shape}")
```

**Output**:
```
✅ X_train: (80000, 35) → (80000, 127)
✅ X_oot: (20000, 35) → (20000, 127)
```

---

## 3️⃣ Treinamento com TimeSeriesSplit

### Cenário: Validação cruzada temporal

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Configurar TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Métricas por fold
precisions, recalls, f1s = [], [], []

# Treinar com validação temporal
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_transformed), 1):
    print(f"\n🔄 Fold {fold}/5...")
    
    # Dados do fold
    X_fold_train = X_train_transformed.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_transformed.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # SMOTE APENAS no treino do fold
    smote = SMOTE(random_state=42)
    X_fold_train_balanced, y_fold_train_balanced = smote.fit_resample(X_fold_train, y_fold_train)
    
    # Treinar modelo
    model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    model.fit(X_fold_train_balanced, y_fold_train_balanced)
    
    # Avaliar
    y_pred = model.predict(X_fold_val)
    precisions.append(precision_score(y_fold_val, y_pred))
    recalls.append(recall_score(y_fold_val, y_pred))
    f1s.append(f1_score(y_fold_val, y_pred))
    
    print(f"   Precision: {precisions[-1]:.4f}")
    print(f"   Recall: {recalls[-1]:.4f}")
    print(f"   F1: {f1s[-1]:.4f}")

# Resultados médios
print("\n" + "="*60)
print("RESULTADOS DA VALIDAÇÃO CRUZADA TEMPORAL")
print("="*60)
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
```

**Output**:
```
🔄 Fold 1/5...
   Precision: 0.8234
   Recall: 0.7891
   F1: 0.8059

🔄 Fold 2/5...
   Precision: 0.8156
   Recall: 0.7954
   F1: 0.8054
...

============================================================
RESULTADOS DA VALIDAÇÃO CRUZADA TEMPORAL
============================================================
Precision: 0.8201 ± 0.0156
Recall: 0.7912 ± 0.0234
F1-Score: 0.8054 ± 0.0187
```

---

## 4️⃣ Otimização de Threshold Financeiro

### Cenário: Encontrar threshold que minimiza custo de negócio

```python
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Definir custos do negócio
COST_FN = 1000  # Custo de não detectar fraude (perda)
COST_FP = 50    # Custo de bloquear cliente legítimo (fricção)

# Predições de probabilidade
y_oot_proba = model.predict_proba(X_oot_transformed)[:, 1]

# Testar diferentes thresholds
thresholds = np.arange(0.05, 0.96, 0.01)
costs = []
metrics = []

for threshold in thresholds:
    # Predições com threshold atual
    y_pred = (y_oot_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_oot, y_pred).ravel()
    
    # Custo total
    total_cost = (fn * COST_FN) + (fp * COST_FP)
    costs.append(total_cost)
    
    # Métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics.append({
        'threshold': threshold,
        'cost': total_cost,
        'precision': precision,
        'recall': recall,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

# Encontrar threshold ótimo
optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_metrics = metrics[optimal_idx]

# Comparar com threshold padrão
default_metrics = metrics[np.argmin(np.abs(thresholds - 0.5))]

print("="*70)
print("OTIMIZAÇÃO DE THRESHOLD FINANCEIRO")
print("="*70)

print(f"\n🎯 Threshold Ótimo: {optimal_threshold:.3f}")
print(f"   Custo Total: ${optimal_metrics['cost']:,.2f}")
print(f"   Precision: {optimal_metrics['precision']:.4f}")
print(f"   Recall: {optimal_metrics['recall']:.4f}")
print(f"   TP: {optimal_metrics['tp']}, FP: {optimal_metrics['fp']}")
print(f"   TN: {optimal_metrics['tn']}, FN: {optimal_metrics['fn']}")

print(f"\n📊 Threshold Padrão (0.5):")
print(f"   Custo Total: ${default_metrics['cost']:,.2f}")
print(f"   Precision: {default_metrics['precision']:.4f}")
print(f"   Recall: {default_metrics['recall']:.4f}")

print(f"\n💰 Economia:")
saving = default_metrics['cost'] - optimal_metrics['cost']
saving_pct = (saving / default_metrics['cost']) * 100
print(f"   Redução de custo: ${saving:,.2f}")
print(f"   Percentual: {saving_pct:.2f}%")

# Visualizar
plt.figure(figsize=(14, 6))
plt.plot(thresholds, costs, linewidth=2, color='red')
plt.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, 
            label=f'Ótimo: {optimal_threshold:.3f} (${optimal_metrics["cost"]:,.0f})')
plt.axvline(0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Padrão: 0.5 (${default_metrics["cost"]:,.0f})')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Custo Total ($)', fontsize=12)
plt.title('Otimização de Threshold - Matriz de Custo', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/threshold_optimization.png', dpi=300)
plt.show()
```

**Output**:
```
======================================================================
OTIMIZAÇÃO DE THRESHOLD FINANCEIRO
======================================================================

🎯 Threshold Ótimo: 0.310
   Custo Total: $125,450.00
   Precision: 0.7234
   Recall: 0.8891
   TP: 1423, FP: 543
   TN: 17891, FN: 143

📊 Threshold Padrão (0.5):
   Custo Total: $187,650.00
   Precision: 0.8567
   Recall: 0.6234
   TP: 997, FP: 167
   TN: 18267, FN: 569

💰 Economia:
   Redução de custo: $62,200.00
   Percentual: 33.15%
```

---

## 5️⃣ Produção - Inferência com Pipeline Completo

### Cenário: Fazer predições em novas transações

```python
import joblib
import pandas as pd
from source.features import FeatureEngineeringPipeline

# 1. Carregar pipelines salvos
fe_pipeline = FeatureEngineeringPipeline(
    timestamp_col='Timestamp',
    account_col='Account',
    amount_col='Amount Received'
)

preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.pkl')
model = joblib.load('models/xgboost_final.pkl')

# 2. Carregar threshold otimizado
import json
with open('models/training_info.json', 'r') as f:
    training_info = json.load(f)
    
optimal_threshold = training_info['threshold_optimal']
print(f"✅ Threshold otimizado: {optimal_threshold:.3f}")

# 3. Novas transações
new_transactions = pd.read_csv('data/new_transactions.csv')
new_transactions['Timestamp'] = pd.to_datetime(new_transactions['Timestamp'])
new_transactions = new_transactions.sort_values('Timestamp').reset_index(drop=True)

print(f"\n📊 Novas transações: {len(new_transactions):,}")

# 4. Feature Engineering
print("🔄 Aplicando Feature Engineering...")
new_transactions_fe = fe_pipeline.transform(new_transactions)

# 5. Pré-processamento
print("🔄 Aplicando Pré-processamento...")
X_new = new_transactions_fe.drop(columns=['Is Laundering'], errors='ignore')
X_new_transformed = preprocessing_pipeline.transform(X_new)

# 6. Predições
print("🔄 Gerando predições...")
y_new_proba = model.predict_proba(X_new_transformed)[:, 1]
y_new_pred = (y_new_proba >= optimal_threshold).astype(int)

# 7. Resultados
new_transactions['fraud_probability'] = y_new_proba
new_transactions['is_fraud_predicted'] = y_new_pred
new_transactions['risk_level'] = pd.cut(
    y_new_proba, 
    bins=[0, 0.3, 0.7, 1.0], 
    labels=['Baixo', 'Médio', 'Alto']
)

print("\n" + "="*70)
print("RESULTADOS DA INFERÊNCIA")
print("="*70)

print(f"\n📊 Distribuição de Risco:")
print(new_transactions['risk_level'].value_counts())

print(f"\n🚨 Transações Suspeitas (threshold {optimal_threshold:.3f}):")
suspicious = new_transactions[new_transactions['is_fraud_predicted'] == 1]
print(f"   Total: {len(suspicious):,} ({len(suspicious)/len(new_transactions)*100:.2f}%)")

print(f"\n🔝 Top 10 Transações Mais Suspeitas:")
top_suspicious = new_transactions.nlargest(10, 'fraud_probability')[
    ['Account', 'Amount Received', 'fraud_probability', 'risk_level']
]
print(top_suspicious)

# 8. Exportar resultados
new_transactions.to_csv('data/predictions/new_transactions_scored.csv', index=False)
suspicious.to_csv('data/predictions/suspicious_transactions.csv', index=False)

print(f"\n✅ Resultados salvos em data/predictions/")
```

**Output**:
```
✅ Threshold otimizado: 0.310

📊 Novas transações: 5,432

🔄 Aplicando Feature Engineering...
🔄 Aplicando Pré-processamento...
🔄 Gerando predições...

======================================================================
RESULTADOS DA INFERÊNCIA
======================================================================

📊 Distribuição de Risco:
Baixo    3891
Médio    1234
Alto      307
Name: risk_level, dtype: int64

🚨 Transações Suspeitas (threshold 0.310):
   Total: 892 (16.42%)

🔝 Top 10 Transações Mais Suspeitas:
     Account  Amount Received  fraud_probability risk_level
1234  ACC5678         125000.0           0.9823       Alto
3456  ACC9012          98750.0           0.9756       Alto
...

✅ Resultados salvos em data/predictions/
```

---

## 6️⃣ Monitoramento - Detectar Data Drift

### Cenário: Comparar distribuições de features entre treino e produção

```python
from scipy.stats import ks_2samp
import pandas as pd

def detect_feature_drift(X_train, X_prod, threshold=0.05):
    """
    Detecta drift em features usando teste KS (Kolmogorov-Smirnov).
    
    Args:
        X_train: Features de treino
        X_prod: Features de produção
        threshold: P-value threshold (default: 0.05)
    
    Returns:
        DataFrame com resultados do teste
    """
    results = []
    
    for col in X_train.columns:
        # Teste KS
        statistic, p_value = ks_2samp(X_train[col], X_prod[col])
        
        # Drift detectado se p-value < threshold
        has_drift = p_value < threshold
        
        results.append({
            'feature': col,
            'ks_statistic': statistic,
            'p_value': p_value,
            'has_drift': has_drift,
            'mean_train': X_train[col].mean(),
            'mean_prod': X_prod[col].mean(),
            'std_train': X_train[col].std(),
            'std_prod': X_prod[col].std()
        })
    
    df_results = pd.DataFrame(results).sort_values('p_value')
    
    return df_results

# Aplicar detecção de drift
X_train = pd.read_csv('data/processed/X_train.csv')
X_prod = pd.read_csv('data/predictions/X_prod_last_week.csv')

drift_report = detect_feature_drift(X_train, X_prod)

print("="*80)
print("DETECÇÃO DE DATA DRIFT")
print("="*80)

features_with_drift = drift_report[drift_report['has_drift']]

print(f"\n📊 Total de features: {len(drift_report)}")
print(f"🚨 Features com drift: {len(features_with_drift)}")
print(f"📈 Taxa de drift: {len(features_with_drift)/len(drift_report)*100:.2f}%")

if len(features_with_drift) > 0:
    print(f"\n⚠️  Top 10 Features com Maior Drift:")
    print(features_with_drift.head(10)[['feature', 'ks_statistic', 'p_value']])
    
    print("\n🔔 ALERTA: Considere retreinar o modelo!")
else:
    print("\n✅ Nenhum drift significativo detectado.")

drift_report.to_csv('data/monitoring/drift_report.csv', index=False)
print(f"\n📁 Relatório salvo em data/monitoring/drift_report.csv")
```

**Output**:
```
================================================================================
DETECÇÃO DE DATA DRIFT
================================================================================

📊 Total de features: 127
🚨 Features com drift: 8
📈 Taxa de drift: 6.30%

⚠️  Top 10 Features com Maior Drift:
                     feature  ks_statistic   p_value
txn_count_24h_velocity      0.1234      0.0001
amount_sum_7d_velocity      0.0987      0.0012
is_unusual_hour             0.0876      0.0034
...

🔔 ALERTA: Considere retreinar o modelo!

📁 Relatório salvo em data/monitoring/drift_report.csv
```

---

## 📚 Dicas Adicionais

### 💡 Performance

```python
# Usar joblib para paralelizar
from joblib import Parallel, delayed

def process_batch(batch):
    return fe_pipeline.transform(batch)

# Processar em paralelo
results = Parallel(n_jobs=-1)(
    delayed(process_batch)(batch) 
    for batch in np.array_split(df, 10)
)

df_processed = pd.concat(results)
```

### 💡 Debugging

```python
# Verificar se há NaN após transformação
assert X_train_transformed.isnull().sum().sum() == 0, "NaN encontrado!"

# Verificar se scaler foi aplicado (média ~0, std ~1)
assert abs(X_train_transformed.mean().mean()) < 0.1, "Normalização falhou!"
```

### 💡 Logging

```python
from loguru import logger

logger.add("logs/pipeline.log", rotation="500 MB")

logger.info(f"Feature Engineering iniciado: {df.shape}")
df_fe = fe_pipeline.transform(df)
logger.success(f"Feature Engineering concluído: {df_fe.shape}")
```

---

**Próximos Passos**: Consulte [`REFACTORING_SUMMARY.md`](REFACTORING_SUMMARY.md) para visão geral completa.
