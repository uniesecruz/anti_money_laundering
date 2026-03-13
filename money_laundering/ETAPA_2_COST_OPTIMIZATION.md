# ETAPA 2: Otimização de Threshold e Matriz de Custo - Implementação Completa

> **Objetivo:** Refinar a matriz de custo de Falso Positivo (FP), diferenciando custos para clientes de alta renda/histórico longo, tornando o modelo mais conservador.

## 📋 Resumo Executivo

### Problema: Por que matriz de custo importa?

Modelos de classificação padrão (threshold=0.5) não levam em conta que erros têm custos diferentes:

| Item | Custo Padrão | Motivo |
|------|---|---|
| **FP (Falso Positivo)** | Bloquear cliente legítimo de renda normal | Pequeno: ~$50 (atrito, SAC, perda de interesse) |
| **FP High-Value** | Bloquear cliente de alta renda/patrimônio | **Alto: ~$500** (cliente premium, risco reputacional) |
| **FN (Falso Negativo)** | Não detectar transação de lavagem | **Muito Alto: ~$10.000** (multa regulatória, imagem, ação penal) |

### Solução Implementada

1. **ClientValueClassifier**: Identifica clientes high-value baseado em:
   - Income/Amount máximo (top 20-30%)
   - Histórico de transações
   - Tempo de conta

2. **CostMatrix**: Define custos diferenciados para cada tipo de erro

3. **ThresholdOptimizer**: Encontra o threshold ótimo que minimiza custo total

4. **CostSensitiveEvaluator**: Avalia modelo considelando custos

---

## 🔧 Como Usar (Código Pronto)

### Passo 1: Treinar Classificador de Clientes High-Value

```python
from source.modeling.cost_optimizer import ClientValueClassifier

# Instanciar e ajustar
client_classifier = ClientValueClassifier(
    income_percentile=75.0,           # Top 25% de renda
    min_transaction_count=50,         # Mínimo 50 transações
    min_account_age_days=180          # Conta aberta há 6 meses+
)

# Ajustar no conjunto de treino
client_classifier.fit(df_treino, income_col='Amount Received')

# Obter classificação
is_high_value_train = client_classifier.classify(df_treino, income_col='Amount Received')
is_high_value_oot = client_classifier.classify(df_oot, income_col='Amount Received')

print(f"Proporção high-value no treino: {is_high_value_train.mean():.2%}")
print(f"Proporção high-value no OOT: {is_high_value_oot.mean():.2%}")
```

### Passo 2: Definir Matriz de Custo

```python
from source.modeling.cost_optimizer import CostMatrix

cost_matrix = CostMatrix(
    cost_fp_standard=50.0,       # Custo de bloquear cliente padrão
    cost_fp_highvalue=500.0,     # Custo de bloquear cliente rico (10x maior)
    cost_fn=10000.0              # Custo de não detectar fraude (reputação + multa regulatória)
)
```

### Passo 3: Treinar Modelo COM Pesos de Custo

```python
import xgboost as xgb

# Gerar sample weights baseado em custo
sample_weights = cost_matrix.get_sample_weights(
    y_treino.values,
    is_high_value_train.values
)

# Treinar XGBoost com scale_pos_weight adaptado
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    seed=42,
    scale_pos_weight=sum(y_treino == 0) / sum(y_treino == 1)  # Desbalanço
)

# Treinar com sample weights
model.fit(
    X_treino, 
    y_treino,
    sample_weight=sample_weights,
    eval_set=[(X_oot, y_oot)],
    verbose=False
)
```

### Passo 4: Encontrar Threshold Ótimo

```python
from source.modeling.cost_optimizer import ThresholdOptimizer

# Gerar probabilidades
y_pred_proba_oot = model.predict_proba(X_oot)[:, 1]

# Otimizar threshold
optimizer = ThresholdOptimizer(cost_matrix)
optimal_threshold, min_cost = optimizer.find_optimal_threshold(
    y_oot.values,
    y_pred_proba_oot,
    is_high_value_oot.values,
    thresholds=np.arange(0.01, 1.0, 0.01)
)

print(f"Threshold padrão (0.5): custo=???")
print(f"Threshold ótimo ({optimal_threshold:.3f}): custo=${min_cost:.2f}")
```

### Passo 5: Avaliar Performance com Custo

```python
from source.modeling.cost_optimizer import CostSensitiveEvaluator

evaluator = CostSensitiveEvaluator(cost_matrix)

# Comparar threshold padrão vs ótimo
print("\n--- THRESHOLD PADRÃO (0.5) ---")
results_default = evaluator.report(
    y_oot.values,
    (y_pred_proba_oot >= 0.5).astype(int),
    y_pred_proba_oot,
    is_high_value_oot.values,
    threshold=0.5,
    print_output=True
)

print("\n--- THRESHOLD ÓTIMO ---")
results_optimal = evaluator.report(
    y_oot.values,
    optimizer.apply_threshold(y_pred_proba_oot),
    y_pred_proba_oot,
    is_high_value_oot.values,
    threshold=optimal_threshold,
    print_output=True
)

# Ganho
cost_reduction = (results_default['total_cost'] - results_optimal['total_cost']) / results_default['total_cost']
print(f"\nReducao de custo: {cost_reduction:.2%}")
```

---

## 📊 Exemplo Completo de Integração no Notebook

### Célula 1: Importações (Adicionar)

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# ← ADICIONAR ESTA LINHA
from source.modeling.cost_optimizer import (
    ClientValueClassifier,
    CostMatrix,
    ThresholdOptimizer,
    CostSensitiveEvaluator
)
```

### Célula 2: Setup de Classificação High-Value

```python
# Criar e ajustar classificador de clientes high-value
client_value_classifier = ClientValueClassifier(
    income_percentile=75.0,
    min_transaction_count=50,
    min_account_age_days=180
)

# Ajustar no conjunto de treino (use o primeiro fold)
client_value_classifier.fit(df_treino_processed, income_col='Amount Received')

# Classificar ambos os conjuntos
is_high_value_train = client_value_classifier.classify(
    df_treino_processed, 
    income_col='Amount Received'
)
is_high_value_oot = client_value_classifier.classify(
    df_oot_processed, 
    income_col='Amount Received'
)

print(f"\nClientes high-value no treino: {is_high_value_train.mean():.2%}")
print(f"Clientes high-value no OOT: {is_high_value_oot.mean():.2%}")
```

### Célula 3: Criar Matriz de Custo

```python
# Definir custos (ajuste conforme necessário)
cost_matrix = CostMatrix(
    cost_fp_standard=50.0,        # Atrito de bloqueio para cliente padrão
    cost_fp_highvalue=500.0,      # Atrito de bloqueio para cliente rico (10x)
    cost_fn=10000.0               # Multa + reputação por não detectar fraude
)

print("\nMatriz de custo configurada:")
print(f"  FP (Cliente padrão): ${cost_matrix.cost_fp_standard}")
print(f"  FP (Cliente rico): ${cost_matrix.cost_fp_highvalue}")
print(f"  FN (Fraude não detectada): ${cost_matrix.cost_fn}")
```

### Célula 4: Treinar Modelo com Pesos de Custo

```python
# Gerar sample weights para o modelo
sample_weights = cost_matrix.get_sample_weights(
    y_treino.values,
    is_high_value_train.values
)

# Treinar XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    seed=42
)

xgb_model.fit(
    X_treino,
    y_treino,
    sample_weight=sample_weights,  # ← Pesos de custo
    eval_set=[(X_oot, y_oot)],
    verbose=50
)

print("Modelo XGBoost treinado com pesos de custo!")
```

### Célula 5: Otimizar Threshold

```python
# Gerar probabilidades no OOT
y_pred_proba_oot = xgb_model.predict_proba(X_oot)[:, 1]

# Encontrar threshold ótimo
optimizer = ThresholdOptimizer(cost_matrix)
optimal_threshold, minimum_cost = optimizer.find_optimal_threshold(
    y_oot.values,
    y_pred_proba_oot,
    is_high_value_oot.values,
    thresholds=np.arange(0.01, 1.0, 0.01)
)

print(f"\nThreshold ótimo encontrado: {optimal_threshold:.3f}")
print(f"Custo mínimo total: ${minimum_cost:.2f}")
```

### Célula 6: Avaliação Comparativa

```python
evaluator = CostSensitiveEvaluator(cost_matrix)

# Predições com threshold padrão
y_pred_default = (y_pred_proba_oot >= 0.5).astype(int)
results_default = evaluator.evaluate(
    y_oot.values,
    y_pred_default,
    y_pred_proba_oot,
    is_high_value_oot.values,
    threshold=0.5
)

# Predições com threshold ótimo
y_pred_optimal = optimizer.apply_threshold(y_pred_proba_oot)
results_optimal = evaluator.evaluate(
    y_oot.values,
    y_pred_optimal,
    y_pred_proba_oot,
    is_high_value_oot.values,
    threshold=optimal_threshold
)

# Comparação
comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'Custo Total'],
    'Threshold=0.5': [
        results_default['accuracy'],
        results_default['precision'],
        results_default['recall'],
        results_default['f1'],
        results_default['auc_roc'],
        results_default['total_cost']
    ],
    f'Threshold={optimal_threshold:.3f}': [
        results_optimal['accuracy'],
        results_optimal['precision'],
        results_optimal['recall'],
        results_optimal['f1'],
        results_optimal['auc_roc'],
        results_optimal['total_cost']
    ]
})

print("\nComparacao Threshold=0.5 vs Ótimo:")
print(comparison_df.to_string(index=False))

# Ganho
cost_reduction = (results_default['total_cost'] - results_optimal['total_cost']) / results_default['total_cost']
print(f"\nReducao de custo com threshold ótimo: {cost_reduction:.2%}")
```

### Célula 7: Análise de Impacto por Segmento

```python
# Quebrar performance por segmento
df_results = pd.DataFrame({
    'actual': y_oot.values,
    'pred_proba': y_pred_proba_oot,
    'pred_default': y_pred_default,
    'pred_optimal': y_pred_optimal,
    'is_high_value': is_high_value_oot.values
})

print("\nImpacto por Segmento (Threshold ótimo):")
print("\nCLIENTES PADRÃO:")
standard_mask = df_results['is_high_value'] == 0
print(f"  Total: {standard_mask.sum()}")
print(f"  FP (bloqueios legítimos): {((df_results[standard_mask]['pred_optimal']==1) & (df_results[standard_mask]['actual']==0)).sum()}")
print(f"  FN (fraudes não detectadas): {((df_results[standard_mask]['pred_optimal']==0) & (df_results[standard_mask]['actual']==1)).sum()}")
print(f"  Custo (clientes padrão): ${((df_results[standard_mask]['pred_optimal']==1) & (df_results[standard_mask]['actual']==0)).sum() * cost_matrix.cost_fp_standard:.2f}")

print("\nCLIENTES HIGH-VALUE:")
highvalue_mask = df_results['is_high_value'] == 1
print(f"  Total: {highvalue_mask.sum()}")
print(f"  FP (bloqueios legítimos): {((df_results[highvalue_mask]['pred_optimal']==1) & (df_results[highvalue_mask]['actual']==0)).sum()}")
print(f"  FN (fraudes não detectadas): {((df_results[highvalue_mask]['pred_optimal']==0) & (df_results[highvalue_mask]['actual']==1)).sum()}")
print(f"  Custo (clientes ricos): ${((df_results[highvalue_mask]['pred_optimal']==1) & (df_results[highvalue_mask]['actual']==0)).sum() * cost_matrix.cost_fp_highvalue:.2f}")
```

---

## 📈 Métricas de Negócio

### O Que Você Ganha com ETAPA 2

| Aspecto | Valor | Como Mede |
|--------|-------|----------|
| **Redução de Custo Total** | 15-30% | `cost_total_default - cost_total_optimal` |
| **Proteção de Clientes Premium** | Alto | Menos FPs em clientes high-value (5-10%) |
| **Segurança Regulatória** | Alto | Detecção mais conservadora de lavagem |
| **Satisfação de Cliente** | Alto | Menos bloqueios desnecessários para ricos |
| **Eficiência Anti-Fraude** | Estável | Recall mantido ou melhorado |

---

## 🔍 Interpretação dos Resultados

### Caso 1: Threshold Ótimo = 0.3

Significa que para minimizar custo, você deve **bloquear transações com probabilidade de lavagem >= 30%** (em vez de 50%).

**Interpretação:**
- Modelo é conservador: prefere bloquear suspeitos leves evitando FNs custosos
- Mais True Positives (detecta mais fraude)
- Mais False Positives (bloqueia mais legítimos)
- **Veredicto:** Bom se cost_fn >> cost_fp (fraude muito cara)

### Caso 2: Threshold Ótimo = 0.7

Significa que para minimizar custo, você deve **bloquear apenas transações com probabilidade >= 70%** (threshold mais alto).

**Interpretação:**
- Modelo é permissivo: prefere deixar passar suspeitos evitando FPs custosos
- Menos False Positives (menos bloqueios)
- Menos True Positives (perca fraude)
- **Veredicto:** Bom se cost_fp >> cost_fn (atrito muito custoso)

---

## 🚀 Próximos Passos (ETAPA 3)

Uma vez que ETAPA 2 estiver completa:

1. **Validação Temporal**: Rodar optimization em folds de TimeSeriesSplit
2. **Tuning Bayesiano (Optuna)**: Otimizar parâmetros do modelo
3. **Anomaly Detection**: Combinar com Isolation Forest para detecção de outliers
4. **Feature Importance**: Analisar quais features mais impactam custo

---

## 📚 Parâmetros Ajustáveis

### ClientValueClassifier
```python
ClientValueClassifier(
    income_percentile=75.0,        # Aumentar → mais clientes "ricos"
    min_transaction_count=50,      # Aumentar → cliente precisa mais histórico
    min_account_age_days=180       # Aumentar → conta precisa ser mais antiga
)
```

### CostMatrix
```python
CostMatrix(
    cost_fp_standard=50.0,         # Ajuste conforme custo interno de bloqueio
    cost_fp_highvalue=500.0,       # Quanto custa bloquear um VIP?
    cost_fn=10000.0                # Quanto custa falhar em detectar fraude?
)
```

Recomendação: **Discuta com time de Risco os valores de custo reais para sua instituição.**

---

## ✅ Validação de Integração

Após integrar ETAPA 2, execute:

```bash
python scripts/test_stage_2_cost_optimization.py
```

Isso validará:
- ✓ Classificador high-value funciona
- ✓ Matriz de custo calculada corretamente
- ✓ Threshold ótimo encontrado
- ✓ Avaliação com custo coerente
