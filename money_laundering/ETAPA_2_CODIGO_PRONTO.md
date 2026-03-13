# ETAPA 2 - Código Pronto para Integração no Notebook

> **Blocos de código prontos paracopiar e colar no notebook `08_treinamento_timeseriesplit.ipynb`**

---

## Bloco 1: Importações (Adicionar ao topo)

Adicione estas importações na célula de imports do notebook (junto com as outras imports):

```python
from source.modeling.cost_optimizer import (
    ClientValueClassifier,
    CostMatrix,
    ThresholdOptimizer,
    CostSensitiveEvaluator,
    create_cost_sensitive_training_data
)
```

---

## Bloco 2: Classificador High-Value

Cole esta célula **depois de carregar os dados** (X_treino, X_oot, y_treino, y_oot):

```python
# ============================================================================
# ETAPA 2: OTIMIZACAO DE THRESHOLD E CUSTOS
# ============================================================================

print("\n" + "="*80)
print("ETAPA 2: Otimizacao de Threshold e Matriz de Custo")
print("="*80)

# Criar e ajustar classificador de clientes high-value
print("\n[1/4] Classificando clientes high-value...")

client_value_classifier = ClientValueClassifier(
    income_percentile=75.0,           # Top 25% de renda
    min_transaction_count=50,         # Mínimo 50 transações
    min_account_age_days=180          # Conta aberta há 6 meses+
)

# Ajustar no conjunto de treino
client_value_classifier.fit(df_treino, income_col='Amount Received')

# Classificar ambos os conjuntos
is_high_value_train = client_value_classifier.classify(
    df_treino, 
    income_col='Amount Received'
)
is_high_value_oot = client_value_classifier.classify(
    df_oot, 
    income_col='Amount Received'
)

print(f"  Clientes high-value no treino: {is_high_value_train.mean():.2%}")
print(f"  Clientes high-value no OOT: {is_high_value_oot.mean():.2%}")
```

---

## Bloco 3: Criar Matriz de Custo

Cole esta célula **imediatamente após** o Bloco 2:

```python
# Definir matriz de custo
print("\n[2/4] Configurando matriz de custo...")

cost_matrix = CostMatrix(
    cost_fp_standard=50.0,        # Custo de bloquear cliente padrão
    cost_fp_highvalue=500.0,      # Custo de bloquear cliente rico (10x)
    cost_fn=10000.0               # Custo de fraude não detectada
)
```

---

## Bloco 4: Treinar Modelo com Pesos de Custo (XGBoost)

Cole esta célula **antes de treinar o modelo XGBoost** (quando você tem X_treino, y_treino):

```python
# Gerar sample weights baseado em matriz de custo
print("\n[3/4] Preparando dados de treino com pesos de custo...")

sample_weights = cost_matrix.get_sample_weights(
    y_treino.values,
    is_high_value_train.values
)

print(f"  Peso médio (positivos): {sample_weights[y_treino == 1].mean():.4f}")
print(f"  Peso médio (negativos): {sample_weights[y_treino == 0].mean():.4f}")
print(f"  Razão de pesos (P/N): {sample_weights[y_treino == 1].mean() / sample_weights[y_treino == 0].mean():.2f}x")

# Treinar XGBoost COM sample weights
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(
    X_treino,
    y_treino,
    sample_weight=sample_weights,  # ← Pesos de custo
    eval_set=[(X_oot, y_oot)],
    verbose=50
)

print("  Modelo XGBoost treinado com pesos de custo!")
```

---

## Bloco 5: Otimizar Threshold

Cole esta célula **depois de treinar o modelo**:

```python
# Otimizar threshold para minimizar custo
print("\n[4/4] Otimizando threshold para minimizar custo...")

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

print(f"  Threshold ótimo encontrado: {optimal_threshold:.3f}")
print(f"  Custo mínimo total: ${minimum_cost:.2f}")
```

---

## Bloco 6: Avaliação Comparativa (Recomendado)

Cole esta célula **após otimizar** para comparar performance:

```python
# Avaliar e comparar threshold padrão vs ótimo
print("\n" + "="*80)
print("COMPARACAO: Threshold Padrão (0.5) vs Ótimo")
print("="*80)

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

# Tabela de comparação
comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'Custo Total (em $)'],
    'Threshold=0.5': [
        f"{results_default['accuracy']:.4f}",
        f"{results_default['precision']:.4f}",
        f"{results_default['recall']:.4f}",
        f"{results_default['f1']:.4f}",
        f"{results_default['auc_roc']:.4f}",
        f"${results_default['total_cost']:.2f}"
    ],
    f'Threshold={optimal_threshold:.3f}': [
        f"{results_optimal['accuracy']:.4f}",
        f"{results_optimal['precision']:.4f}",
        f"{results_optimal['recall']:.4f}",
        f"{results_optimal['f1']:.4f}",
        f"{results_optimal['auc_roc']:.4f}",
        f"${results_optimal['total_cost']:.2f}"
    ]
})

print("\n")
print(comparison_df.to_string(index=False))

# Ganho absoluto
cost_reduction = results_default['total_cost'] - results_optimal['total_cost']
cost_reduction_pct = cost_reduction / max(results_default['total_cost'], 1)

print(f"\nReducao de custo: ${cost_reduction:.2f} ({cost_reduction_pct:.2%})")
print(f"Economia para cada 1000 transacoes: ${cost_reduction * 1000 / len(y_oot):.2f}")
```

---

## Bloco 7: Avaliação por Segmento (Opcional)

Cole esta célula **para entender impacto em cada segmento**:

```python
# Análise detalhada por segmento
print("\n" + "="*80)
print("ANALISE POR SEGMENTO DE CLIENTE")
print("="*80)

df_analysis = pd.DataFrame({
    'actual': y_oot.values,
    'pred_optimal': y_pred_optimal,
    'is_high_value': is_high_value_oot.values
})

print("\nCLIENTES PADRAO (Bottom 75%):")
standard_mask = df_analysis['is_high_value'] == 0
total_standard = standard_mask.sum()
fp_standard = ((df_analysis[standard_mask]['pred_optimal']==1) & 
               (df_analysis[standard_mask]['actual']==0)).sum()
fn_standard = ((df_analysis[standard_mask]['pred_optimal']==0) & 
               (df_analysis[standard_mask]['actual']==1)).sum()
tp_standard = ((df_analysis[standard_mask]['pred_optimal']==1) & 
               (df_analysis[standard_mask]['actual']==1)).sum()

print(f"  Total de transacoes: {total_standard}")
print(f"  True Positives (fraudes detectadas): {tp_standard}")
print(f"  False Positives (bloqueios indevidos): {fp_standard}")
print(f"  False Negatives (fraudes nao detectadas): {fn_standard}")
print(f"  Custo em FPs: ${fp_standard * cost_matrix.cost_fp_standard:.2f}")
print(f"  Custo em FNs: ${fn_standard * cost_matrix.cost_fn:.2f}")

print("\nCLIENTES HIGH-VALUE (Top 25%):")
highvalue_mask = df_analysis['is_high_value'] == 1
total_highvalue = highvalue_mask.sum()
fp_highvalue = ((df_analysis[highvalue_mask]['pred_optimal']==1) & 
                (df_analysis[highvalue_mask]['actual']==0)).sum()
fn_highvalue = ((df_analysis[highvalue_mask]['pred_optimal']==0) & 
                (df_analysis[highvalue_mask]['actual']==1)).sum()
tp_highvalue = ((df_analysis[highvalue_mask]['pred_optimal']==1) & 
                (df_analysis[highvalue_mask]['actual']==1)).sum()

print(f"  Total de transacoes: {total_highvalue}")
print(f"  True Positives (fraudes detectadas): {tp_highvalue}")
print(f"  False Positives (bloqueios indevidos): {fp_highvalue}")
print(f"  False Negatives (fraudes nao detectadas): {fn_highvalue}")
print(f"  Custo em FPs: ${fp_highvalue * cost_matrix.cost_fp_highvalue:.2f}")
print(f"  Custo em FNs: ${fn_highvalue * cost_matrix.cost_fn:.2f}")

print("\n" + "="*80)
```

---

## Bloco 8: Aplicar Threshold no Notebook Inteiro (Para Deploy)

Após validar tudo, use este código para aplicar o threshold ótimo a novos dados:

```python
# Para usar o modelo em produção com threshold ótimo
def predict_with_optimal_threshold(X, model, optimizer, y_pred_proba=None):
    """
    Faz predição usando o threshold ótimo já calculado.
    """
    if y_pred_proba is None:
        y_pred_proba = model.predict_proba(X)[:, 1]
    
    y_pred = optimizer.apply_threshold(y_pred_proba)
    return y_pred

# Usar assim:
y_pred_new = predict_with_optimal_threshold(X_novo, xgb_model, optimizer)
```

---

## Checklist de Integração

- [ ] Adicionou importações (Bloco 1)
- [ ] Criou classificador high-value (Bloco 2)
- [ ] Configurou matriz de custo (Bloco 3)
- [ ] Treinou modelo com pesos (Bloco 4)
- [ ] Otimizou threshold (Bloco 5)
- [ ] Validou threshold com comparação (Bloco 6)
- [ ] (Opcional) Analisou por segmento (Bloco 7)
- [ ] Executou: `python scripts/test_stage_2_cost_optimization.py` com sucesso

---

## Próximas Etapas

Após integrar ETAPA 2:

1. **Validação em TimeSeriesSplit**: Rodar optimization em cada fold temporal
2. **ETAPA 3**: Tuning Bayesiano com Optuna + Isolation Forest
3. **ETAPA 4**: Aplicar transformações Yeo-Johnson + Target Encoding
4. **ETAPA 5**: Implementar Precision@Top-K + PSI monitoring

