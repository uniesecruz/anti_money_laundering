# 🎯 Guia Rápido: Como Integrar ETAPA 2 no Notebook

> **5 minutos para integrar Cost Optimization no seu notebook de treino**

---

## ⚡ TL;DR (30 segundos)

1. Abra `notebooks/08_treinamento_timeseriesplit.ipynb`
2. Copie o **Bloco 1-7** do arquivo `ETAPA_2_CODIGO_PRONTO.md`
3. Cole nas células apropriadas do notebook
4. Execute e aproveite os resultados!

---

## 📍 Passo a Passo Detalhado

### PASSO 1: Abrir o Notebook

```
Arquivo: notebooks/08_treinamento_timeseriesplit.ipynb
```

### PASSO 2: Adicionar Importações

**Onde?** Na primeira célula de imports (topo do notebook)

**O que copiar?** Bloco 1 de `ETAPA_2_CODIGO_PRONTO.md`:

```python
from source.modeling.cost_optimizer import (
    ClientValueClassifier,
    CostMatrix,
    ThresholdOptimizer,
    CostSensitiveEvaluator,
    create_cost_sensitive_training_data
)
```

**Onde colar?** Adicione ao final da célula de imports existente

---

### PASSO 3: Classificar Clientes High-Value

**Quando?** Depois de carregar os dados (X_treino, X_oot, y_treino, y_oot)

**Onde?** Crie uma nova célula após carregar dados

**O que copiar?** Bloco 2 de `ETAPA_2_CODIGO_PRONTO.md`

```python
# ETAPA 2: OTIMIZACAO DE THRESHOLD E CUSTOS
print("\n[1/4] Classificando clientes high-value...")

client_value_classifier = ClientValueClassifier(
    income_percentile=75.0,
    min_transaction_count=50,
    min_account_age_days=180
)

client_value_classifier.fit(df_treino, income_col='Amount Received')

is_high_value_train = client_value_classifier.classify(
    df_treino, 
    income_col='Amount Received'
)
is_high_value_oot = client_value_classifier.classify(
    df_oot, 
    income_col='Amount Received'
)

print(f"Clientes high-value no treino: {is_high_value_train.mean():.2%}")
print(f"Clientes high-value no OOT: {is_high_value_oot.mean():.2%}")
```

---

### PASSO 4: Configurar Matriz de Custo

**Quando?** Imediatamente após PASSO 3

**O que copiar?** Bloco 3 de `ETAPA_2_CODIGO_PRONTO.md`

```python
print("\n[2/4] Configurando matriz de custo...")

cost_matrix = CostMatrix(
    cost_fp_standard=50.0,
    cost_fp_highvalue=500.0,
    cost_fn=10000.0
)
```

**Nota:** Ajuste os valores conforme a realidade de sua instituição:
- `cost_fp_standard`: Quanto custa bloquear cliente padrão?
- `cost_fp_highvalue`: Quanto custa bloquear cliente VIP?
- `cost_fn`: Quanto custa não detectar fraude?

---

### PASSO 5: Treinar Modelo com Pesos de Custo

**Quando?** Antes de chamar `model.fit()`

**O que copiar?** Bloco 4 de `ETAPA_2_CODIGO_PRONTO.md`

```python
print("\n[3/4] Preparando dados com pesos de custo...")

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
    eval_metric='logloss'
)

xgb_model.fit(
    X_treino,
    y_treino,
    sample_weight=sample_weights,  # ← CRUCIAL: Pesos de custo
    eval_set=[(X_oot, y_oot)],
    verbose=50
)
```

**Importante:** 
- Substitua `xgb_model` pelo nome do seu modelo se for diferente
- Use o mesmo `sample_weight` para qualquer modelo (XGBoost, LightGBM, etc)

---

### PASSO 6: Otimizar Threshold

**Quando?** Depois de treinar o modelo

**O que copiar?** Bloco 5 de `ETAPA_2_CODIGO_PRONTO.md`

```python
print("\n[4/4] Otimizando threshold...")

y_pred_proba_oot = xgb_model.predict_proba(X_oot)[:, 1]

optimizer = ThresholdOptimizer(cost_matrix)
optimal_threshold, minimum_cost = optimizer.find_optimal_threshold(
    y_oot.values,
    y_pred_proba_oot,
    is_high_value_oot.values,
    thresholds=np.arange(0.01, 1.0, 0.01)
)

print(f"Threshold ótimo: {optimal_threshold:.3f}")
print(f"Custo mínimo: ${minimum_cost:.2f}")
```

**Resultado esperado:**
```
Threshold ótimo encontrado: 0.410
Custo total mínimo: $51700.00
```

---

### PASSO 7: Comparar Resultados (RECOMENDADO)

**Quando?** Para validar que o novo threshold é realmente melhor

**O que copiar?** Bloco 6 de `ETAPA_2_CODIGO_PRONTO.md`

```python
print("\nCOMPARACAO: Threshold=0.5 vs Ótimo")
print("=" * 80)

evaluator = CostSensitiveEvaluator(cost_matrix)

# Threshold padrão
y_pred_default = (y_pred_proba_oot >= 0.5).astype(int)
results_default = evaluator.evaluate(
    y_oot.values,
    y_pred_default,
    y_pred_proba_oot,
    is_high_value_oot.values,
    threshold=0.5
)

# Threshold ótimo
y_pred_optimal = optimizer.apply_threshold(y_pred_proba_oot)
results_optimal = evaluator.evaluate(
    y_oot.values,
    y_pred_optimal,
    y_pred_proba_oot,
    is_high_value_oot.values,
    threshold=optimal_threshold
)

# Tabela
comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'Custo Total'],
    'Threshold=0.5': [
        f"{results_default['accuracy']:.4f}",
        f"{results_default['precision']:.4f}",
        f"{results_default['recall']:.4f}",
        f"${results_default['total_cost']:.2f}"
    ],
    f'Threshold={optimal_threshold:.3f}': [
        f"{results_optimal['accuracy']:.4f}",
        f"{results_optimal['precision']:.4f}",
        f"{results_optimal['recall']:.4f}",
        f"${results_optimal['total_cost']:.2f}"
    ]
})

print(comparison_df.to_string(index=False))

cost_reduction = results_default['total_cost'] - results_optimal['total_cost']
print(f"\nEconomia: ${cost_reduction:.2f}")
```

**Resultado esperado:**
```
                Métrica  Threshold=0.5  Threshold=0.410
                Accuracy       0.9500       0.9525
                Precision      0.3067       0.1853
                   Recall      0.9200       0.9600
            Custo Total        $60500       $51700

Economia: $8800
```

---

### OPCIONAL: PASSO 8 - Análise por Segmento

**Quando?** Se você quer entender o impacto em cada tipo de cliente

**O que copiar?** Bloco 7 de `ETAPA_2_CODIGO_PRONTO.md`

```python
print("ANALISE POR SEGMENTO")
print("=" * 80)

df_analysis = pd.DataFrame({
    'actual': y_oot.values,
    'pred_optimal': y_pred_optimal,
    'is_high_value': is_high_value_oot.values
})

# Clientes padrão
standard_mask = df_analysis['is_high_value'] == 0
fp_std = ((df_analysis[standard_mask]['pred_optimal']==1) & 
          (df_analysis[standard_mask]['actual']==0)).sum()

print(f"CLIENTES PADRÃO:")
print(f"  FP (bloqueios indevidos): {fp_std}")
print(f"  Custo: ${fp_std * cost_matrix.cost_fp_standard:.2f}")

# Clientes high-value
hv_mask = df_analysis['is_high_value'] == 1
fp_hv = ((df_analysis[hv_mask]['pred_optimal']==1) & 
         (df_analysis[hv_mask]['actual']==0)).sum()

print(f"CLIENTES HIGH-VALUE:")
print(f"  FP (bloqueios indevidos): {fp_hv}")
print(f"  Custo: ${fp_hv * cost_matrix.cost_fp_highvalue:.2f}")
```

---

## 📋 Checklist de Integração

```
☐ PASSO 1: Abriu o notebook 08_treinamento_timeseriesplit.ipynb
☐ PASSO 2: Adicionou imports do cost_optimizer
☐ PASSO 3: Classificou clientes high-value (proporção OK? 10-30%)
☐ PASSO 4: Configurou CostMatrix com valores realistas
☐ PASSO 5: Treinou modelo COM sample_weight (não esqueceu!)
☐ PASSO 6: Otimizou threshold (encontrou valor < 0.5?)
☐ PASSO 7: Comparou resultados (economia > 0?)
☐ PASSO 8: (Opcional) Analisou segmentos de clientes
☐ VALIDAÇÃO: Executou python scripts/test_stage_2_cost_optimization.py
```

---

## 🐛 Troubleshooting

### Problema: ImportError (source.modeling.cost_optimizer não encontrado)

**Solução:** Verifique se o arquivo `source/modeling/cost_optimizer.py` existe

```bash
ls -la source/modeling/cost_optimizer.py
```

Se não existir, você provavelmente não puxou o git commit. Execute:
```bash
git pull
```

---

### Problema: "KeyError: 'Amount Received'" ao classificar clientes

**Solução:** Sua coluna de valor tem outro nome. Descubra com:

```python
print(df_treino.columns)
```

Depois ajuste:
```python
# Em vez de:
client_value_classifier.fit(df_treino, income_col='Amount Received')

# Use:
client_value_classifier.fit(df_treino, income_col='seu_nome_da_coluna')
```

---

### Problema: Threshold ótimo == 0.01 (muito baixo)

**Possible Causes:**
1. `cost_fn` muito alto comparado a `cost_fp`
2. Modelo com muitos FNs

**Solução:** Aumentar `cost_fp_standard` e `cost_fp_highvalue`

```python
# Em vez de:
cost_matrix = CostMatrix(cost_fp_standard=50, cost_fp_highvalue=500, cost_fn=10000)

# Tente:
cost_matrix = CostMatrix(cost_fp_standard=500, cost_fp_highvalue=5000, cost_fn=10000)
```

---

### Problema: Callback de early stopping conflita com sample_weight

**Solução:** Use dessa forma:

```python
xgb_model.fit(
    X_treino,
    y_treino,
    sample_weight=sample_weights,
    eval_set=[(X_oot, y_oot)],
    eval_metric='logloss',
    callbacks=[xgb.callback.EarlyStopping(rounds=10, metric_name='logloss')]
)
```

---

## 📞 Perguntas Comuns

**P: Preciso usar sample_weight?**
R: Tecnicamente não, mas sem weights você não aproveitará a matriz de custo. O modelo treina igual, mas não é cost-aware.

**P: Posso usar isto com LightGBM?**
R: Sim! Só mude para `lgb.LGBMClassifier()` e use o mesmo `sample_weight`.

**P: Meu threshold ótimo mudou muito entre treino e OOT. É normal?**
R: Um pouco é normal (dados diferentes). Muito (ex: 0.3 vs 0.7) pode indicate:
- Modelo instável
- Distribuição muito diferente entre treino e OOT
- Ajustar custos pode ajudar

**P: Posso guardar o `optimizer` para usar depois em produção?**
R: Sim! Com pickle:
```python
import pickle
pickle.dump(optimizer, open('optimal_threshold.pkl', 'wb'))

# Depois:
optimizer = pickle.load(open('optimal_threshold.pkl', 'rb'))
y_pred = optimizer.apply_threshold(y_pred_proba)
```

---

## 🎓 Para Entender Melhor

- Leia `ETAPA_2_SUMMARY.md` para conceitos
- Leia `ETAPA_2_COST_OPTIMIZATION.md` para detalhes técnicos
- Execute `python scripts/test_stage_2_cost_optimization.py` para validar

---

## ✅ Você Está Pronto!

Siga os PASSOS 1-7 acima e terá Cost Optimization rodando em 10 minutos.

Perguntas? Verifique a seção **Troubleshooting** acima.

**Próximo passo:** ETAPA 3 (Bayesian Optimization + Anomaly Detection)

🚀
