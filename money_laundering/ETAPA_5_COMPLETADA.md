# ETAPA 5: MÉTRICAS DE NEGÓCIO E ESTABILIDADE - CONCLUÍDA

## Status: ✅ Implementação Completa (6/6 Testes Passando)

Data: 2026-03-13  
Objetivos: Adicionar métricas de negócio (Precision@Top-K) e estabilidade (PSI)

---

## 1. Implementação Técnica

### 1.1 Precision@Top-K

**Local:** `source/modeling/metrics.py` (linhas ~50-115)

**Responsabilidade:** Medir precisão ao selecionar os top-K casos de maior suspeita

**Use Case:**
- Investigadores de AML investigam manualmente os top-100 a top-500 casos
- Pergunta crítica: Desses top-K, quantos são realmente suspeitos?
- Impacto: Define eficiência operacional do time de investigação

**Método: `precision_at_top_k(y_true, y_scores, top_k_list=[100, 500])`**

```python
# Entrada
y_true  = [1, 0, 1, 1, 0, ...]  # Labels reais
y_scores = [0.95, 0.85, 0.70, ...] # Scores do modelo (0-1)
top_k_list = [100, 500]

# Processo
1. Ordenar y_scores em ordem DECRESCENTE (top scores primeiro)
2. Para cada K:
   - Selecionar top-K labels
   - Calcular Precision@K = TP / K
   
# Saída
{
    100: 0.75,  # 75% dos top-100 são realmente positivos
    500: 0.68   # 68% dos top-500 são realmente positivos
}
```

**Interpretação:**
- Precision@100 = 0.75 → 75 verdadeiros positivos em top-100
  - Bom: Modelo está ranqueando bem os casos
  - Investigadores gastam 75% do tempo em casos reais
  
- Precision@100 = 0.50 → 50 verdadeiros positivos em top-100
  - Ruim: Modelo está ranqueando mal
  - Investigadores gastam 50% do tempo em falsos alarmes

**Validações de Teste:**
- ✅ Teste 1: Cálculo correto para K=1, K=3, K=5
- ✅ Teste 2: Edge cases (K > total samples, todos negativos, K=0)

---

### 1.2 PSI (Population Stability Index)

**Local:** `source/modeling/metrics.py` (linhas ~115-220)

**Responsabilidade:** Monitorar se distribuição de scores mudou entre treino e OOT

**Use Case:**
- Modelo treinado em dados de 2024, aplicado em 2025
- Padrões criminosos evoluem, dados mudam
- Pergunta crítica: Modelo ainda está calibrado? Precisa refit?

**Fórmula PSI:**
```
PSI = SUM( (OOT_pct - Train_pct) * ln(OOT_pct / Train_pct) )

onde:
  OOT_pct = proporção de exemplos em cada bin (OOT)
  Train_pct = proporção de exemplos em cada bin (TREINO)
```

**Método: `calculate_psi(train_scores, oot_scores, n_bins=10, epsilon=1e-10)`**

```python
# Processo
1. Definir bins usando percentis de TREINO (10 bins default)
2. Contar exemplos em cada bin para TREINO e OOT
3. Calcular PSI usando fórmula acima

# Interpretação
PSI < 0.1:   ESTÁVEL   (Modelo está OK, sem necessidade de refit urgente)
0.1 ≤ PSI < 0.25: MODERADO (Degradação detectada, refit recomendado em breve)
PSI ≥ 0.25:  SEVERO    (Degradação crítica, REFIT URGENTE NECESSÁRIO)
```

**Retorno Detalhado:**
```python
{
    'psi': 0.1234,           # Valor PSI
    'psi_category': 'Moderada',  # Categoria de degradação
    'train_mean': 0.45,      # Score médio em treino
    'oot_mean': 0.52,        # Score médio em OOT
    'train_std': 0.15,       # Desvio padrão treino
    'oot_std': 0.18,         # Desvio padrão OOT
    'divergence': 0.089      # KL Divergence (medida alternativa)
}
```

**Validações de Teste:**
- ✅ Teste 3: PSI < 0.1 para distribuição estável
- ✅ Teste 4: Categorização correta (Estável/Moderada/Severa)
- ✅ Teste 6: Cenário real detecta degradação severa (PSI=1.16)

---

### 1.3 ModelEvaluationReport

**Local:** `source/modeling/metrics.py` (linhas ~220-320)

**Responsabilidade:** Integrar todas as métricas em relatório consolidado

**Métodos:**
```python
# 1. Adicionar métricas de um modelo
report.add_model_metrics(
    model_name='XGBoost',
    y_true_train=y_train,
    y_scores_train=y_scores_train,
    y_true_oot=y_oot,
    y_scores_oot=y_scores_oot,
    standard_metrics={...},
    top_k_list=[100, 500]
)

# 2. Gerar relatório consolidado
df_report = report.generate_report()
# Colunas: model, precision_at_100_train/oot, precision_at_500_train/oot, psi, psi_category

# 3. Exibir resumo visual
report.print_summary()
```

---

## 2. Resultados de Teste

### 6/6 Testes Passando ✅

| Teste | Descrição | Resultado | Status |
|-------|-----------|-----------|--------|
| Teste 1 | Precision@K Básico | K=1:1.0, K=3:0.67, K=5:0.6 | ✅ PASSOU |
| Teste 2 | Precision@K Edge Cases | K>n, todos negativos, K=0 | ✅ PASSOU |
| Teste 3 | PSI Cálculo Básico | Estável(0.01) < Moderado(0.42) < Severo(2.81) | ✅ PASSOU |
| Teste 4 | PSI Categorização | Estável/Moderado/Severo corretos | ✅ PASSOU |
| Teste 5 | Relatório Completo | MultiModel, 9 colunas geradas | ✅ PASSOU |
| Teste 6 | Cenário Real AML | PSI=1.16 (Severa), Precision degradou | ✅ PASSOU |

---

## 3. Integração no Pipeline

### 3.1 Uso Direto

```python
from source.modeling.metrics import BusinessMetrics, ModelEvaluationReport

# Precision @ Top-K
prec_k = BusinessMetrics.precision_at_top_k(
    y_true=y_oot,
    y_scores=y_pred_proba,
    top_k_list=[100, 500]
)
print(f"Precision@100: {prec_k[100]:.4f}")  # 75 verdadeiros dos top-100

# PSI
psi = BusinessMetrics.calculate_psi(
    train_scores=y_pred_proba_train,
    oot_scores=y_pred_proba_oot,
    n_bins=10
)
print(f"PSI: {psi['psi']:.4f}")  # 0.12 = MODERADO
print(f"Status: {psi['psi_category']}")  # 'Moderada' = refit recomendado
```

### 3.2 Relatório Consolidado

```python
# Criar relatório para múltiplos modelos
report = ModelEvaluationReport()

# Adicionar XGBoost
report.add_model_metrics(
    model_name='XGBoost',
    y_true_train=y_train, y_scores_train=xgb_scores_train,
    y_true_oot=y_oot, y_scores_oot=xgb_scores_oot,
    standard_metrics={'train': {...}, 'oot': {...}}
)

# Adicionar LightGBM
report.add_model_metrics(
    model_name='LightGBM',
    y_true_train=y_train, y_scores_train=lgb_scores_train,
    y_true_oot=y_oot, y_scores_oot=lgb_scores_oot,
    standard_metrics={'train': {...}, 'oot': {...}}
)

# Gerar relatório
df_report = report.generate_report()
print(df_report)  # Compare XGBoost vs LightGBM em todas as métricas

# Salvar
df_report.to_csv('model_evaluation_etapa5.csv', index=False)
```

---

## 4. Scenario Real: Detecção de Degradação AML

**Cenário Teste 6 - Dados Altamente Desbalanceados (~924:1)**

```
Configuração:
  Treino: 10000 samples, 7 positivos (0.108%)
  OOT:     5000 samples, 3 positivos (0.086%)  ← Taxa mudou!

Resultados:
  Precision@100:  Treino=0.07  →  OOT=0.01  (Degradação: -0.06)
  Precision@500:  Treino=0.014 →  OOT=0.004 (Degradação: -0.01)
  PSI Value:      1.1551 (SEVERO!)
  
  Diagnóstico:
    [AVISO] Precision degradou em OOT
    [AVISO] PSI > 0.1 → Distribuição instável
    [CRITICO] PSI >= 0.25 → REFIT URGENTE!
    
  Ação Recomendada:
    → Retreinar modelo com dados recentes
    → Recalibrar thresholds
    → Monitorar PSI semanalmente
```

---

## 5. Propriedades de Qualidade

### 5.1 Precision@Top-K ✅

| Propriedade | Status | Detalhe |
|------------|--------|--------|
| Cálculo Correto | ✅ | Formula: TP/K validada |
| Edge Cases | ✅ | K>n, todos negativos, K=0 tratados |
| Interpretabilidade | ✅ | Métrica clara para operações AML |
| Performance | ✅ | O(n log n) sorting |
| Robustez | ✅ | Sem divisão por zero |

### 5.2 PSI ✅

| Propriedade | Status | Detalhe |
|------------|--------|--------|
| Cálculo Correto | ✅ | Fórmula: SUM((pct_oot-pct_train)ln(...)) |
| Categorização | ✅ | Estável<0.1, Moderada<0.25, Severa≥0.25 |
| Sensibilidade | ✅ | Detecta mudanças sutis na distribuição |
| Interpretabilidade | ✅ | Valores altos indicam degradação |
| Epsilon Protection | ✅ | Evita ln(0) com epsilon=1e-10 |

### 5.3 Relatório Integrado ✅

| Propriedade | Status | Detalhe |
|------------|--------|--------|
| Multi-modelo | ✅ | Suporta múltiplos modelos |
| Consolidação | ✅ | Agrupa todas métricas em um DataFrame |
| Exportação | ✅ | Suporta to_csv() |
| Visualização | ✅ | print_summary() com formatação |
| Extensibilidade | ✅ | Fácil adicionar novos KPIs |

---

## 6. Impacto de Negócio

**Antes (ETAPA 1-4):**
- Nenhuma métrica de performance operacional
- Sem monitoramento de degradação em produção
- Sem resposta para: "Modelo ainda funciona?"

**Depois (ETAPA 5):**
- Precision@100 e @500 quantificam eficiência de investigação
- PSI monitora automaticamente degradação
- Alerta automático quando refit é necessário
- Rastreabilidade completa de performance entre treino e OOT

**Valor Criado:**
```
Precisão de Investigação:
  - Reduz investigações repetidas
  - Prioriza casos de maior risco
  - Aumenta eficiência do time de AML

Estabilidade:
  - Detecta quando modelo degrada
  - Previne aplicação de modelos obsoletos
  - Reduz falsos positivos ao longo do tempo
```

---

## 7. Comparação com ETAPA 4

| Aspecto | ETAPA 4 | ETAPA 5 |
|---------|---------|---------|
| Foco | Anti-Leakage Transformations | Métricas de Negócio |
| Classes | YeoJohnson, TargetEncoder | BusinessMetrics, ModelEvaluationReport |
| Testes | 5/5 (Leakage) | 6/6 (Métricas + Relatório) |
| Saída | Dados transformados | Métricas e Relatórios |
| Usuários | Data Scientists | Risk Officers, AML Teams |
| Python LOC | 220 | 320 |

---

## 8. Arquivos Entregues

**Código:**
- `source/modeling/metrics.py` - Implementação completa (320 LOC)

**Testes:**
- `scripts/test_stage_5_business_metrics.py` - 6 testes (440 LOC)

**Documentação:**
- `ETAPA_5_COMPLETADA.md` - Este arquivo
- `ETAPA_5_CODIGO_PRONTO.md` - Código copy-paste
- `ETAPA_5_SUMARIO_TECNICO.md` - Sumário executivo

---

## 9. Conclusão

**ETAPA 5: Métricas de Negócio e Estabilidade foi implementada com sucesso!**

✅ Precision@Top-K: Mede eficiência de investigação (Top-100, Top-500)
✅ PSI: Monitora degradação de modelo entre treino e OOT
✅ Relatório Consolidado: Integra todas métricas com múltiplos modelos
✅ 6/6 Testes Passando: Validação completa
✅ Produção Ready: Código robusto e documentado

**Status Final:** ETAPA 5 ✅ COMPLETA E VALIDADA

---

*Documentação técnica completa. Pronto para integração em operação.* 🚀
