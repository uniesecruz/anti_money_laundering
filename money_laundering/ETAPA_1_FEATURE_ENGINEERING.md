# ETAPA 1: Refinamento de Feature Engineering - Implementação Completa

## 📋 Resumo das Alterações

Este documento descreve a implementação completa da **Etapa 1** de refatoração do sistema AML.

### Features Implementadas

#### 1. **Velocity Features** com Janelas Deslizantes
Implementação em `source/features.py` - Classe `VelocityFeatureGenerator`:

```python
windows = {
    '1h': '1H',      # Última 1 hora
    '24h': '24H',    # Últimas 24 horas
    '7d': '7D'       # Últimos 7 dias
}
```

**Features geradas por janela:**
- `txn_count_{window}_velocity`: Contagem de transações
- `amount_sum_{window}_velocity`: Soma total de valores
- `amount_mean_{window}_velocity`: Média de valores
- `amount_max_{window}_velocity`: Máximo valor
- `amount_std_{window}_velocity`: Desvio padrão (apenas 7d)

**Total: 14 features de velocidade**

#### 2. **Ratio Features** - Comparação com Histórico
Implementação em `source/features.py` - Classe `RatioFeatureGenerator`:

Janela padrão: **30 dias** de histórico

**Features geradas:**
- `amount_to_historical_mean_ratio`: Valor atual / Média histórica
- `amount_to_historical_max_ratio`: Valor atual / Máximo histórico
- `amount_zscore_historical`: Z-score (desvio padronizado)

**Total: 3 features de ratio**

#### 3. **Behavioral Features com Smurfing Detection** 🚩
Implementação em `source/features.py` - Classe `BehavioralFeatureGenerator`:

**Features Convencionais:**
- `time_since_last_txn_seconds`: Tempo desde última transação
- `bank_change_flag`: Mudança de banco
- `is_new_country`: Novo país (não visto antes)
- `is_unusual_hour`: Fora do horário comercial
- `hour_of_day`: Hora do dia

**Features de Smurfing Detection (NOVO):**
- `smurf_txn_count_24h_behavioral`: Contagem de transações entre $8k-$10k em 24h
- `smurf_amount_sum_24h_behavioral`: Soma de transações "pequenas" em 24h
- `smurf_proximity_score_behavioral`: Score de proximidade com $10k threshold

**Total: 8 features comportamentais**

---

## 🔧 Como Usar

### Opção 1: Pipeline Completo (Recomendado)

```python
from source.features import FeatureEngineeringPipeline
import pandas as pd

# Criar pipeline com configuração padrão
pipeline = FeatureEngineeringPipeline(
    timestamp_col='Timestamp',
    account_col='Account',
    amount_col='Amount Received',
    smurf_threshold=10000.0,  # Threshold de smurfing
    smurf_time_window='24H'
)

# Aplicar em dados de treino
df_treino_fe = pipeline.fit_transform(df_treino)

# Aplicar em dados de OOT (IMPORTANTE: usar fit_transform separadamente)
df_oot_fe = pipeline.fit_transform(df_oot)

print(f"Total de features geradas: {df_treino_fe.shape[1] - df_treino.shape[1]}")
```

### Opção 2: Usar Geradores Individuais

```python
from source.features import (
    VelocityFeatureGenerator,
    RatioFeatureGenerator,
    BehavioralFeatureGenerator
)

# Velocity Features
velocity_gen = VelocityFeatureGenerator(
    windows={'1h': '1H', '24h': '24H', '7d': '7D'}
)
df = velocity_gen.fit_transform(df)

# Ratio Features
ratio_gen = RatioFeatureGenerator(window='30D')
df = ratio_gen.fit_transform(df)

# Behavioral Features com Smurfing
behavioral_gen = BehavioralFeatureGenerator(
    smurf_threshold=10000.0,
    smurf_time_window='24H'
)
df = behavioral_gen.fit_transform(df)
```

---

## ⚠️ Pontos Críticos de Data Leakage Prevention

### 1. **Ordenação Temporal**
```python
# SEMPRE ordenar antes de gerar features
df = df.sort_values('Timestamp').reset_index(drop=True)
```

### 2. **Usar Janelas com `closed='left'`**
- Isso EXCLUI a transação atual da janela de cálculo
- Implementado em todas as classes do pipeline
- Garante que não há information leakage

### 3. **Dividir Treino/OOT ANTES de Feature Engineering**
```python
# ✅ CORRETO
df_treino_fe = pipeline.fit_transform(df_treino)  # Fit em treino
df_oot_fe = pipeline.fit_transform(df_oot)        # Fit em OOT (separado)

# ❌ ERRADO
df_combined_fe = pipeline.fit_transform(df_combined)  # Mistura treino+OOT
```

---

## 📊 Interpretação das Features

### Velocity Features
- **Interpretação Econômica**: Atividade anormal (muitas transações em pouco tempo)
- **Exemplo**: Se `txn_count_24h_velocity = 50`, a conta teve 50 transações em 24h

### Ratio Features
- **Interpretação Econômica**: Desvio do padrão histórico
- **Exemplo**: Se `amount_to_historical_mean_ratio = 5.0`, a transação é 5x maior que a média

### Behavioral Features
- **Smurfing**: Padrão de múltiplas transações próximas de $10k
- **Novo País**: Desvio geográfico
- **Hora Incomum**: Transação fora do horário comercial

---

## 📈 Impacto Esperado no Modelo

### Em Casos de Lavagem (Positivos)

1. **Velocity aumenta**: Lavadores precisam mover muito volume - `txn_count_24h_velocity` sobe
2. **Ratio aumenta**: Transações maiores que o normal - `amount_to_historical_mean_ratio` sobe
3. **Smurfing ativa**: Padrão de transações estruturadas - `smurf_txn_count_24h_behavioral` sobe

### Em Casos Legítimos (Negativos)

1. **Velocity estável**: Padrão consistente de transações
2. **Ratio próximo a 1**: Transações normais comparadas ao histórico
3. **Smurfing inativo**: Nenhuma indicação de estruturação

---

## 🧪 Validação e Testes

### Test 1: Verificar Ausência de NaNs
```python
fe_cols = [col for col in df.columns if 'velocity' in col or 'ratio' in col or 'behavioral' in col]
assert df[fe_cols].isna().sum().sum() == 0, "Há NaNs nas features!"
```

### Test 2: Verificar Ordem Temporal
```python
assert df['Timestamp'].is_monotonic_increasing, "Dados não estão ordenados!"
```

### Test 3: Validar Ranges
```python
# Velocity features devem ser >= 0
assert (df[[col for col in df.columns if 'velocity' in col]] >= 0).all().all()

# Ratio features podem ser negativas (z-score)
# Smurf features devem estar entre 0 e 1 (score de proximidade)
assert (df['smurf_proximity_score_behavioral'] >= 0).all()
assert (df['smurf_proximity_score_behavioral'] <= 1).all()
```

---

## 🔄 Integração com Pipeline de Transformação

### Sequência Recomendada

```python
# 1. Carregar dados (já divididos em treino/OOT)
df_treino = pd.read_csv('data/processed/df_treino.csv')
df_oot = pd.read_csv('data/processed/df_oot.csv')

# 2. Aplicar Feature Engineering (ETAPA 1 - ESTE ARQUIVO)
fe_pipeline = FeatureEngineeringPipeline()
df_treino_fe = fe_pipeline.fit_transform(df_treino)
df_oot_fe = fe_pipeline.fit_transform(df_oot)

# 3. Aplicar Transformações (escaling, encoding, etc)
# Isso será feito na ETAPA 4

# 4. Treinar modelo (ETAPA 3 - com Optuna e Isolation Forest)
# Isso será feito depois
```

---

## 📝 Notas Técnicas

### Implementação de Janelas Deslizantes
- Usamos `rolling(window_size, closed='left')` do pandas
- `closed='left'` significa que a transação atual é EXCLUÍDA
- Isso previne data leakage automático

### Tratamento de NaNs
- Primeiras transações de cada conta não têm histórico prévio
- Preenchemos com 0 (sem atividade anterior)
- Esta é uma escolha razoável: assume "novo" como ausência de padrão

### Eficiência Computacional
- Operações groupby + rolling são otimizadas em pandas
- Para datasets muito grandes, considerar implementação em PySpark
- Implementação PySpark futura pode usar `Window` functions de SQL

---

## 🚀 Próximos Passos

- **ETAPA 2**: Otimizar matriz de custos (FP vs FN)
- **ETAPA 3**: Bayesian Optimization + Isolation Forest
- **ETAPA 4**: Target Encoding + Yeo-Johnson (anti-leakage)
- **ETAPA 5**: Métricas de negócio (Precision@Top-K, PSI)

---

## 📚 Referências

- Velocity Features: [Kaggle Discussion on Velocity](https://www.kaggle.com/discussion)
- Smurfing Detection: [FinCEN Smurfing Guidelines](https://www.fincen.gov/)
- Data Leakage Prevention: [Kaggle's "Data Leakage" Course](https://www.kaggle.com/learn/data-leakage)

