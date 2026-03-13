# ETAPA 1: Blocos de Código Prontos para Substituição

Este arquivo contém fragmentos de código **prontos para usar** nas etapas seguintes do projeto.

---

## 📌 1. Integração com Notebook 05_feature_engineering.ipynb

Se você está usando o notebook simplificado, pode substituir a seção de feature engineering assim:

```python
# SUBSTITUIR ESTA CÉLULA NO NOTEBOOK
# (Antiga abordagem simplificada)

# POR ISTO:

from source.features import FeatureEngineeringPipeline
import pandas as pd

# Criar pipeline com configuração padrão
pipeline = FeatureEngineeringPipeline(
    timestamp_col='Timestamp',
    account_col='Account',
    amount_col='Amount Received',
    bank_col='Receiving Currency',
    country_col='From Bank',
    velocity_windows={'1h': '1H', '24h': '24H', '7d': '7D'},
    ratio_window='30D',
    smurf_threshold=10000.0,
    smurf_time_window='24H'
)

# Aplicar em treino
df_treino_fe = pipeline.fit_transform(df_treino)

# Aplicar em OOT
df_oot_fe = pipeline.fit_transform(df_oot)

print(f"Treino: {df_treino_fe.shape}")
print(f"OOT: {df_oot_fe.shape}")

# Salvar resultados
df_treino_fe.to_csv(get_data_path('df_treino_fe.csv', 'processed'), index=False)
df_oot_fe.to_csv(get_data_path('df_oot_fe.csv', 'processed'), index=False)
```

---

## 📌 2. Integração com Pipeline de Modelagem

Se você está preparando dados para o notebook 08_treinamento:

```python
# source/modeling/train.py - Adicionar no início do módulo

from source.features import FeatureEngineeringPipeline

class DataPreprocessor:
    """Preprocessador com Feature Engineering integrado."""
    
    def __init__(self, use_fe=True):
        self.use_fe = use_fe
        if use_fe:
            self.fe_pipeline = FeatureEngineeringPipeline()
    
    def prepare_data(self, df_treino, df_oot):
        """
        Prepara dados com Feature Engineering.
        
        Args:
            df_treino: DataFrame de treino
            df_oot: DataFrame de OOT
        
        Returns:
            Tuple(df_treino_processado, df_oot_processado)
        """
        
        if self.use_fe:
            print("[INFO] Aplicando Feature Engineering...")
            df_treino = self.fe_pipeline.fit_transform(df_treino)
            df_oot = self.fe_pipeline.fit_transform(df_oot)
            print(f"[OK] Treino: {df_treino.shape}, OOT: {df_oot.shape}")
        
        return df_treino, df_oot
```

Uso:

```python
preprocessor = DataPreprocessor(use_fe=True)
df_treino_fe, df_oot_fe = preprocessor.prepare_data(df_treino, df_oot)
```

---

## 📌 3. Métodos para Análise de Features

Adicione estes métodos  à sua classe de análise:

```python
def analyze_velocity_features(self, df):
    """Análise de features de velocidade."""
    velocity_cols = [col for col in df.columns if 'velocity' in col]
    
    print("\n" + "="*80)
    print("ANALISE: VELOCITY FEATURES")
    print("="*80)
    
    for col in sorted(velocity_cols):
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std:  {df[col].std():.2f}")
        print(f"  Min:  {df[col].min():.2f}")
        print(f"  Max:  {df[col].max():.2f}")
        print(f"  Correlação com target: {df[col].corr(df['Is Laundering']):.4f}")

def analyze_ratio_features(self, df):
    """Análise de features de ratio."""
    ratio_cols = [col for col in df.columns if 'ratio' in col or 'zscore' in col]
    
    print("\n" + "="*80)
    print("ANALISE: RATIO FEATURES")
    print("="*80)
    
    for col in sorted(ratio_cols):
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std:  {df[col].std():.2f}")
        print(f"  Min:  {df[col].min():.2f}")
        print(f"  Max:  {df[col].max():.2f}")
        print(f"  Correlação com target: {df[col].corr(df['Is Laundering']):.4f}")

def analyze_smurfing_features(self, df):
    """Análise de features de smurfing."""
    smurf_cols = [col for col in df.columns if 'smurf' in col]
    
    print("\n" + "="*80)
    print("ANALISE: SMURFING DETECTION FEATURES")
    print("="*80)
    
    # Detecção de casos
    smurf_potential = (df['smurf_txn_count_24h_behavioral'] > 0).sum()
    print(f"\nTransações com potencial smurfing: {smurf_potential} ({100*smurf_potential/len(df):.2f}%)")
    
    # Distribuição por classe
    print(f"\nDistribuição de smurfing_txn_count_24h_behavioral:")
    for label in [0, 1]:
        subset = df[df['Is Laundering'] == label]
        smurf_count = (subset['smurf_txn_count_24h_behavioral'] > 0).sum()
        print(f"  Classe {label}: {smurf_count}/{len(subset)} ({100*smurf_count/len(subset):.2f}%)")
    
    for col in smurf_cols:
        print(f"\n{col}:")
        print(f"  Correlação com target: {df[col].corr(df['Is Laundering']):.4f}")

def compare_fe_importance(self, df_original, df_with_fe):
    """Compara importância relativa das features (original vs. FE)."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Features originais
    original_cols = [col for col in df_original.columns 
                    if col not in ['Timestamp', 'Is Laundering']]
    
    # Novas features
    new_features = [col for col in df_with_fe.columns 
                   if col not in df_original.columns and col != 'Is Laundering']
    
    # Treinar RF com features originais
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(df_original[original_cols], df_original['Is Laundering'])
    
    # Treinar RF com FE
    rf_fe = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_fe.fit(df_with_fe[original_cols + new_features], df_with_fe['Is Laundering'])
    
    print("\n" + "="*80)
    print("COMPARACAO: IMPORTANCIA DAS FEATURES")
    print("="*80)
    
    print(f"\nScore AUC (original): {rf_original.score(df_original[original_cols], df_original['Is Laundering']):.4f}")
    print(f"Score AUC (com FE): {rf_fe.score(df_with_fe[original_cols + new_features], df_with_fe['Is Laundering']):.4f}")
    
    # Top features de FE
    importances = rf_fe.feature_importances_
    feature_importance = list(zip(original_cols + new_features, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Features (com FE):")
    for feat, imp in feature_importance[:10]:
        marker = "[NEW FE]" if feat in new_features else "[ORIGINAL]"
        print(f"  {marker:12} {feat:40} {imp:.6f}")
```

---

## 📌 4. Validação Customizada

Adicione esta função para validar features em novos dados:

```python
def validate_features_in_data(df, expected_velocity_cols=13, expected_ratio_cols=3, expected_behavioral_cols=3):
    """
    Valida que um DataFrame tem todas as features esperadas.
    
    Args:
        df: DataFrame a validar
        expected_*: Número esperado de features em cada categoria
    
    Returns:
        Dict com resultados de validação
    """
    
    velocity_cols = [col for col in df.columns if 'velocity' in col]
    ratio_cols = [col for col in df.columns if 'ratio' in col or 'zscore' in col]
    behavioral_cols = [col for col in df.columns if 'behavioral' in col]
    
    results = {
        'velocity_features': len(velocity_cols) == expected_velocity_cols,
        'ratio_features': len(ratio_cols) == expected_ratio_cols,
        'behavioral_features': len(behavioral_cols) == expected_behavioral_cols,
        'no_nans': df.isnull().sum().sum() == 0,
        'total_features': len(velocity_cols) + len(ratio_cols) + len(behavioral_cols)
    }
    
    # Imprimir resultados
    print("Validação de Features:")
    print(f"  Velocity: {len(velocity_cols)}/{expected_velocity_cols} ✓" if results['velocity_features'] else f"  Velocity: {len(velocity_cols)}/{expected_velocity_cols} ✗")
    print(f"  Ratio: {len(ratio_cols)}/{expected_ratio_cols} ✓" if results['ratio_features'] else f"  Ratio: {len(ratio_cols)}/{expected_ratio_cols} ✗")
    print(f"  Behavioral: {len(behavioral_cols)}/{expected_behavioral_cols} ✓" if results['behavioral_features'] else f"  Behavioral: {len(behavioral_cols)}/{expected_behavioral_cols} ✗")
    print(f"  Sem NaNs: ✓" if results['no_nans'] else f"  Sem NaNs: ✗ ({df.isnull().sum().sum()} NaNs encontrados)")
    
    return results
```

---

## 📌 5. Configuração para Diferentes Cenários

### Cenário 1: Dataset Pequeno (< 10k transações)

```python
pipeline = FeatureEngineeringPipeline(
    velocity_windows={'24h': '24H', '7d': '7D'},  # Apenas 24h e 7d
    ratio_window='30D'
)
```

### Cenário 2: Dataset Grande (> 100k transações)

```python
from multiprocessing import cpu_count

pipeline = FeatureEngineeringPipeline(
    velocity_windows={'1h': '1H', '24h': '24H', '7d': '7D', '30d': '30D'},  # Adiciona 30d
    ratio_window='60D'  # Histórico maior
)
```

### Cenário 3: Otimizado para Smurfing

```python
pipeline = FeatureEngineeringPipeline(
    smurf_threshold=9900.0,  # Mais sensível
    smurf_time_window='12H'  # Janela menor
)
```

---

## 📌 6. Exportação para XGBoost/LightGBM

Após criar as features, prepare para treinamento:

```python
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def prepare_for_xgboost(df_feat, target_col='Is Laundering', drop_cols=None):
    """
    Prepara features para XGBoost.
    
    Args:
        df_feat: DataFrame com features FE
        target_col: Nome da coluna target
        drop_cols: Colunas para remover
    
    Returns:
        Tuple(X, y)
    """
    
    if drop_cols is None:
        drop_cols = ['Timestamp', 'Account', 'From Bank', 'Receiving Currency']
    
    # Remove dropped columns se presentes
    X = df_feat.drop(columns=[col for col in drop_cols if col in df_feat.columns], errors='ignore')
    y = df_feat[target_col]
    
    # Validar que não há NaNs
    assert X.isnull().sum().sum() == 0, "Features contêm NaNs!"
    
    print(f"[OK] X shape: {X.shape}")
    print(f"[OK] y shape: {y.shape}")
    print(f"[OK] Classes: {y.value_counts().to_dict()}")
    
    return X, y

# Uso
X_train, y_train = prepare_for_xgboost(df_treino_fe)
X_oot, y_oot = prepare_for_xgboost(df_oot_fe)

# Treinar XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()  # Ajusta desequilíbrio
)

xgb_model.fit(X_train, y_train)
```

---

## 📌 7. Salvar/Carregar Features

```python
import pickle

def save_fe_pipeline(pipeline, path='models/fe_pipeline.pkl'):
    """Salva o pipeline de FE treinado."""
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[OK] Pipeline salvo em {path}")

def load_fe_pipeline(path='models/fe_pipeline.pkl'):
    """Carrega o pipeline de FE."""
    with open(path, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"[OK] Pipeline carregado de {path}")
    return pipeline

# Salvar após treino
save_fe_pipeline(pipeline)

# Carregar em produção
pipeline_prod = load_fe_pipeline()
df_new_fe = pipeline_prod.fit_transform(df_new)
```

---

## ✅ Checklist de Integração

- [ ] Importar `FeatureEngineeringPipeline` em seu notebook
- [ ] Testar com dados de treino e OOT separadamente
- [ ] Validar que não há NaNs após FE
- [ ] Validar que valores estão em ranges esperados
- [ ] Comparar importância de features original vs. FE
- [ ] Salvar pipeline para produção
- [ ] Adicionar validação em dados novos

---

## 🔗 Próximos Passos

Agora que a ETAPA 1 está pronta, você pode:

1. **ETAPA 2:** Implementar matriz de custo diferenciada para FP/FN
2. **ETAPA 3:** Usar Optuna para tuning de scale_pos_weight + Isolation Forest
3. **ETAPA 4:** Implementar Target Encoding + Yeo-Johnson
4. **ETAPA 5:** Calcular Precision@Top-K + PSI

---

**Última atualização:** 13 de Março de 2026
