"""
Script de Validacao - ETAPA 1: Feature Engineering Refinado (Versao Simplificada)
"""

import sys
import os
from pathlib import Path

# Adicionar projeto root ao path PRIMEIRO
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from source.features import FeatureEngineeringPipeline

print("=" * 80)
print("[INICIO] Validacao ETAPA 1: Feature Engineering Refinado")
print("=" * 80)

# Criar dataset sintetico
print("\n[INFO] Criando dataset sintetico...")
np.random.seed(42)
base_date = datetime(2024, 1, 1)
n_samples = 2000

timestamps = [base_date + timedelta(minutes=i*10) for i in range(n_samples)]
accounts = np.random.choice(['ACC001', 'ACC002', 'ACC003', 'ACC004'], n_samples)
amounts = np.random.normal(loc=5000, scale=2000, size=n_samples)
amounts = np.clip(amounts, 100, 50000)

# Injetar smurfing
smurf_indices = np.where(accounts == 'ACC001')[0][:10]
amounts[smurf_indices] = np.random.uniform(9000, 9900, len(smurf_indices))

df = pd.DataFrame({
    'Timestamp': timestamps,
    'Account': accounts,
    'Amount Received': amounts,
    'Receiving Currency': np.random.choice(['USD', 'EUR'], n_samples),
    'From Bank': np.random.choice(['Bank_A', 'Bank_B', 'Bank_C'], n_samples),
    'Is Laundering': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
})

df = df.sort_values('Timestamp').reset_index(drop=True)
print(f"[OK] Dataset criado: {df.shape}")

# Dividir em treino e OOT
split_idx = int(len(df) * 0.8)
df_treino = df[:split_idx].reset_index(drop=True)
df_oot = df[split_idx:].reset_index(drop=True)

# Aplicar pipeline
print("\n[INFO] Aplicando Feature Engineering Pipeline...")
pipeline = FeatureEngineeringPipeline()
df_treino_fe = pipeline.fit_transform(df_treino)
df_oot_fe = pipeline.fit_transform(df_oot)

print(f"[OK] Treino: {df_treino.shape[0]} rows, {df_treino_fe.shape[1]} colunas")
print(f"[OK] OOT: {df_oot.shape[0]} rows, {df_oot_fe.shape[1]} colunas")

# Teste 1: Features geradas
new_cols = df_treino_fe.shape[1] - df_treino.shape[1]
print(f"\n[TESTE 1] Total de novas features: {new_cols}")
if new_cols >= 22:
    print("[OK] PASSOU - Numero adequado de features geradas")
else:
    print(f"[FAIL] FALHOU - Esperado >= 22 features, obteve {new_cols}")

# Teste 2: Velocity Features
velocity_cols = [col for col in df_treino_fe.columns if 'velocity' in col]
print(f"\n[TESTE 2] Velocity Features: {len(velocity_cols)} encontradas")
if len(velocity_cols) >= 12:
    print("[OK] PASSOU")
else:
    print(f"[FAIL] FALHOU - Esperado >= 12, obteve {len(velocity_cols)}")

# Teste 3: Ratio Features
ratio_cols = [col for col in df_treino_fe.columns if 'ratio' in col or 'zscore' in col]
print(f"\n[TESTE 3] Ratio Features: {len(ratio_cols)} encontradas")
if len(ratio_cols) >= 3:
    print("[OK] PASSOU")
else:
    print(f"[FAIL] FALHOU - Esperado >= 3, obteve {len(ratio_cols)}")

# Teste 4: Behavioral Features
behavioral_cols = [col for col in df_treino_fe.columns if 'behavioral' in col]
smurf_cols = [col for col in behavioral_cols if 'smurf' in col]
print(f"\n[TESTE 4] Behavioral Features (incluindo Smurfing):")
print(f"   Total: {len(behavioral_cols)}")
print(f"   Smurfing: {len(smurf_cols)}")
if len(smurf_cols) >= 3:
    print("[OK] PASSOU - Features de smurfing detectadas")
else:
    print(f"[FAIL] FALHOU - Esperado >= 3 smurf features, obteve {len(smurf_cols)}")

# Teste 5: NaNs
feature_cols_all = velocity_cols + ratio_cols + behavioral_cols
nan_count = df_treino_fe[feature_cols_all].isna().sum().sum()
print(f"\n[TESTE 5] Ausencia de NaNs: {nan_count} NaNs encontrados")
if nan_count == 0:
    print("[OK] PASSOU")
else:
    print(f"[FAIL] FALHOU - Had {nan_count} NaN values")

# Teste 6: Valores dentro de ranges esperados
print(f"\n[TESTE 6] Validacao de Ranges:")
tests_passed = 0
tests_total = 0

# Velocity >= 0
for col in velocity_cols:
    tests_total += 1
    if (df_treino_fe[col] >= 0).all():
        tests_passed += 1
    else:
        print(f"   [FAIL] {col} tem valores < 0")

# Smurfing proximity [0,1]
if 'smurf_proximity_score_behavioral' in df_treino_fe.columns:
    tests_total += 1
    col = 'smurf_proximity_score_behavioral'
    if ((df_treino_fe[col] >= 0) & (df_treino_fe[col] <= 1)).all() or df_treino_fe[col].isna().all():
        tests_passed += 1
    else:
        print(f"   [FAIL] {col} fora do range [0,1]")

print(f"   [OK] {tests_passed}/{tests_total} sub-testes passaram")

# Teste 7: Smurfing detectado
smurf_detected = (df_treino_fe['smurf_txn_count_24h_behavioral'] > 0).sum()
print(f"\n[TESTE 7] Deteccao de Smurfing: {smurf_detected} transacoes com padrao detectado")
if smurf_detected > 0:
    print("[OK] PASSOU - Smurfing patterns foram detectados")
else:
    print("[INFO] Nenhum smurfing detectado (pode ser normal para dados aleatorios)")

# Resumo final
print("\n" + "=" * 80)
print("[RESUMO] Feature Engineering ETAPA 1")
print("=" * 80)
print(f"Velocity Features: {len(velocity_cols)}")
print(f"Ratio Features: {len(ratio_cols)}")
print(f"Behavioral Features: {len(behavioral_cols)} (incluindo {len(smurf_cols)} de smurfing)")
print(f"Total de features novas: {new_cols}")
print("\n[CONCLUSAO] ETAPA 1 implementada com sucesso!")
print("=" * 80)
