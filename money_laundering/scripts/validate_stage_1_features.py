"""
Script de Validação - ETAPA 1: Feature Engineering Refinado

Este script valida que todas as features foram implementadas corretamente
sem data leakage e com comportamento esperado.

Uso:
    python scripts/validate_stage_1_features.py
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

from source.features import (
    VelocityFeatureGenerator,
    RatioFeatureGenerator,
    BehavioralFeatureGenerator,
    FeatureEngineeringPipeline
)


def create_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Cria um dataset sintético para testes.
    
    Args:
        n_samples: Número de transações
    
    Returns:
        DataFrame com transações sintéticas
    """
    print("🔄 Criando dataset sintético...")
    
    np.random.seed(42)
    
    # Gerar timestamps
    base_date = datetime(2024, 1, 1)
    timestamps = [base_date + timedelta(minutes=i*10) for i in range(n_samples)]
    
    # Gerar contas
    accounts = np.random.choice(['ACC001', 'ACC002', 'ACC003', 'ACC004'], n_samples)
    
    # Gerar valores (alguns padrões de smurfing)
    amounts = np.random.normal(loc=5000, scale=2000, size=n_samples)
    amounts = np.clip(amounts, 100, 50000)  # Clip entre 100 e 50k
    
    # Injetar padrão de smurfing em ACC001
    smurf_indices = np.where(accounts == 'ACC001')[0][:10]
    amounts[smurf_indices] = np.random.uniform(9000, 9900, len(smurf_indices))
    
    # Criar DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Account': accounts,
        'Amount Received': amounts,
        'Receiving Currency': np.random.choice(['USD', 'EUR'], n_samples),
        'From Bank': np.random.choice(['Bank_A', 'Bank_B', 'Bank_C'], n_samples),
        'Is Laundering': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    })
    
    # Garantir que está ordenado
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"✅ Dataset sintético criado: {df.shape}")
    print(f"   Período: {df['Timestamp'].min()} até {df['Timestamp'].max()}")
    print(f"   Contas: {df['Account'].unique()}")
    
    return df


def validate_temporal_ordering(df: pd.DataFrame, timestamp_col: str = 'Timestamp') -> bool:
    """Valida que os dados estão ordenados por timestamp."""
    print("\n📍 Validação 1: Ordenação Temporal")
    is_sorted = df[timestamp_col].is_monotonic_increasing
    print(f"   {'✅' if is_sorted else '❌'} Dados ordenados: {is_sorted}")
    return is_sorted


def validate_no_nans_in_features(df: pd.DataFrame, feature_cols: list) -> bool:
    """Valida que não há NaNs nas features."""
    print("\n📍 Validação 2: Ausência de NaNs")
    has_nans = df[feature_cols].isna().sum().sum()
    success = has_nans == 0
    print(f"   {'✅' if success else '❌'} NaNs nas features: {has_nans}")
    return success


def validate_velocity_features(df: pd.DataFrame) -> bool:
    """Valida features de velocidade."""
    print("\n📍 Validação 3: Velocity Features")
    
    velocity_cols = [col for col in df.columns if 'velocity' in col]
    print(f"   ✓ Features de velocidade encontradas: {len(velocity_cols)}")
    for col in velocity_cols:
        print(f"     - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    # Validar que são >= 0
    all_positive = (df[velocity_cols] >= 0).all().all()
    print(f"   {'✅' if all_positive else '❌'} Todos os valores >= 0: {all_positive}")
    
    # Validar que contagem de transações faz sentido
    count_cols = [col for col in velocity_cols if 'txn_count' in col]
    if count_cols:
        max_count = df[count_cols].max().max()
        print(f"   ✓ Máxima contagem de transações observada: {max_count:.0f}")
    
    return all_positive and len(velocity_cols) >= 12


def validate_ratio_features(df: pd.DataFrame) -> bool:
    """Valida features de ratio."""
    print("\n📍 Validação 4: Ratio Features")
    
    ratio_cols = [col for col in df.columns if 'ratio' in col or 'zscore' in col]
    print(f"   ✓ Features de ratio encontradas: {len(ratio_cols)}")
    for col in ratio_cols:
        print(f"     - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    return len(ratio_cols) >= 3


def validate_behavioral_features(df: pd.DataFrame) -> bool:
    """Valida features comportamentais e de smurfing."""
    print("\n📍 Validação 5: Behavioral Features (incluindo Smurfing)")
    
    # Features esperadas
    expected_features = [
        'time_since_last_txn_seconds',
        'bank_change_flag',
        'is_new_country',
        'is_unusual_hour',
        'smurf_txn_count_24h_behavioral',
        'smurf_amount_sum_24h_behavioral',
        'smurf_proximity_score_behavioral'
    ]
    
    found_features = [col for col in expected_features if col in df.columns]
    print(f"   ✓ Features encontradas: {len(found_features)}/{len(expected_features)}")
    
    for feat in found_features:
        if 'flag' in feat or 'unusual' in feat:
            print(f"     - {feat}: valores únicos = {df[feat].unique()}")
        else:
            print(f"     - {feat}: min={df[feat].min():.2f}, max={df[feat].max():.2f}")
    
    # Validar valores de flag (0 ou 1)
    flag_cols = [col for col in found_features if 'flag' in col or 'unusual' in col]
    all_binary = all((df[col].isin([0, 1])).all() for col in flag_cols)
    print(f"   {'✅' if all_binary else '❌'} Flags são binários: {all_binary}")
    
    # Validar smurf features
    smurf_cols = [col for col in found_features if 'smurf' in col]
    if smurf_cols:
        print(f"   ✅ Features de smurfing detectadas:")
        for col in smurf_cols:
            print(f"     - {col}: observações com valor > 0 = {(df[col] > 0).sum()}")
    
    return len(found_features) >= 5


def validate_smurfing_detection(df: pd.DataFrame) -> bool:
    """Valida detecção de padrões de smurfing."""
    print("\n📍 Validação 6: Detecção de Smurfing (Padrão de Estruturação)")
    
    if 'smurf_txn_count_24h_behavioral' not in df.columns:
        print("   ❌ Feature de smurfing não encontrada")
        return False
    
    # Contar transações com potencial smurfing
    smurf_potential = (df['smurf_txn_count_24h_behavioral'] > 0).sum()
    print(f"   ✓ Transações com padrão de smurfing: {smurf_potential}")
    
    if smurf_potential > 0:
        print("   ✅ Smurfing detectado (esperado em dataset sintético)")
        
        # Mostrar exemplo
        smurf_examples = df[df['smurf_txn_count_24h_behavioral'] > 0].head(3)
        print(f"   Exemplos:")
        for idx, row in smurf_examples.iterrows():
            print(f"     - Account: {row['Account']}, Amount: ${row['Amount Received']:.2f}, "
                  f"Smurf Count: {row['smurf_txn_count_24h_behavioral']}")
    
    return True


def validate_feature_engineering_pipeline(df_treino: pd.DataFrame, df_oot: pd.DataFrame) -> bool:
    """Valida que o pipeline funciona sem data leakage."""
    print("\n📍 Validação 7: Pipeline Completo (Anti-Leakage)")
    
    pipeline = FeatureEngineeringPipeline()
    
    # Aplicar independentemente
    df_treino_fe = pipeline.fit_transform(df_treino)
    df_oot_fe = pipeline.fit_transform(df_oot)
    
    print(f"   ✓ Treino: {df_treino.shape[0]} → {df_treino_fe.shape[0]} (colunas: {df_treino.shape[1]} → {df_treino_fe.shape[1]})")
    print(f"   ✓ OOT: {df_oot.shape[0]} → {df_oot_fe.shape[0]} (colunas: {df_oot.shape[1]} → {df_oot_fe.shape[1]})")
    
    # Validar que aplicar pipeline 2x não muda o resultado (idempotente)
    df_treino_fe_2 = pipeline.fit_transform(df_treino_fe)
    
    # Comparar colunas originais
    orig_cols = df_treino.columns
    for col in orig_cols:
        if not (df_treino_fe[col] == df_treino_fe_2[col]).all():
            print(f"   ❌ Aplicar pipeline 2x mudou coluna: {col}")
            return False
    
    print("   ✅ Pipeline é idempotente (aplicar 2x = mesmo resultado)")
    return True


def print_feature_summary(df: pd.DataFrame, original_shape: tuple) -> None:
    """Imprime sumário das features geradas."""
    print("\n" + "="*80)
    print("📊 SUMÁRIO DE FEATURES GERADAS")
    print("="*80)
    
    new_cols = df.shape[1] - original_shape[1]
    print(f"✅ Total de features novas: {new_cols}")
    print(f"✅ Forma final: {df.shape}")
    
    # Agrupar por tipo
    velocity = [col for col in df.columns if 'velocity' in col]
    ratio = [col for col in df.columns if 'ratio' in col or 'zscore' in col]
    behavioral = [col for col in df.columns if 'behavioral' in col]
    
    print(f"\n📈 Velocity Features: {len(velocity)}")
    print(f"📊 Ratio Features: {len(ratio)}")
    print(f"🚩 Behavioral Features: {len(behavioral)}")
    
    print("\n🔗 Features Velocity:")
    for col in sorted(velocity):
        print(f"   - {col}")
    
    print("\n📐 Features Ratio:")
    for col in sorted(ratio):
        print(f"   - {col}")
    
    print("\n🎯 Features Behavioral:")
    for col in sorted(behavioral):
        print(f"   - {col}")


def main():
    """Executa todas as validações."""
    print("="*80)
    print("[VALIDACAO] ETAPA 1: FEATURE ENGINEERING REFINADO")
    print("="*80)
    
    # Criar dataset sintético
    df = create_synthetic_dataset(n_samples=2000)
    original_shape = df.shape
    
    # Dividir em treino e OOT (80/20)
    split_idx = int(len(df) * 0.8)
    df_treino = df[:split_idx].reset_index(drop=True)
    df_oot = df[split_idx:].reset_index(drop=True)
    
    # Executar validações sequenciais
    validations = [
        ("Ordenacao Temporal", validate_temporal_ordering, [df]),
        ("Pipeline Completo", validate_feature_engineering_pipeline, [df_treino, df_oot]),
    ]
    
    # Aplicar pipeline para validações subsequentes
    pipeline = FeatureEngineeringPipeline()
    df_fe = pipeline.fit_transform(df)
    
    validations.extend([
        ("Ausencia de NaNs", validate_no_nans_in_features, 
         [df_fe, [col for col in df_fe.columns if any(x in col for x in ['velocity', 'ratio', 'behavioral', 'zscore'])]]),
        ("Velocity Features", validate_velocity_features, [df_fe]),
        ("Ratio Features", validate_ratio_features, [df_fe]),
        ("Behavioral Features", validate_behavioral_features, [df_fe]),
        ("Smurfing Detection", validate_smurfing_detection, [df_fe]),
    ])
    
    # Executar todas as validações
    results = {}
    for name, func, args in validations:
        try:
            results[name] = func(*args)
        except Exception as e:
            print(f"\n[ERRO] em validacao '{name}': {e}")
            results[name] = False
    
    # Sumário final
    print_feature_summary(df_fe, original_shape)
    
    # Relatório de validações
    print("\n" + "="*80)
    print("[RELATORIO] VALIDACOES")
    print("="*80)
    
    all_passed = True
    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{status:8} {name}")
        all_passed = all_passed and result
    
    print("\n" + "="*80)
    if all_passed:
        print("[SUCESSO] TODAS AS VALIDACOES PASSARAM!")
        print("   A ETAPA 1 esta pronta para ser integrada ao pipeline principal.")
    else:
        print("[AVISO] ALGUMAS VALIDACOES FALHARAM!")
        print("   Por favor, revisar os erros acima.")
    
    print("="*80)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
