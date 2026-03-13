"""
ETAPA 4: Testes de Tratamento de Dados e Anti-Leakage

Testes para validar:
1. YeoJohnsonTransformerSafe: Lambda calculado APENAS no treino
2. TargetEncoderRegularized: Target Encoding com regularizacao
3. Anti-leakage: Fit no treino, Transform em OOT

Autor: TCC - AML Detection
Data: 2026-03-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Adicionar projeto ao path
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))

from source.preprocessing import YeoJohnsonTransformerSafe, TargetEncoderRegularized

def create_synthetic_data(n_train: int = 1000, n_oot: int = 500, seed: int = 42) -> tuple:
    """Cria dados sinteticos para testes."""
    np.random.seed(seed)
    
    # Dataset de treino
    X_train = pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, n_train),
        'numeric_2': np.random.exponential(1, n_train),
        'numeric_3': np.random.uniform(-10, 10, n_train),
        'category_low': np.random.choice(['A', 'B', 'C', 'D'], n_train),
        'category_high': np.random.choice([f'cat_{i}' for i in range(100)], n_train),
    })
    
    # Target correlacionado com numericas
    y_train = (
        (X_train['numeric_1'] > 0).astype(int) * 0.4 +
        (X_train['numeric_2'] > 1).astype(int) * 0.3 +
        np.random.randint(0, 2, n_train) * 0.3
    ).astype(int)
    
    # Dataset de OOT
    X_oot = pd.DataFrame({
        'numeric_1': np.random.normal(0.2, 1.1, n_oot),
        'numeric_2': np.random.exponential(1.2, n_oot),
        'numeric_3': np.random.uniform(-10, 10, n_oot),
        'category_low': np.random.choice(['A', 'B', 'C', 'D'], n_oot),
        'category_high': np.random.choice([f'cat_{i}' for i in range(100)], n_oot),
    })
    
    y_oot = (
        (X_oot['numeric_1'] > 0).astype(int) * 0.4 +
        (X_oot['numeric_2'] > 1).astype(int) * 0.3 +
        np.random.randint(0, 2, n_oot) * 0.3
    ).astype(int)
    
    # Dataset com categorias desconhecidas
    X_test_unknown = pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, 200),
        'numeric_2': np.random.exponential(1, 200),
        'numeric_3': np.random.uniform(-10, 10, 200),
        'category_low': np.random.choice(['A', 'B', 'C', 'E'], 200),
        'category_high': np.random.choice([f'cat_{i}' for i in range(50, 150)], 200),
    })
    
    return X_train, y_train, X_oot, y_oot, X_test_unknown


def test_yeo_johnson_anti_leakage():
    """Teste 1: Lambda calculado APENAS em treino"""
    print("\n" + "="*80)
    print("[TESTE 1] YeoJohnsonTransformerSafe - Anti-Leakage")
    print("="*80)
    
    try:
        X_train, y_train, X_oot, y_oot, _ = create_synthetic_data()
        X_train_numeric = X_train[['numeric_1', 'numeric_2', 'numeric_3']].copy()
        X_oot_numeric = X_oot[['numeric_1', 'numeric_2', 'numeric_3']].copy()
        
        # Fit APENAS no treino
        transformer = YeoJohnsonTransformerSafe()
        transformer.fit(X_train_numeric)
        
        lambda_train = transformer.lambda_.copy()
        print(f"  Lambda calculado (treino):")
        for col, lam in lambda_train.items():
            print(f"    - {col}: L={lam:.6f}")
        
        # Transform treino e OOT
        X_train_transformed = transformer.transform(X_train_numeric)
        X_oot_transformed = transformer.transform(X_oot_numeric)
        
        # Validacoes
        assert X_train_transformed.shape == X_train_numeric.shape
        assert X_oot_transformed.shape == X_oot_numeric.shape
        assert transformer.lambda_ == lambda_train
        
        means_train = X_train_transformed.mean()
        means_oot = X_oot_transformed.mean()
        
        print(f"\n  Validacoes:")
        print(f"    OK - Lambda nao mudou apos transform")
        print(f"    OK - Shape preservado: {X_train_numeric.shape} -> {X_train_transformed.shape}")
        print(f"    OK - Medias (treino): {means_train.values}")
        print(f"    OK - Medias (OOT): {means_oot.values}")
        
        assert not np.array_equal(X_train_numeric.values, X_train_transformed.values)
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_encoder_regularization():
    """Teste 2: Regularizacao funcionando"""
    print("\n" + "="*80)
    print("[TESTE 2] TargetEncoderRegularized - Regularizacao")
    print("="*80)
    
    try:
        X_train, y_train, X_oot, y_oot, X_test_unknown = create_synthetic_data()
        X_train_cat = X_train[['category_high']].copy()
        X_oot_cat = X_oot[['category_high']].copy()
        X_test_cat = X_test_unknown[['category_high']].copy()
        
        encoder = TargetEncoderRegularized(smoothing=1.0, min_samples_leaf=5)
        encoder.fit(X_train_cat, y_train)
        
        X_train_encoded = encoder.transform(X_train_cat)
        X_oot_encoded = encoder.transform(X_oot_cat)
        X_test_encoded = encoder.transform(X_test_cat)
        
        print(f"\n  Analise de Regularizacao:")
        
        unique_encodings = len(X_train_encoded['category_high'].unique())
        print(f"    - Unique encodings (treino): {unique_encodings}")
        assert unique_encodings <= len(X_train_cat['category_high'].unique())
        
        min_val = X_train_encoded['category_high'].min()
        max_val = X_train_encoded['category_high'].max()
        print(f"    - Range de valores: [{min_val:.4f}, {max_val:.4f}]")
        assert 0 <= min_val <= 1 and 0 <= max_val <= 1
        
        unknown_values = X_test_encoded[X_test_encoded['category_high'].isnull()].shape[0]
        print(f"    - Categorias desconhecidas: {unknown_values}")
        assert unknown_values == 0
        
        assert not np.array_equal(X_oot_cat.values, X_oot_encoded.values)
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_data_leakage():
    """Teste 3: Zero Data Leakage"""
    print("\n" + "="*80)
    print("[TESTE 3] Zero Data Leakage - Fit Treino, Transform OOT")
    print("="*80)
    
    try:
        X_train, y_train, X_oot, y_oot, _ = create_synthetic_data(seed=42)
        
        print("\n  Testando Yeo-Johnson Anti-Leakage:")
        
        yj = YeoJohnsonTransformerSafe()
        yj.fit(X_train[['numeric_1', 'numeric_2']])
        lambda_after_fit = yj.lambda_.copy()
        
        X_oot_transformed = yj.transform(X_oot[['numeric_1', 'numeric_2']])
        lambda_after_oot = yj.lambda_.copy()
        
        assert lambda_after_fit == lambda_after_oot
        print(f"    OK - Lambda nao mudou apos transform em OOT")
        
        print("\n  Testando Target Encoding Anti-Leakage:")
        
        te = TargetEncoderRegularized(smoothing=1.0, min_samples_leaf=10)
        te.fit(X_train[['category_high']], y_train)
        mappings_after_fit = te.mappings_.copy()
        
        X_oot_encoded = te.transform(X_oot[['category_high']])
        mappings_after_oot = te.mappings_.copy()
        
        assert mappings_after_fit == mappings_after_oot
        print(f"    OK - Mappings nao mudaram apos transform em OOT")
        
        print("\n  Testando Determinismo:")
        
        yj2 = YeoJohnsonTransformerSafe()
        yj2.fit(X_train[['numeric_1', 'numeric_2']])
        
        assert yj.lambda_ == yj2.lambda_
        print(f"    OK - Lambda e determinisitco (mesmos dados)")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_encoder_smoothing_effect():
    """Teste 4: Efeito do Smoothing"""
    print("\n" + "="*80)
    print("[TESTE 4] Target Encoder - Efeito do Smoothing")
    print("="*80)
    
    try:
        X_train, y_train, _, _, _ = create_synthetic_data(n_train=2000)
        X_train_cat = X_train[['category_high']].copy()
        
        encoders = {
            'low (0.1)': TargetEncoderRegularized(smoothing=0.1, min_samples_leaf=1),
            'medium (1.0)': TargetEncoderRegularized(smoothing=1.0, min_samples_leaf=1),
            'high (10.0)': TargetEncoderRegularized(smoothing=10.0, min_samples_leaf=1),
        }
        
        results = {}
        
        for name, encoder in encoders.items():
            encoder.fit(X_train_cat, y_train)
            X_encoded = encoder.transform(X_train_cat)
            
            variance = X_encoded['category_high'].var()
            results[name] = variance
            
            print(f"  Smoothing {name}: Variance = {variance:.6f}")
        
        var_low = results['low (0.1)']
        var_med = results['medium (1.0)']
        var_high = results['high (10.0)']
        
        assert var_low >= var_med >= var_high
        print(f"\n    OK - Variancia diminui com smoothing crescente")
        print(f"    OK - Regularizacao funcionando corretamente")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_high_cardinality_handling():
    """Teste 5: Alta Cardinalidade (100+ categorias)"""
    print("\n" + "="*80)
    print("[TESTE 5] Target Encoder - Alta Cardinalidade (100+ categorias)")
    print("="*80)
    
    try:
        np.random.seed(42)
        n_categories = 500
        n_samples = 10000
        
        categories = [f'cat_{i}' for i in range(n_categories)]
        X = pd.DataFrame({
            'high_card': np.random.choice(categories, n_samples)
        })
        y = pd.Series(np.random.binomial(1, 0.1, n_samples))
        
        encoder = TargetEncoderRegularized(smoothing=1.0, min_samples_leaf=5)
        encoder.fit(X, y)
        X_encoded = encoder.transform(X)
        
        print(f"\n  Analise de Alta Cardinalidade:")
        print(f"    - Categories no treino: {n_categories}")
        print(f"    - Samples no treino: {n_samples}")
        print(f"    - Unique encodings: {len(X_encoded['high_card'].unique())}")
        
        assert len(X_encoded['high_card'].unique()) <= n_categories
        
        X_unknown = pd.DataFrame({
            'high_card': [f'cat_{i}' for i in range(n_categories, n_categories + 100)]
        })
        X_unknown_encoded = encoder.transform(X_unknown)
        
        unique_unknown = X_unknown_encoded['high_card'].nunique()
        print(f"    - Categorias desconhecidas: 100")
        print(f"    - Unique encodings (unknown): {unique_unknown}")
        
        assert unique_unknown == 1
        print(f"    - OK - Todas categorias desconhecidas")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "="*80)
    print("VALIDACAO ETAPA 4: TRATAMENTO DE DADOS E ANTI-LEAKAGE")
    print("="*80)
    
    results = {}
    
    results['Yeo-Johnson Anti-Leakage'] = test_yeo_johnson_anti_leakage()
    results['Target Encoder Regularization'] = test_target_encoder_regularization()
    results['Zero Data Leakage'] = test_no_data_leakage()
    results['Target Encoder Smoothing'] = test_target_encoder_smoothing_effect()
    results['High Cardinality Handling'] = test_high_cardinality_handling()
    
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "PASSOU" if passed_flag else "FALHOU"
        symbol = "[OK]" if passed_flag else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")
    
    print("\n" + "="*80)
    if passed == total:
        print(f"CONCLUSAO: ETAPA 4 implementada com sucesso! [{passed}/{total} testes]")
        print("="*80)
        return 0
    else:
        print(f"CONCLUSAO: Alguns testes falharam. [{passed}/{total} testes]")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

