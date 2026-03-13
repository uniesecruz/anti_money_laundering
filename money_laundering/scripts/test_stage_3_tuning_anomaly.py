#!/usr/bin/env python3
"""
Script de Validação - ETAPA 3: Tuning e Modelos de Anomalia

Valida se Optuna + Isolation Forest estão funcionando corretamente.

Execução: python scripts/test_stage_3_tuning_anomaly.py
"""

import sys
from pathlib import Path

# Adicionar source ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from source.modeling.optuna_tuner import (
    AnomalyDetector,
    OptunaHyperparametersOptimizer,
    AMLTunerPipeline
)


def test_anomaly_detector():
    """Teste 1: AnomalyDetector funciona"""
    print("\n[TESTE 1] AnomalyDetector")
    
    # Gerar dados dummy
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feat_{i}' for i in range(10)])
    
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X)
    
    # Predictions
    predictions = detector.detect(X)
    anomaly_ratio = detector.get_anomaly_ratio(X)
    
    # Validações
    assert len(predictions) == len(X), "Tamanho != input"
    assert set(predictions) == {-1, 1}, "Predictions devem ser -1 ou 1"
    assert 0 <= anomaly_ratio <= 1, "Anomaly ratio fora de range"
    assert 0.02 <= anomaly_ratio <= 0.08, f"Ratio {anomaly_ratio} não está perto de 5%"
    
    # Anomaly scores
    scores = detector.get_anomaly_score(X)
    assert len(scores) == len(X), "Scores != input"
    assert 0 <= scores.min() and scores.max() <= 1, "Scores fora de [0, 1]"
    
    print(f"  Status: PASSOU")
    print(f"  Anomaly ratio: {anomaly_ratio:.2%}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return True


def test_optuna_objective_functions():
    """Teste 2: Objective functions de Optuna retornam scores válidos"""
    print("\n[TESTE 2] OptunaHyperparametersOptimizer (Objective Functions)")
    
    np.random.seed(42)
    
    # Dados pequenos para teste rápido
    n_samples = 500
    X_train = np.random.randn(n_samples, 20)
    y_train = np.random.binomial(1, 0.1, n_samples)
    
    X_val = np.random.randn(100, 20)
    y_val = np.random.binomial(1, 0.1, 100)
    
    tscv = TimeSeriesSplit(n_splits=2)
    
    # Testar com XGBoost
    try:
        optimizer = OptunaHyperparametersOptimizer(
            model_type='xgboost',
            n_trials=2,  # Apenas 2 trials para teste rápido
            random_state=42
        )
        
        result = optimizer.optimize(X_train, y_train, X_val, y_val, tscv)
        
        assert 'best_params' in result, "best_params faltando"
        assert 'best_value' in result, "best_value faltando"
        assert isinstance(result['best_value'], (float, np.floating)), "best_value não é float"
        assert 0 <= result['best_value'] <= 1, "best_value fora de range [0, 1]"
        
        # Validar parâmetros
        params = result['best_params']
        assert 'scale_pos_weight' in params, "scale_pos_weight faltando (CRÍTICO)"
        assert params['scale_pos_weight'] > 0, "scale_pos_weight inválido"
        
        print(f"  Status: PASSOU (XGBoost)")
        print(f"  Best F1-Score: {result['best_value']:.4f}")
        print(f"  scale_pos_weight: {params['scale_pos_weight']:.4f}")
        
    except Exception as e:
        print(f"  Status: FALHOU")
        print(f"  Erro: {e}")
        return False
    
    return True


def test_aml_tuner_pipeline_integration():
    """Teste 3: Pipeline inteiro funciona end-to-end"""
    print("\n[TESTE 3] AMLTunerPipeline (End-to-End)")
    
    np.random.seed(42)
    
    # Dados de teste
    n_train = 500
    n_val = 100
    n_features = 20
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.binomial(1, 0.1, n_train))
    
    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.binomial(1, 0.1, n_val))
    
    tscv = TimeSeriesSplit(n_splits=2)
    
    try:
        # Criar e executar pipeline
        pipeline = AMLTunerPipeline(
            model_type='xgboost',
            n_optuna_trials=3,  # Teste rápido
            anomaly_contamination=0.05,
            random_state=42
        )
        
        pipeline.fit(X_train, y_train, X_val, y_val, tscv)
        
        # Validações
        assert pipeline.best_model is not None, "best_model é None"
        assert pipeline.X_train_augmented is not None, "X_train_augmented é None"
        assert 'anomaly_score' in pipeline.X_train_augmented.columns, "anomaly_score feature faltando"
        
        # Testar predições
        y_pred = pipeline.predict(X_val)
        y_pred_proba = pipeline.predict_proba(X_val)
        
        assert len(y_pred) == len(X_val), "Tamanho predições != X_val"
        assert len(y_pred_proba) == len(X_val), "Tamanho proba != X_val"
        # Check that predictions are binary (0 or 1 only)
        unique_preds = set(np.unique(y_pred))
        assert unique_preds.issubset({0, 1}), f"Predictions contêm valores inválidos: {unique_preds}"
        assert 0 <= y_pred_proba.min() and y_pred_proba.max() <= 1, "Proba fora de [0, 1]"
        
        print(f"  Status: PASSOU")
        print(f"  Best params scale_pos_weight: {pipeline.optimizer.best_params['scale_pos_weight']:.4f}")
        print(f"  Features após augmentation: {pipeline.X_train_augmented.shape[1]}")
        print(f"  Anomaly score range: [{pipeline.X_train_augmented['anomaly_score'].min():.4f}, {pipeline.X_train_augmented['anomaly_score'].max():.4f}]")
        
    except Exception as e:
        print(f"  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        return False
    
    return True


def test_anomaly_feature_quality():
    """Teste 4: Anomaly features têm qualidade/variância"""
    print("\n[TESTE 4] Qualidade das Anomaly Features")
    
    np.random.seed(42)
    
    # Dados com outliers intencionais
    n_normal = 950
    n_anomaly = 50
    
    X_normal = np.random.randn(n_normal, 15) * 1.0
    X_anomaly = np.random.randn(n_anomaly, 15) * 5.0  # Muita variância = anomalia
    
    X = pd.DataFrame(
        np.vstack([X_normal, X_anomaly]),
        columns=[f'feat_{i}' for i in range(15)]
    )
    
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X)
    scores = detector.get_anomaly_score(X)
    
    # Scores das anomalias devem ser maiores (em média)
    scores_normal = scores[:n_normal]
    scores_anomaly = scores[n_normal:]
    
    mean_normal = scores_normal.mean()
    mean_anomaly = scores_anomaly.mean()
    
    assert mean_anomaly > mean_normal, f"Anomaly scores não distinguem well: {mean_anomaly} vs {mean_normal}"
    
    print(f"  Status: PASSOU")
    print(f"  Mean anomaly score (normal): {mean_normal:.4f}")
    print(f"  Mean anomaly score (anomaly): {mean_anomaly:.4f}")
    print(f"  Razão: {mean_anomaly / mean_normal:.2f}x")
    return True


def test_scale_pos_weight_optimization():
    """Teste 5: scale_pos_weight é otimizado diferente que padrão"""
    print("\n[TESTE 5] scale_pos_weight Optimization (vs Default)")
    
    np.random.seed(42)
    
    # Dados altamente desbalanceados
    n = 1000
    X = pd.DataFrame(np.random.randn(n, 15), columns=[f'feat_{i}' for i in range(15)])
    y = pd.Series([0]*950 + [1]*50)  # 95% vs 5%
    
    X_train, X_val = X.iloc[:700], X.iloc[700:]
    y_train, y_val = y.iloc[:700], y.iloc[700:]
    
    tscv = TimeSeriesSplit(n_splits=2)
    
    try:
        optimizer = OptunaHyperparametersOptimizer(
            model_type='xgboost',
            n_trials=3,
            random_state=42
        )
        
        result = optimizer.optimize(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            tscv=tscv
        )
        
        optimized_spw = result['best_params']['scale_pos_weight']
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        # scale_pos_weight ótimo deve estar perto do imbalance ratio
        # (realmente pode variar bastante, mas deve estar no ballpark)
        assert optimized_spw > 1, "scale_pos_weight deve ser > 1"
        assert optimized_spw < imbalance_ratio * 5, "scale_pos_weight muito alto"
        
        print(f"  Status: PASSOU")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"  Optimized scale_pos_weight: {optimized_spw:.4f}")
        print(f"  Razão SPW/Imbalance: {optimized_spw/imbalance_ratio:.2f}x")
        
    except Exception as e:
        print(f"  Status: FALHOU")
        print(f"  Erro: {e}")
        return False
    
    return True


def main():
    """Executa todos os testes"""
    print("\n" + "="*80)
    print("VALIDACAO ETAPA 3: TUNING COM OPTUNA + ISOLATION FOREST")
    print("="*80)
    
    tests = [
        ("AnomalyDetector", test_anomaly_detector),
        ("Optuna Objective Functions", test_optuna_objective_functions),
        ("AMLTunerPipeline Integration", test_aml_tuner_pipeline_integration),
        ("Anomaly Feature Quality", test_anomaly_feature_quality),
        ("scale_pos_weight Optimization", test_scale_pos_weight_optimization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  Status: ERRO")
            print(f"  Exceção: {type(e).__name__}: {e}")
            results[test_name] = False
    
    # Sumário
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
        print(f"CONCLUSAO: ETAPA 3 implementada com sucesso! [{passed}/{total} testes]")
        print("="*80)
        return 0
    else:
        print(f"CONCLUSAO: Alguns testes falharam. [{passed}/{total} testes]")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
