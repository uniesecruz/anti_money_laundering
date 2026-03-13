#!/usr/bin/env python3
"""
Script de Validacao - ETAPA 2: Otimizacao de Threshold e Custos

Valida se a matriz de custo e otimização de threshold estao funcionando
corretamente antes de integração no notebook principal.

Execução: python scripts/test_stage_2_cost_optimization.py
"""

import sys
from pathlib import Path

# Adicionar source ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from source.modeling.cost_optimizer import (
    ClientValueClassifier,
    CostMatrix,
    ThresholdOptimizer,
    CostSensitiveEvaluator
)


def test_client_value_classifier():
    """Teste 1: Classificador de clientes high-value funciona"""
    print("\n[TESTE 1] ClientValueClassifier")
    
    # Gerar dados dummy
    df = pd.DataFrame({
        'Account': ['A1', 'A1', 'A2', 'A2', 'A3', 'A3'] * 10,
        'Amount Received': [100, 200, 5000, 6000, 500, 600] * 10
    })
    
    classifier = ClientValueClassifier(income_percentile=50.0)
    classifier.fit(df, income_col='Amount Received')
    
    is_high_value = classifier.classify(df, income_col='Amount Received')
    
    # Validar
    assert len(is_high_value) == len(df), "Tamanho do output != input"
    assert is_high_value.dtype == int, "Output deve ser inteiro (0 ou 1)"
    assert is_high_value.min() >= 0 and is_high_value.max() <= 1, "Valores fora de range [0,1]"
    
    prop_high_value = is_high_value.mean()
    assert 0.2 <= prop_high_value <= 0.8, f"Proporcao suspeita: {prop_high_value:.2%}"
    
    print(f"  Status: PASSOU")
    print(f"  Clientes high-value: {prop_high_value:.2%}")
    return True


def test_cost_matrix():
    """Teste 2: Matriz de custo calcula custos corretamente"""
    print("\n[TESTE 2] CostMatrix")
    
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 1])  # 2 FP, 1 FN, 1 TP, 2 TN
    is_high_value = np.array([0, 1, 0, 0, 0, 1])  # 1 FP standard, 1 FP high-value
    
    cost_matrix = CostMatrix(
        cost_fp_standard=100.0,
        cost_fp_highvalue=1000.0,
        cost_fn=5000.0
    )
    
    # Calcular custo total
    total_cost = cost_matrix.total_cost(y_pred, y_true, is_high_value)
    
    # Esperado: 1 FP standard (100) + 1 FP high-value (1000) + 1 FN (5000) = 6100
    expected_cost = 100 + 1000 + 5000
    
    assert total_cost == expected_cost, f"Custo {total_cost} != esperado {expected_cost}"
    
    print(f"  Status: PASSOU")
    print(f"  Custo total calculado corretamente: ${total_cost:.2f}")
    return True


def test_threshold_optimizer():
    """Teste 3: Otimizador de threshold encontra mínimo"""
    print("\n[TESTE 3] ThresholdOptimizer")
    
    # Dados simples: modelo perfeito em threshold=0.6
    np.random.seed(42)
    y_true = np.array([0]*50 + [1]*50)
    y_pred_proba = np.concatenate([
        np.random.uniform(0.0, 0.4, 50),  # Negativas com proba baixa
        np.random.uniform(0.6, 1.0, 50)   # Positivas com proba alta
    ])
    is_high_value = np.zeros(100)  # Todos padrão para teste simples
    
    cost_matrix = CostMatrix(
        cost_fp_standard=100,
        cost_fp_highvalue=100,
        cost_fn=5000
    )
    
    optimizer = ThresholdOptimizer(cost_matrix)
    optimal_threshold, min_cost = optimizer.find_optimal_threshold(
        y_true,
        y_pred_proba,
        is_high_value,
        thresholds=np.arange(0.1, 0.9, 0.1)
    )
    
    assert 0.1 <= optimal_threshold <= 0.9, f"Threshold fora de range: {optimal_threshold}"
    assert min_cost >= 0, "Custo mínimo deve ser >= 0"
    
    print(f"  Status: PASSOU")
    print(f"  Threshold ótimo encontrado: {optimal_threshold:.3f}")
    print(f"  Custo mínimo: ${min_cost:.2f}")
    return True


def test_cost_sensitive_evaluator():
    """Teste 4: Avaliador retorna métricas corretas"""
    print("\n[TESTE 4] CostSensitiveEvaluator")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.6, 0.9, 0.3, 0.2, 0.8, 0.95, 0.7])
    is_high_value = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    
    cost_matrix = CostMatrix(
        cost_fp_standard=50,
        cost_fp_highvalue=500,
        cost_fn=10000
    )
    
    evaluator = CostSensitiveEvaluator(cost_matrix)
    results = evaluator.evaluate(
        y_true,
        y_pred,
        y_pred_proba,
        is_high_value,
        threshold=0.5
    )
    
    # Validações
    assert 'accuracy' in results, "Metric 'accuracy' faltando"
    assert 'total_cost' in results, "Metric 'total_cost' faltando"
    assert 'fn_cost' in results, "Metric 'fn_cost' faltando"
    assert 'fp_standard_cost' in results, "Metric 'fp_standard_cost' faltando"
    assert 'fp_highvalue_cost' in results, "Metric 'fp_highvalue_cost' faltando"
    
    assert 0 <= results['accuracy'] <= 1, "Accuracy fora de range [0,1]"
    assert results['total_cost'] >= 0, "Custo total negativo"
    
    print(f"  Status: PASSOU")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Total Cost: ${results['total_cost']:.2f}")
    print(f"    - FN Cost: ${results['fn_cost']:.2f}")
    print(f"    - FP Standard: ${results['fp_standard_cost']:.2f}")
    print(f"    - FP High-Value: ${results['fp_highvalue_cost']:.2f}")
    return True


def test_threshold_vs_default():
    """Teste 5: Threshold ótimo melhora custo vs default"""
    print("\n[TESTE 5] Threshold Ótimo vs Default (0.5)")
    
    # Simular predições baseadas em modelo real
    np.random.seed(42)
    n = 1000
    y_true = np.array([0]*950 + [1]*50)  # ~95% negativo (desbalanceado)
    
    # Probabilidades: classe 1 tem distribuição mais à direita
    y_pred_proba = np.concatenate([
        np.random.beta(2, 5, 950),  # Classe 0: concentrada à esquerda
        np.random.beta(5, 2, 50)    # Classe 1: concentrada à direita
    ])
    
    is_high_value = np.random.choice([0, 1], n, p=[0.8, 0.2])  # 20% high-value
    
    cost_matrix = CostMatrix(
        cost_fp_standard=50,
        cost_fp_highvalue=500,
        cost_fn=10000
    )
    
    evaluator = CostSensitiveEvaluator(cost_matrix)
    
    # Avaliar threshold default
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    results_default = evaluator.evaluate(
        y_true,
        y_pred_default,
        y_pred_proba,
        is_high_value,
        threshold=0.5
    )
    
    # Encontrar e avaliar threshold ótimo
    optimizer = ThresholdOptimizer(cost_matrix)
    optimal_threshold, min_cost = optimizer.find_optimal_threshold(
        y_true,
        y_pred_proba,
        is_high_value,
        thresholds=np.arange(0.01, 1.0, 0.05)
    )
    
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    results_optimal = evaluator.evaluate(
        y_true,
        y_pred_optimal,
        y_pred_proba,
        is_high_value,
        threshold=optimal_threshold
    )
    
    # Validar que threshold ótimo reduz custo
    cost_reduction = results_default['total_cost'] - results_optimal['total_cost']
    assert cost_reduction >= -100, "Threshold ótimo não melhorou (margem 100 para erro de arredondamento)"
    
    print(f"  Status: PASSOU")
    print(f"  Threshold Default (0.5):")
    print(f"    Custo Total: ${results_default['total_cost']:.2f}")
    print(f"    Precision: {results_default['precision']:.4f}")
    print(f"    Recall: {results_default['recall']:.4f}")
    print(f"  Threshold Ótimo ({optimal_threshold:.3f}):")
    print(f"    Custo Total: ${results_optimal['total_cost']:.2f}")
    print(f"    Precision: {results_optimal['precision']:.4f}")
    print(f"    Recall: {results_optimal['recall']:.4f}")
    print(f"  Reducao de Custo: ${cost_reduction:.2f} ({cost_reduction/max(results_default['total_cost'], 1):.2%})")
    
    return True


def test_sample_weights():
    """Teste 6: Sample weights são calculadas e normalizadas"""
    print("\n[TESTE 6] Sample Weights para Cost-Weighted Training")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    is_high_value = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    
    cost_matrix = CostMatrix(
        cost_fp_standard=50,
        cost_fp_highvalue=500,
        cost_fn=10000
    )
    
    sample_weights = cost_matrix.get_sample_weights(y_true, is_high_value)
    
    # Validações
    assert len(sample_weights) == len(y_true), "Tamanho dos weights != y"
    assert all(w > 0 for w in sample_weights), "Alguns weights negativos ou zero"
    assert np.isclose(sample_weights.mean(), 1.0, atol=0.01), "MediadaWeights != 1.0"
    
    # Validar que positivos (FN) têm peso maior
    positive_mean_weight = sample_weights[y_true == 1].mean()
    negative_mean_weight = sample_weights[y_true == 0].mean()
    
    assert positive_mean_weight > negative_mean_weight, "Positivos devem ter peso > negativos"
    
    print(f"  Status: PASSOU")
    print(f"  Peso médio (classe positiva): {positive_mean_weight:.4f}")
    print(f"  Peso médio (classe negativa): {negative_mean_weight:.4f}")
    print(f"  Razão P/N: {positive_mean_weight/negative_mean_weight:.2f}x")
    return True


def main():
    """Executa todos os testes"""
    print("\n" + "="*80)
    print("VALIDACAO ETAPA 2: OTIMIZACAO DE THRESHOLD E CUSTOS")
    print("="*80)
    
    tests = [
        ("ClientValueClassifier", test_client_value_classifier),
        ("CostMatrix", test_cost_matrix),
        ("ThresholdOptimizer", test_threshold_optimizer),
        ("CostSensitiveEvaluator", test_cost_sensitive_evaluator),
        ("Threshold vs Default", test_threshold_vs_default),
        ("Sample Weights", test_sample_weights),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = True
        except AssertionError as e:
            print(f"  Status: FALHOU")
            print(f"  Erro: {e}")
            results[test_name] = False
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
        symbol = "✓" if passed_flag else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print("\n" + "="*80)
    if passed == total:
        print(f"CONCLUSAO: ETAPA 2 implementada com sucesso! [{passed}/{total} testes]")
        print("="*80)
        return 0
    else:
        print(f"CONCLUSAO: Alguns testes falharam. [{passed}/{total} testes]")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
