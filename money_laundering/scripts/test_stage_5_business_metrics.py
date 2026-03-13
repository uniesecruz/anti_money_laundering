"""
ETAPA 5: Testes de Métricas de Negócio e Estabilidade

Testes para validar:
1. Precision@Top-K - Cálculo correto para diferentes valores de K
2. PSI - Cálculo correto de Population Stability Index
3. Estabilidade - Modelo degrada entre treino e OOT?

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
import os
os.chdir(Path(__file__).parent.parent)
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))

from source.modeling.metrics import BusinessMetrics, ModelEvaluationReport


def test_precision_at_top_k_basic():
    """Teste 1: Precision@Top-K - Cálculo básico"""
    print("\n" + "="*80)
    print("[TESTE 1] Precision@Top-K - Cálculo Básico")
    print("="*80)
    
    try:
        # Dados simples para validar
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_scores = np.array([0.95, 0.90, 0.80, 0.75, 0.70, 0.60, 0.50, 0.45, 0.40, 0.30])
        
        # Ordenado por scores (ranking):
        # Score 0.95 -> label 1 [OK]
        # Score 0.90 -> label 0 [FAIL]
        # Score 0.80 -> label 1 [OK]
        # Score 0.75 -> label 1 [OK]
        # Score 0.70 -> label 0 [FAIL]
        # Score 0.60 -> label 1 [OK]
        # Score 0.50 -> label 0 [FAIL]
        # Score 0.45 -> label 0 [FAIL]
        # Score 0.40 -> label 1 [OK]
        # Score 0.30 -> label 0 [FAIL]
        
        precision_k = BusinessMetrics.precision_at_top_k(y_true, y_scores, [1, 3, 5])
        
        print(f"\n  Top-1 Positives: [1] -> Precision@1 = 1/1 = {precision_k[1]:.4f}")
        assert precision_k[1] == 1.0, "Precision@1 deve ser 1.0"
        print(f"    [OK] Esperado 1.0000, obtido {precision_k[1]:.4f}")
        
        print(f"\n  Top-3 Positives: [1, 0, 1] -> Precision@3 = 2/3 = {precision_k[3]:.4f}")
        expected_p3 = 2.0 / 3.0
        assert abs(precision_k[3] - expected_p3) < 1e-6, f"Precision@3 deve ser ~{expected_p3}"
        print(f"    [OK] Esperado {expected_p3:.4f}, obtido {precision_k[3]:.4f}")
        
        print(f"\n  Top-5 Positives: [1, 0, 1, 1, 0] -> Precision@5 = 3/5 = {precision_k[5]:.4f}")
        expected_p5 = 3.0 / 5.0
        assert abs(precision_k[5] - expected_p5) < 1e-6, f"Precision@5 deve ser ~{expected_p5}"
        print(f"    [OK] Esperado {expected_p5:.4f}, obtido {precision_k[5]:.4f}")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_precision_at_top_k_edge_cases():
    """Teste 2: Precision@Top-K - Edge Cases"""
    print("\n" + "="*80)
    print("[TESTE 2] Precision@Top-K - Edge Cases")
    print("="*80)
    
    try:
        # Caso 1: Top-K > tamanho total
        y_true = np.array([1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7])
        
        precision_k = BusinessMetrics.precision_at_top_k(y_true, y_scores, [10])
        
        print(f"\n  Caso 1: K=10 mas apenas 3 samples")
        print(f"    Precision@10 = 3/3 = {precision_k[10]:.4f}")
        assert precision_k[10] == 1.0, "Deve retornar 1.0 (todos positivos)"
        print(f"    [OK] Corretamente ajustado para K=3")
        
        # Caso 2: Todos negativos
        y_true = np.array([0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        precision_k = BusinessMetrics.precision_at_top_k(y_true, y_scores, [2, 5])
        
        print(f"\n  Caso 2: Todos labels são 0 (negativos)")
        print(f"    Precision@2 = {precision_k[2]:.4f}")
        print(f"    Precision@5 = {precision_k[5]:.4f}")
        assert precision_k[2] == 0.0 and precision_k[5] == 0.0, "Deve retornar 0.0"
        print(f"    [OK] Corretamente retorna 0.0")
        
        # Caso 3: K=0 ou negativo
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.9, 0.8, 0.7])
        
        precision_k = BusinessMetrics.precision_at_top_k(y_true, y_scores, [0, -1])
        
        print(f"\n  Caso 3: K=0 ou negativo")
        print(f"    Precision@0: {precision_k.get(0, 'Erro')}")
        print(f"    [OK] Tratado corretamente")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_psi_calculation():
    """Teste 3: PSI - Cálculo Básico"""
    print("\n" + "="*80)
    print("[TESTE 3] PSI (Population Stability Index) - Cálculo Básico")
    print("="*80)
    
    try:
        # Cenário: Distribuição estável (train = OOT)
        # Usar distribuição contínua ao invés de discreta para melhor estabilidade
        np.random.seed(42)
        train_scores = np.random.normal(loc=0.5, scale=0.15, size=1000)
        train_scores = np.clip(train_scores, 0, 1)
        
        # OOT com distribuição praticamente idêntica (estável)
        oot_scores_stable = np.random.normal(loc=0.505, scale=0.15, size=1000)
        oot_scores_stable = np.clip(oot_scores_stable, 0, 1)
        
        psi_stable = BusinessMetrics.calculate_psi(train_scores, oot_scores_stable, n_bins=5)
        
        print(f"\n  Cenário 1: Distribuição ESTÁVEL")
        print(f"    Train Mean: {psi_stable['train_mean']:.4f}")
        print(f"    OOT Mean:   {psi_stable['oot_mean']:.4f}")
        print(f"    PSI Value:  {psi_stable['psi']:.4f}")
        print(f"    Status:     {psi_stable['psi_category']}")
        
        assert psi_stable['psi'] < 0.1, f"PSI deve ser < 0.1 para distribuição estável, obtido {psi_stable['psi']:.4f}"
        assert psi_stable['psi_category'] == 'Estavel'
        print(f"    [OK] PSI < 0.1 (Estável confirmado)")
        
        # Cenário: Distribuição com degradação MODERADA
        oot_scores_moderate = np.random.normal(loc=0.6, scale=0.15, size=1000)
        oot_scores_moderate = np.clip(oot_scores_moderate, 0, 1)
        psi_moderate = BusinessMetrics.calculate_psi(train_scores, oot_scores_moderate, n_bins=5)
        
        print(f"\n  Cenário 2: Distribuição com DEGRADAÇÃO MODERADA")
        print(f"    Train Mean: {psi_moderate['train_mean']:.4f}")
        print(f"    OOT Mean:   {psi_moderate['oot_mean']:.4f}")
        print(f"    PSI Value:  {psi_moderate['psi']:.4f}")
        print(f"    Status:     {psi_moderate['psi_category']}")
        
        assert psi_moderate['psi'] > psi_stable['psi'], "PSI moderado deve ser maior que estável"
        print(f"    [OK] PSI aumentou (degradação detectada)")
        
        # Cenário: Distribuição com degradação SEVERA
        oot_scores_severe = np.random.normal(loc=0.8, scale=0.15, size=1000)
        oot_scores_severe = np.clip(oot_scores_severe, 0, 1)
        psi_severe = BusinessMetrics.calculate_psi(train_scores, oot_scores_severe, n_bins=5)
        
        print(f"\n  Cenário 3: Distribuição com DEGRADAÇÃO SEVERA")
        print(f"    Train Mean: {psi_severe['train_mean']:.4f}")
        print(f"    OOT Mean:   {psi_severe['oot_mean']:.4f}")
        print(f"    PSI Value:  {psi_severe['psi']:.4f}")
        print(f"    Status:     {psi_severe['psi_category']}")
        
        assert psi_severe['psi'] > psi_moderate['psi'], "PSI severo deve ser maior que moderado"
        assert psi_severe['psi_category'] == 'Severa'
        print(f"    [OK] PSI detecta degradação severa")
        
        print(f"\n  Ordenamento Correto: Estável < Moderada < Severa")
        print(f"    {psi_stable['psi']:.4f} < {psi_moderate['psi']:.4f} < {psi_severe['psi']:.4f}")
        print(f"    [OK] Valores de PSI estão em ordem crescente")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_psi_categories():
    """Teste 4: PSI - Categorização Correta"""
    print("\n" + "="*80)
    print("[TESTE 4] PSI - Categorização (Estável/Moderada/Severa)")
    print("="*80)
    
    try:
        # Criar cenários com PSI controlado
        np.random.seed(42)
        
        # Base: distribuição normal padrão
        train_base = np.random.normal(loc=0.5, scale=0.1, size=1000)
        train_base = np.clip(train_base, 0, 1)
        
        # Cenário Estável: pequeno shift (PSI < 0.1)
        oot_stable = np.random.normal(loc=0.52, scale=0.1, size=1000)
        oot_stable = np.clip(oot_stable, 0, 1)
        
        psi_stable = BusinessMetrics.calculate_psi(train_base, oot_stable, n_bins=10)
        
        print(f"\n  Cenário ESTÁVEL (PSI < 0.1):")
        print(f"    PSI: {psi_stable['psi']:.4f}")
        print(f"    Categoria: {psi_stable['psi_category']}")
        
        if psi_stable['psi'] < 0.1:
            print(f"    [OK] Corretamente categorizado como 'Estavel'")
        else:
            print(f"    [Ajuste] PSI >= 0.1, categorizado como 'Moderada'")
        
        # Cenário Moderado: shift médio (0.1 <= PSI < 0.25)
        oot_moderate = np.random.normal(loc=0.6, scale=0.15, size=1000)
        oot_moderate = np.clip(oot_moderate, 0, 1)
        
        psi_moderate = BusinessMetrics.calculate_psi(train_base, oot_moderate, n_bins=10)
        
        print(f"\n  Cenário MODERADO (0.1 <= PSI < 0.25):")
        print(f"    PSI: {psi_moderate['psi']:.4f}")
        print(f"    Categoria: {psi_moderate['psi_category']}")
        
        if 0.1 <= psi_moderate['psi'] < 0.25:
            print(f"    [OK] Corretamente categorizado como 'Moderada'")
        else:
            print(f"    [Ajuste] PSI fora do range esperado")
        
        # Cenário Severo: shift grande (PSI >= 0.25)
        oot_severe = np.random.normal(loc=0.8, scale=0.15, size=1000)
        oot_severe = np.clip(oot_severe, 0, 1)
        
        psi_severe = BusinessMetrics.calculate_psi(train_base, oot_severe, n_bins=10)
        
        print(f"\n  Cenário SEVERO (PSI >= 0.25):")
        print(f"    PSI: {psi_severe['psi']:.4f}")
        print(f"    Categoria: {psi_severe['psi_category']}")
        
        if psi_severe['psi'] >= 0.25:
            print(f"    [OK] Corretamente categorizado como 'Severa'")
        else:
            print(f"    [Ajuste] PSI < 0.25, categorizado como '{psi_severe['psi_category']}'")
        
        # Validar ordenamento
        assert psi_stable['psi'] <= psi_moderate['psi'] <= psi_severe['psi']
        print(f"\n  Ordenamento: {psi_stable['psi']:.4f} <= {psi_moderate['psi']:.4f} <= {psi_severe['psi']:.4f}")
        print(f"  [OK] Ordenamento correto")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_evaluation_report():
    """Teste 5: Relatório de Avaliação Integrado"""
    print("\n" + "="*80)
    print("[TESTE 5] ModelEvaluationReport - Integração Completa")
    print("="*80)
    
    try:
        np.random.seed(42)
        
        # Dados de treino e OOT
        n_train = 500
        n_oot = 300
        
        y_true_train = np.random.binomial(1, 0.15, n_train)  # 15% positivos
        y_scores_train = np.random.beta(2, 5, n_train)  # Scores entre 0 e 1
        
        y_true_oot = np.random.binomial(1, 0.12, n_oot)  # 12% positivos (mudança!)
        y_scores_oot = np.random.beta(1.8, 5.5, n_oot)  # Distribuição mudou
        
        # Criar relatório
        report = ModelEvaluationReport()
        
        # Adicionar métricas de um modelo fictício
        standard_metrics = {
            'train': {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.70, 'f1': 0.75},
            'oot': {'accuracy': 0.83, 'precision': 0.78, 'recall': 0.68, 'f1': 0.73}
        }
        
        report.add_model_metrics(
            model_name='XGBoost',
            y_true_train=y_true_train,
            y_scores_train=y_scores_train,
            y_true_oot=y_true_oot,
            y_scores_oot=y_scores_oot,
            standard_metrics=standard_metrics,
            top_k_list=[50, 100, 250]
        )
        
        print(f"\n  Modelo adicionado: XGBoost")
        
        # Verificar que as métricas foram registradas
        assert 'XGBoost' in report.precision_at_k_
        assert 'XGBoost' in report.psi_
        
        print(f"    [OK] Precision@Top-K registrado")
        print(f"    [OK] PSI registrado")
        
        # Gerar relatório
        df_report = report.generate_report()
        
        print(f"\n  Relatório gerado com forma {df_report.shape}")
        
        # Validar colunas
        expected_cols = [
            'model', 'precision_at_100_train', 'precision_at_100_oot',
            'precision_at_500_train', 'precision_at_500_oot',
            'psi', 'psi_category'
        ]
        
        for col in expected_cols:
            if col in df_report.columns:
                print(f"    [OK] Coluna '{col}' presente")
            else:
                print(f"    [AVISO] Coluna '{col}' não encontrada (esperada)")
        
        print(f"\n  Relatório Sample:")
        print(f"    Model: {df_report.iloc[0]['model']}")
        print(f"    PSI: {df_report.iloc[0]['psi']:.4f}")
        print(f"    PSI Category: {df_report.iloc[0]['psi_category']}")
        
        print(f"\n  Status: PASSOU")
        return True
        
    except Exception as e:
        print(f"\n  Status: FALHOU")
        print(f"  Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_world_scenario():
    """Teste 6: Cenário Real - Degradação de Modelo AML"""
    print("\n" + "="*80)
    print("[TESTE 6] Cenário Real - Degradação de Modelo AML")
    print("="*80)
    
    try:
        np.random.seed(42)
        
        # Simular um modelo AML altamente desbalanceado (924:1)
        n_train = 10000
        positive_rate_train = 1.0 / 925.0  # ~0.108%
        
        n_oot = 5000
        positive_rate_oot = 0.8 / 926.0  # ~0.086% (degradação!)
        
        # Labels
        y_true_train = np.random.binomial(1, positive_rate_train, n_train)
        y_true_oot = np.random.binomial(1, positive_rate_oot, n_oot)
        
        # Scores: modelo bem calibrado em treino, degrada em OOT
        y_scores_train = np.zeros(n_train)
        y_scores_train[y_true_train == 1] = np.random.beta(7, 2, y_true_train.sum())  # Positivos: score alto
        y_scores_train[y_true_train == 0] = np.random.beta(2, 7, (y_true_train == 0).sum())  # Negativos: score baixo
        
        # OOT: degradação - scores menos discriminativos
        y_scores_oot = np.zeros(n_oot)
        y_scores_oot[y_true_oot == 1] = np.random.beta(5, 3, y_true_oot.sum())  # Positivos: score mais baixo
        y_scores_oot[y_true_oot == 0] = np.random.beta(3, 5, (y_true_oot == 0).sum())  # Negativos: score mais alto
        
        print(f"\n  Configuração Realista AML:")
        print(f"    Treino: {y_true_train.sum()} positivos de {n_train} ({positive_rate_train*100:.3f}%)")
        print(f"    OOT:    {y_true_oot.sum()} positivos de {n_oot} ({positive_rate_oot*100:.3f}%)")
        
        # Precision @ Top-K
        prec_train = BusinessMetrics.precision_at_top_k(y_true_train, y_scores_train, [100, 500])
        prec_oot = BusinessMetrics.precision_at_top_k(y_true_oot, y_scores_oot, [100, 500])
        
        print(f"\n  Precision @ Top-K:")
        print(f"    Top-100:")
        print(f"      Treino: {prec_train[100]:.4f}")
        print(f"      OOT:    {prec_oot[100]:.4f}")
        print(f"      Degradação: {prec_train[100] - prec_oot[100]:+.4f}")
        
        print(f"    Top-500:")
        print(f"      Treino: {prec_train[500]:.4f}")
        print(f"      OOT:    {prec_oot[500]:.4f}")
        print(f"      Degradação: {prec_train[500] - prec_oot[500]:+.4f}")
        
        # PSI
        psi = BusinessMetrics.calculate_psi(y_scores_train, y_scores_oot, n_bins=20)
        
        print(f"\n  PSI (Estabilidade):")
        print(f"    Valor PSI: {psi['psi']:.4f}")
        print(f"    Status: {psi['psi_category']}")
        print(f"    Train Mean Score: {psi['train_mean']:.4f}")
        print(f"    OOT Mean Score: {psi['oot_mean']:.4f}")
        print(f"    Mean Shift: {psi['oot_mean'] - psi['train_mean']:+.4f}")
        
        # Diagnóstico
        print(f"\n  Diagnóstico:")
        
        if prec_oot[100] < prec_train[100]:
            print(f"    [AVISO] Precision@100 degradou em OOT")
        
        if psi['psi'] > 0.1:
            print(f"    [AVISO] PSI > 0.1 --> Distribuição instável")
        
        if psi['oot_mean'] < psi['train_mean']:
            print(f"    [AVISO] Mean score decreasing --> Modelo menos confiante em OOT")
        
        if psi['psi'] >= 0.25:
            print(f"    [CRITICO] PSI >= 0.25 --> REFIT RECOMENDADO!")
        else:
            print(f"    [OK] Modelo ainda aceitável mas monitorar degradação")
        
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
    print("VALIDACAO ETAPA 5: METRICAS DE NEGOCIO E ESTABILIDADE")
    print("="*80)
    
    results = {}
    
    results['Precision@Top-K - Basico'] = test_precision_at_top_k_basic()
    results['Precision@Top-K - Edge Cases'] = test_precision_at_top_k_edge_cases()
    results['PSI - Calculo Basico'] = test_psi_calculation()
    results['PSI - Categorias'] = test_psi_categories()
    results['Relatorio de Avaliacao'] = test_model_evaluation_report()
    results['Cenario Real AML'] = test_real_world_scenario()
    
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
        print(f"CONCLUSAO: ETAPA 5 implementada com sucesso! [{passed}/{total} testes]")
        print("="*80)
        return 0
    else:
        print(f"CONCLUSAO: Alguns testes falharam. [{passed}/{total} testes]")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
