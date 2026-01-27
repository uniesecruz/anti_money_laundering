"""
Análise de Qualidade da Amostra PySpark

Script para verificar a qualidade estatística da amostra gerada,
comparando-a com a população original (se disponível).

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

# Adicionar source ao path
sys.path.append(str(Path(__file__).parent))

from source.config import INTERIM_DATA_DIR, EXTERNAL_DATA_DIR


def compare_distributions(
    sample_df: pd.DataFrame,
    population_df: pd.DataFrame = None,
    col: str = 'Is Laundering'
) -> dict:
    """
    Compara distribuições entre amostra e população.
    
    Args:
        sample_df: DataFrame da amostra
        population_df: DataFrame da população (opcional)
        col: Coluna para comparar
        
    Returns:
        Dicionário com estatísticas
    """
    results = {
        'column': col,
        'sample_stats': {},
        'population_stats': {},
        'test_results': {}
    }
    
    # Estatísticas da amostra
    if col in sample_df.columns:
        if pd.api.types.is_numeric_dtype(sample_df[col]):
            results['sample_stats'] = {
                'mean': sample_df[col].mean(),
                'std': sample_df[col].std(),
                'median': sample_df[col].median(),
                'min': sample_df[col].min(),
                'max': sample_df[col].max()
            }
        else:
            value_counts = sample_df[col].value_counts()
            results['sample_stats'] = {
                'counts': value_counts.to_dict(),
                'proportions': (value_counts / len(sample_df)).to_dict()
            }
    
    # Estatísticas da população (se disponível)
    if population_df is not None and col in population_df.columns:
        if pd.api.types.is_numeric_dtype(population_df[col]):
            results['population_stats'] = {
                'mean': population_df[col].mean(),
                'std': population_df[col].std(),
                'median': population_df[col].median(),
                'min': population_df[col].min(),
                'max': population_df[col].max()
            }
            
            # Z-test
            pop_mean = results['population_stats']['mean']
            pop_std = results['population_stats']['std']
            sample_mean = results['sample_stats']['mean']
            n_sample = len(sample_df)
            
            if pop_std > 0:
                se = pop_std / np.sqrt(n_sample)
                z_score = (sample_mean - pop_mean) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                results['test_results'] = {
                    'test': 'Z-test',
                    'z_score': z_score,
                    'p_value': p_value,
                    'is_valid': p_value > 0.05
                }
        else:
            value_counts = population_df[col].value_counts()
            results['population_stats'] = {
                'counts': value_counts.to_dict(),
                'proportions': (value_counts / len(population_df)).to_dict()
            }
            
            # Chi-quadrado test
            sample_counts = sample_df[col].value_counts().to_dict()
            pop_props = results['population_stats']['proportions']
            
            categories = list(pop_props.keys())
            observed = [sample_counts.get(cat, 0) for cat in categories]
            expected = [pop_props[cat] * len(sample_df) for cat in categories]
            
            if sum(expected) > 0:
                chi2_stat, p_value = stats.chisquare(observed, expected)
                
                results['test_results'] = {
                    'test': 'Chi-quadrado',
                    'chi2_stat': chi2_stat,
                    'p_value': p_value,
                    'is_valid': p_value > 0.05
                }
    
    return results


def analyze_sample(
    sample_path: Path,
    population_trans_path: Path = None,
    population_accounts_path: Path = None
) -> None:
    """
    Analisa qualidade estatística da amostra.
    
    Args:
        sample_path: Caminho para amostra
        population_trans_path: Caminho para transações originais (opcional)
        population_accounts_path: Caminho para contas originais (opcional)
    """
    logger.info("="*80)
    logger.info("ANÁLISE DE QUALIDADE DA AMOSTRA")
    logger.info("="*80)
    
    # 1. Carregar amostra
    if not sample_path.exists():
        logger.error(f"Amostra não encontrada: {sample_path}")
        return
    
    logger.info(f"\nCarregando amostra: {sample_path.name}")
    sample_df = pd.read_csv(sample_path)
    
    logger.success(f"✓ Amostra carregada")
    logger.info(f"  Shape: {sample_df.shape}")
    logger.info(f"  Memória: {sample_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # 2. Estatísticas básicas da amostra
    logger.info("\n" + "="*80)
    logger.info("ESTATÍSTICAS DESCRITIVAS DA AMOSTRA")
    logger.info("="*80)
    
    # Target distribution
    if 'Is Laundering' in sample_df.columns:
        logger.info("\n📊 Distribuição do Target (Is Laundering):")
        target_counts = sample_df['Is Laundering'].value_counts()
        for value, count in target_counts.items():
            pct = (count / len(sample_df)) * 100
            logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
    
    # Variáveis numéricas
    numeric_cols = ['Amount Received', 'Amount Paid']
    logger.info("\n📈 Variáveis Numéricas:")
    for col in numeric_cols:
        if col in sample_df.columns:
            logger.info(f"\n  {col}:")
            logger.info(f"    Média:    {sample_df[col].mean():,.2f}")
            logger.info(f"    Mediana:  {sample_df[col].median():,.2f}")
            logger.info(f"    Std Dev:  {sample_df[col].std():,.2f}")
            logger.info(f"    Min:      {sample_df[col].min():,.2f}")
            logger.info(f"    Max:      {sample_df[col].max():,.2f}")
    
    # Variáveis categóricas
    categorical_cols = ['Payment Format', 'From Bank', 'To Bank']
    logger.info("\n📊 Variáveis Categóricas:")
    for col in categorical_cols:
        if col in sample_df.columns:
            n_unique = sample_df[col].nunique()
            logger.info(f"  {col}: {n_unique} categorias únicas")
    
    # 3. Comparação com população (se disponível)
    if population_trans_path and population_trans_path.exists():
        logger.info("\n" + "="*80)
        logger.info("COMPARAÇÃO COM POPULAÇÃO ORIGINAL")
        logger.info("="*80)
        logger.warning("\n⚠️  AVISO: Carregar população completa pode consumir muita memória!")
        logger.info("Tentando carregar apenas as colunas necessárias...\n")
        
        try:
            # Carregar apenas algumas linhas para inferir tipos
            sample_pop = pd.read_csv(population_trans_path, nrows=1000)
            
            # Colunas de interesse
            cols_to_load = ['Is Laundering', 'Amount Received', 'Amount Paid', 'Payment Format']
            cols_to_load = [c for c in cols_to_load if c in sample_pop.columns]
            
            logger.info(f"Carregando colunas: {cols_to_load}")
            
            # Carregar população (apenas colunas relevantes)
            pop_df = pd.read_csv(population_trans_path, usecols=cols_to_load)
            
            logger.success(f"✓ População carregada: {pop_df.shape}")
            
            # Comparar distribuições
            for col in cols_to_load:
                if col in sample_df.columns and col in pop_df.columns:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Coluna: {col}")
                    logger.info(f"{'='*80}")
                    
                    result = compare_distributions(sample_df, pop_df, col)
                    
                    # Mostrar estatísticas
                    logger.info("\nAmostra:")
                    for key, value in result['sample_stats'].items():
                        if isinstance(value, dict):
                            logger.info(f"  {key}:")
                            for k, v in list(value.items())[:5]:  # Top 5
                                logger.info(f"    {k}: {v}")
                        else:
                            logger.info(f"  {key}: {value:.4f}")
                    
                    logger.info("\nPopulação:")
                    for key, value in result['population_stats'].items():
                        if isinstance(value, dict):
                            logger.info(f"  {key}:")
                            for k, v in list(value.items())[:5]:  # Top 5
                                logger.info(f"    {k}: {v}")
                        else:
                            logger.info(f"  {key}: {value:.4f}")
                    
                    # Mostrar teste estatístico
                    if result['test_results']:
                        test = result['test_results']
                        logger.info(f"\n{test['test']}:")
                        
                        if 'z_score' in test:
                            logger.info(f"  Z-score:  {test['z_score']:.4f}")
                        if 'chi2_stat' in test:
                            logger.info(f"  χ²:       {test['chi2_stat']:.4f}")
                        
                        logger.info(f"  p-value:  {test['p_value']:.4f}")
                        
                        if test['is_valid']:
                            logger.success(f"  ✓ VÁLIDA (p > 0.05)")
                        else:
                            logger.warning(f"  ✗ INVÁLIDA (p ≤ 0.05)")
        
        except Exception as e:
            logger.error(f"Erro ao carregar população: {e}")
            logger.info("Continuando apenas com análise da amostra...")
    
    # 4. Conclusão
    logger.info("\n" + "="*80)
    logger.success("ANÁLISE CONCLUÍDA")
    logger.info("="*80)
    logger.info("\n📋 Resumo:")
    logger.info(f"  - Amostra: {len(sample_df):,} registros")
    logger.info(f"  - Colunas: {len(sample_df.columns)}")
    logger.info(f"  - Memória: {sample_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    if 'Is Laundering' in sample_df.columns:
        target_dist = sample_df['Is Laundering'].value_counts()
        logger.info(f"  - Target balanceado: {target_dist.to_dict()}")
    
    logger.info("\n✅ A amostra está pronta para uso no pipeline Pandas!")
    logger.info("   Execute: python source/dataset.py")


def main():
    """Função principal."""
    # Caminhos
    sample_path = INTERIM_DATA_DIR / 'HI-Large_sampled.csv'
    trans_path = EXTERNAL_DATA_DIR / 'HI-Large_Trans.csv'
    accounts_path = EXTERNAL_DATA_DIR / 'HI-Large_accounts.csv'
    
    # Analisar (comparação com população é opcional)
    analyze_sample(
        sample_path=sample_path,
        population_trans_path=trans_path if trans_path.exists() else None,
        population_accounts_path=accounts_path if accounts_path.exists() else None
    )


if __name__ == '__main__':
    main()
