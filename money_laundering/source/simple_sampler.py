"""
Simple Sampler - Amostragem com Pandas (sem PySpark)

Versão simplificada para validação do pipeline usando apenas Pandas.
Lê o arquivo em chunks e cria amostra aleatória estratificada.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np
from loguru import logger

# Adicionar source ao path
sys.path.append(str(Path(__file__).parent.parent))

from source.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR


def sample_large_csv_in_chunks(
    trans_path: Path,
    accounts_path: Path,
    output_path: Path,
    sample_fraction: float = 0.01,  # 1% dos dados
    chunk_size: int = 100000,
    random_state: int = 42
) -> None:
    """
    Cria amostra de dataset grande usando chunks do Pandas.
    
    Args:
        trans_path: Caminho para transações
        accounts_path: Caminho para contas
        output_path: Caminho para salvar amostra
        sample_fraction: Fração da amostra (0.01 = 1%)
        chunk_size: Tamanho do chunk
        random_state: Seed para reprodutibilidade
    """
    logger.info("="*80)
    logger.info("AMOSTRAGEM SIMPLES COM PANDAS (SEM PYSPARK)")
    logger.info("="*80)
    
    # 1. Carregar contas (menor, cabe na memória)
    logger.info(f"\nCarregando contas: {accounts_path.name}")
    accounts_df = pd.read_csv(accounts_path)
    logger.success(f"✓ Contas carregadas: {accounts_df.shape}")
    
    # 2. Processar transações em chunks
    logger.info(f"\nProcessando transações em chunks de {chunk_size:,} linhas...")
    logger.info(f"Amostrando {sample_fraction:.1%} dos dados")
    
    sample_chunks = []
    total_rows = 0
    chunk_count = 0
    
    np.random.seed(random_state)
    
    try:
        for chunk in pd.read_csv(trans_path, chunksize=chunk_size):
            chunk_count += 1
            total_rows += len(chunk)
            
            # Renomear colunas
            chunk.columns = [
                'Timestamp', 'From Bank', 'From Account', 'To Bank', 'To Account',
                'Amount Received', 'Receiving Currency', 'Amount Paid',
                'Payment Currency', 'Payment Format', 'Is Laundering'
            ]
            
            # Amostragem estratificada por 'Is Laundering'
            if 'Is Laundering' in chunk.columns:
                sampled = chunk.groupby('Is Laundering', group_keys=False).apply(
                    lambda x: x.sample(frac=sample_fraction, random_state=random_state)
                )
            else:
                sampled = chunk.sample(frac=sample_fraction, random_state=random_state)
            
            sample_chunks.append(sampled)
            
            if chunk_count % 10 == 0:
                logger.info(f"  Processados {total_rows:,} linhas ({chunk_count} chunks)...")
    
    except Exception as e:
        logger.error(f"Erro ao processar chunks: {e}")
        logger.info(f"Usando {len(sample_chunks)} chunks processados até agora...")
    
    # 3. Combinar chunks amostrados
    logger.info("\nCombinando chunks amostrados...")
    sample_df = pd.concat(sample_chunks, ignore_index=True)
    
    logger.success(f"✓ Amostra combinada: {sample_df.shape}")
    logger.info(f"  Total de linhas processadas: {total_rows:,}")
    logger.info(f"  Amostra final: {len(sample_df):,} linhas")
    
    # 4. Enriquecer com dados de contas
    logger.info("\nEnriquecendo com dados de contas...")
    
    # Join com FROM
    sample_enriched = pd.merge(
        sample_df,
        accounts_df,
        left_on=['From Bank', 'From Account'],
        right_on=['Bank ID', 'Account Number'],
        how='left'
    )
    
    sample_enriched = sample_enriched.rename(columns={
        'Bank Name': 'From Bank Name',
        'Entity ID': 'From Entity ID',
        'Entity Name': 'From Entity Name'
    })
    
    # Join com TO
    sample_enriched = pd.merge(
        sample_enriched,
        accounts_df,
        left_on=['To Bank', 'To Account'],
        right_on=['Bank ID', 'Account Number'],
        how='left',
        suffixes=('', '_To')
    )
    
    sample_enriched = sample_enriched.rename(columns={
        'Bank Name': 'To Bank Name',
        'Entity ID': 'To Entity ID',
        'Entity Name': 'To Entity Name'
    })
    
    # Limpar colunas duplicadas
    cols_to_drop = [col for col in sample_enriched.columns if col.endswith('_To') or col in ['Bank ID', 'Account Number']]
    sample_enriched = sample_enriched.drop(columns=cols_to_drop, errors='ignore')
    
    logger.success(f"✓ Dados enriquecidos: {sample_enriched.shape}")
    
    # 5. Salvar amostra
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_enriched.to_csv(output_path, index=False)
    
    logger.success(f"\n{'='*80}")
    logger.success(f"AMOSTRA SALVA COM SUCESSO!")
    logger.success(f"{'='*80}")
    logger.info(f"Arquivo: {output_path}")
    logger.info(f"Shape: {sample_enriched.shape}")
    logger.info(f"Tamanho: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    if 'Is Laundering' in sample_enriched.columns:
        logger.info(f"Distribuição Target: {sample_enriched['Is Laundering'].value_counts().to_dict()}")


def main():
    """Função principal."""
    logger.info("="*80)
    logger.info("SIMPLE SAMPLER - VALIDAÇÃO DO PIPELINE")
    logger.info("="*80)
    
    # Caminhos
    accounts_path = EXTERNAL_DATA_DIR / 'HI-Large_accounts.csv'
    trans_path = EXTERNAL_DATA_DIR / 'HI-Large_Trans.csv'
    output_path = INTERIM_DATA_DIR / 'HI-Large_sampled.csv'
    
    # Validar existência
    if not accounts_path.exists():
        logger.error(f"Arquivo não encontrado: {accounts_path}")
        return
    
    if not trans_path.exists():
        logger.error(f"Arquivo não encontrado: {trans_path}")
        return
    
    # Gerar amostra
    sample_large_csv_in_chunks(
        trans_path=trans_path,
        accounts_path=accounts_path,
        output_path=output_path,
        sample_fraction=0.01,  # 1% dos dados (ajuste conforme necessário)
        chunk_size=100000,
        random_state=42
    )
    
    logger.info("\n" + "="*80)
    logger.success("PRÓXIMOS PASSOS:")
    logger.info("="*80)
    logger.info("  python source/dataset.py")


if __name__ == '__main__':
    main()
