"""
Script de Preparação de Dados para Detecção de Lavagem de Dinheiro

Este script processa os dados brutos e cria os conjuntos de treino e OOT (Out-of-Time).
Utiliza apenas caminhos relativos para garantir portabilidade.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))

from source.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT


def load_and_enrich_data(
    accounts_path: Path,
    trans_path: Path
) -> pd.DataFrame:
    """
    Carrega e enriquece dados de transações com informações das contas.
    
    Args:
        accounts_path: Caminho para arquivo de contas
        trans_path: Caminho para arquivo de transações
    
    Returns:
        DataFrame de transações enriquecido
    """
    logger.info(f"Carregando dados de {accounts_path.name} e {trans_path.name}...")
    
    # Carregar arquivos
    accounts_df = pd.read_csv(accounts_path)
    trans_df = pd.read_csv(trans_path)
    
    logger.info(f"  Contas:      {accounts_df.shape}")
    logger.info(f"  Transações:  {trans_df.shape}")
    
    # Renomear colunas de transações para clareza
    trans_df.columns = [
        'Timestamp', 'From Bank', 'From Account', 'To Bank', 'To Account',
        'Amount Received', 'Receiving Currency', 'Amount Paid',
        'Payment Currency', 'Payment Format', 'Is Laundering'
    ]
    
    # 1. Juntar com informações da conta de origem (FROM)
    trans_enriched = pd.merge(
        trans_df,
        accounts_df,
        left_on=['From Bank', 'From Account'],
        right_on=['Bank ID', 'Account Number'],
        how='left'
    )
    
    # Renomear colunas FROM
    trans_enriched = trans_enriched.rename(columns={
        'Bank Name': 'From Bank Name',
        'Entity ID': 'From Entity ID',
        'Entity Name': 'From Entity Name'
    })
    
    # 2. Juntar com informações da conta de destino (TO)
    trans_enriched = pd.merge(
        trans_enriched,
        accounts_df,
        left_on=['To Bank', 'To Account'],
        right_on=['Bank ID', 'Account Number'],
        how='left',
        suffixes=('', '_To')
    )
    
    # Renomear colunas TO
    trans_enriched = trans_enriched.rename(columns={
        'Bank Name': 'To Bank Name',
        'Entity ID': 'To Entity ID',
        'Entity Name': 'To Entity Name'
    })
    
    logger.success(f"Dados enriquecidos! Shape: {trans_enriched.shape}")
    
    return trans_enriched


def split_train_oot(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    time_col: str = 'Timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide dados em treino e OOT (Out-of-Time).
    
    Se time_col está disponível, faz divisão temporal.
    Caso contrário, faz divisão aleatória estratificada.
    
    Args:
        df: DataFrame completo
        test_size: Proporção para OOT
        random_state: Seed para reprodutibilidade
        time_col: Coluna temporal para ordenação
    
    Returns:
        Tuple (df_treino, df_oot)
    """
    logger.info(f"Dividindo dados em treino ({1-test_size:.0%}) e OOT ({test_size:.0%})...")
    
    # Verificar se existe coluna temporal
    if time_col in df.columns:
        logger.info(f"Usando divisão TEMPORAL baseada em '{time_col}'")
        
        # Converter para datetime se necessário
        if df[time_col].dtype == 'object':
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Ordenar por tempo
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        # Calcular ponto de corte
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        df_treino = df_sorted.iloc[:split_idx].copy()
        df_oot = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"  Período Treino: {df_treino[time_col].min()} a {df_treino[time_col].max()}")
        logger.info(f"  Período OOT:    {df_oot[time_col].min()} a {df_oot[time_col].max()}")
        
    else:
        logger.warning(f"Coluna '{time_col}' não encontrada. Usando divisão ALEATÓRIA.")
        
        # Divisão aleatória estratificada
        target_col = 'Is Laundering'
        
        df_treino, df_oot = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col] if target_col in df.columns else None
        )
    
    logger.success(f"Divisão concluída!")
    logger.info(f"  Treino: {df_treino.shape}")
    logger.info(f"  OOT:    {df_oot.shape}")
    
    # Distribuição do target
    target_col = 'Is Laundering'
    if target_col in df_treino.columns:
        logger.info(f"  Distribuição Treino: {df_treino[target_col].value_counts().to_dict()}")
        logger.info(f"  Distribuição OOT:    {df_oot[target_col].value_counts().to_dict()}")
    
    return df_treino, df_oot


def save_data(
    df_treino: pd.DataFrame,
    df_oot: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Salva dados processados.
    
    Args:
        df_treino: DataFrame de treino
        df_oot: DataFrame OOT
        output_dir: Diretório de saída
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar DataFrames completos
    treino_path = output_dir / 'df_treino.csv'
    oot_path = output_dir / 'df_oot.csv'
    
    df_treino.to_csv(treino_path, index=False)
    df_oot.to_csv(oot_path, index=False)
    
    logger.success(f"Dados salvos em {output_dir}")
    logger.info(f"  - {treino_path.name}")
    logger.info(f"  - {oot_path.name}")
    
    # Também salvar dados brutos enriquecidos (união de treino + oot)
    df_all = pd.concat([df_treino, df_oot], ignore_index=True)
    raw_enriched_path = RAW_DATA_DIR / 'trans_enriched.csv'
    df_all.to_csv(raw_enriched_path, index=False)
    
    logger.info(f"  - {raw_enriched_path.relative_to(PROJ_ROOT)}")


def main(
    dataset: str = 'LI-Medium',
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Função principal de preparação de dados.
    
    Args:
        dataset: Nome do dataset (LI-Small, LI-Medium, LI-Large, HI-Small, HI-Medium, HI-Large)
        test_size: Proporção para OOT
        random_state: Seed para reprodutibilidade
    """
    logger.info("="*80)
    logger.info("PREPARAÇÃO DE DADOS - DETECÇÃO DE LAVAGEM DE DINHEIRO")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Diretório do projeto: {PROJ_ROOT}")
    
    # Caminhos relativos usando pathlib
    accounts_path = EXTERNAL_DATA_DIR / f'{dataset}_accounts.csv'
    trans_path = EXTERNAL_DATA_DIR / f'{dataset}_Trans.csv'
    
    # Verificar se arquivos existem
    if not accounts_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {accounts_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {trans_path}")
    
    # 1. Carregar e enriquecer dados
    df_enriched = load_and_enrich_data(accounts_path, trans_path)
    
    # 2. Dividir em treino e OOT
    df_treino, df_oot = split_train_oot(
        df_enriched,
        test_size=test_size,
        random_state=random_state
    )
    
    # 3. Salvar dados processados
    save_data(df_treino, df_oot, PROCESSED_DATA_DIR)
    
    # 4. Estatísticas finais
    logger.info("\n" + "="*80)
    logger.info("ESTATÍSTICAS DOS DADOS")
    logger.info("="*80)
    
    logger.info(f"\nColunas ({len(df_treino.columns)}):")
    for col in df_treino.columns:
        dtype = df_treino[col].dtype
        n_unique = df_treino[col].nunique()
        n_missing = df_treino[col].isna().sum()
        pct_missing = (n_missing / len(df_treino)) * 100
        
        logger.info(
            f"  {col:30s} | {str(dtype):15s} | "
            f"Únicos: {n_unique:8d} | Missing: {n_missing:8d} ({pct_missing:5.2f}%)"
        )
    
    logger.info("\n" + "="*80)
    logger.success("PREPARAÇÃO DE DADOS CONCLUÍDA COM SUCESSO!")
    logger.info("="*80)
    
    logger.info(f"\nPróximos passos:")
    logger.info(f"  1. Execute o script de treinamento:")
    logger.info(f"     python source/modeling/train_pipeline.py")


if __name__ == '__main__':
    # Usar caminhos relativos à raiz do projeto
    main(
        dataset='LI-Medium',
        test_size=0.2,
        random_state=42
    )
