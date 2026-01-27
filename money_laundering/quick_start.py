#!/usr/bin/env python
"""
Quick Start - Arquitetura Híbrida PySpark → Pandas

Script de exemplo mostrando o fluxo completo de uso.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
import sys

# Adicionar source ao path
sys.path.append(str(Path(__file__).parent))

from loguru import logger


def main():
    """Fluxo completo da arquitetura híbrida."""
    
    logger.info("="*80)
    logger.info("QUICK START - ARQUITETURA HÍBRIDA")
    logger.info("="*80)
    
    # Verificar estrutura
    proj_root = Path(__file__).parent
    sampled_path = proj_root / 'data' / 'interim' / 'HI-Large_sampled.csv'
    trans_path = proj_root / 'data' / 'external' / 'HI-Large_Trans.csv'
    
    logger.info("\n📋 CHECKLIST:")
    
    # 1. Verificar dados originais
    if trans_path.exists():
        size_gb = trans_path.stat().st_size / (1024**3)
        logger.success(f"✓ HI-Large_Trans.csv encontrado ({size_gb:.2f} GB)")
    else:
        logger.warning(f"✗ HI-Large_Trans.csv não encontrado em data/external/")
        logger.info("\nPara testar, use datasets menores primeiro:")
        logger.info("  python source/dataset.py  # com dataset='LI-Medium'")
        return
    
    # 2. Verificar amostra
    if sampled_path.exists():
        size_mb = sampled_path.stat().st_size / (1024**2)
        logger.success(f"✓ HI-Large_sampled.csv existe ({size_mb:.2f} MB)")
        logger.info("\nVocê pode pular o Passo 1 e ir direto para o Passo 2!")
    else:
        logger.warning(f"✗ HI-Large_sampled.csv não encontrado")
    
    # 3. Instruções
    logger.info("\n" + "="*80)
    logger.info("📚 INSTRUÇÕES DE USO:")
    logger.info("="*80)
    
    if not sampled_path.exists():
        logger.info("\n🚀 PASSO 1: Gerar Amostra PySpark (EXECUTE UMA VEZ)")
        logger.info("-" * 80)
        logger.info("Este passo lê os 5 GB de dados, faz validação estatística,")
        logger.info("e salva uma amostra de ~200 MB representativa da população.")
        logger.info("")
        logger.info("  python source/spark_sampler.py")
        logger.info("")
        logger.info("⏱️  Tempo estimado: 5-15 minutos")
        logger.info("📊 Output: data/interim/HI-Large_sampled.csv")
        logger.info("")
    
    logger.info("📊 PASSO 2: Processar com Pandas (EXECUTE SEMPRE)")
    logger.info("-" * 80)
    logger.info("Este passo carrega a amostra (~200 MB) e faz o split Treino/OOT.")
    logger.info("")
    logger.info("  python source/dataset.py")
    logger.info("")
    logger.info("⏱️  Tempo estimado: 30 segundos - 2 minutos")
    logger.info("📊 Output: data/processed/df_treino.csv e df_oot.csv")
    logger.info("")
    
    logger.info("🔬 PASSO 3: Feature Engineering & Modelagem")
    logger.info("-" * 80)
    logger.info("Após ter df_treino.csv e df_oot.csv, prossiga normalmente:")
    logger.info("")
    logger.info("  jupyter lab notebooks/05_feature_engineering.ipynb")
    logger.info("  jupyter lab notebooks/08_treinamento_timeseriesplit.ipynb")
    logger.info("")
    
    # 4. Dicas
    logger.info("="*80)
    logger.info("💡 DICAS:")
    logger.info("="*80)
    logger.info("")
    logger.info("1. Validar setup ANTES de começar:")
    logger.info("     python validate_hybrid_setup.py")
    logger.info("")
    logger.info("2. Para datasets pequenos (LI-Medium), desabilite Spark:")
    logger.info("     # Em source/dataset.py, linha ~235:")
    logger.info("     main(dataset='LI-Medium', use_spark_sample=False)")
    logger.info("")
    logger.info("3. Tutorial interativo completo:")
    logger.info("     jupyter lab notebooks/00_arquitetura_hibrida_tutorial.ipynb")
    logger.info("")
    logger.info("4. Documentação detalhada:")
    logger.info("     cat ARQUITETURA_HIBRIDA.md")
    logger.info("     cat SUMARIO_TECNICO.md")
    logger.info("")
    
    logger.info("="*80)
    logger.success("PRONTO PARA COMEÇAR! 🚀")
    logger.info("="*80)


if __name__ == '__main__':
    main()
