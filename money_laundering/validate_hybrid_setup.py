"""
Script de Validação da Arquitetura Híbrida PySpark → Pandas

Verifica se todos os componentes estão instalados e configurados corretamente.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
import sys

# Cores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def check(condition: bool, message: str) -> bool:
    """Imprime resultado do check."""
    if condition:
        print(f"{Colors.GREEN}✓{Colors.RESET} {message}")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.RESET} {message}")
        return False

def main():
    """Valida setup da arquitetura híbrida."""
    
    print("="*80)
    print(f"{Colors.BLUE}VALIDAÇÃO DA ARQUITETURA HÍBRIDA PYSPARK → PANDAS{Colors.RESET}")
    print("="*80)
    
    all_good = True
    
    # 1. Verificar imports Python
    print(f"\n{Colors.YELLOW}1. Verificando dependências Python...{Colors.RESET}")
    
    try:
        import pandas as pd
        all_good &= check(True, f"pandas {pd.__version__} instalado")
    except ImportError:
        all_good &= check(False, "pandas NÃO instalado (pip install pandas)")
    
    try:
        import numpy as np
        all_good &= check(True, f"numpy {np.__version__} instalado")
    except ImportError:
        all_good &= check(False, "numpy NÃO instalado (pip install numpy)")
    
    try:
        from scipy import stats
        import scipy
        all_good &= check(True, f"scipy {scipy.__version__} instalado")
    except ImportError:
        all_good &= check(False, "scipy NÃO instalado (pip install scipy)")
    
    try:
        from loguru import logger
        all_good &= check(True, "loguru instalado")
    except ImportError:
        all_good &= check(False, "loguru NÃO instalado (pip install loguru)")
    
    try:
        from pyspark.sql import SparkSession
        import pyspark
        all_good &= check(True, f"pyspark {pyspark.__version__} instalado")
    except ImportError:
        all_good &= check(False, "pyspark NÃO instalado (pip install pyspark)")
    
    # 2. Verificar estrutura de diretórios
    print(f"\n{Colors.YELLOW}2. Verificando estrutura de diretórios...{Colors.RESET}")
    
    # Detectar raiz do projeto
    current_file = Path(__file__).resolve()
    proj_root = current_file.parent
    
    data_dir = proj_root / 'data'
    external_dir = data_dir / 'external'
    interim_dir = data_dir / 'interim'
    processed_dir = data_dir / 'processed'
    source_dir = proj_root / 'source'
    
    all_good &= check(data_dir.exists(), f"data/ existe")
    all_good &= check(external_dir.exists(), f"data/external/ existe")
    all_good &= check(interim_dir.exists(), f"data/interim/ existe")
    all_good &= check(processed_dir.exists(), f"data/processed/ existe")
    all_good &= check(source_dir.exists(), f"source/ existe")
    
    # 3. Verificar scripts principais
    print(f"\n{Colors.YELLOW}3. Verificando scripts da arquitetura...{Colors.RESET}")
    
    spark_sampler = source_dir / 'spark_sampler.py'
    dataset_script = source_dir / 'dataset.py'
    config_script = source_dir / 'config.py'
    
    all_good &= check(spark_sampler.exists(), "source/spark_sampler.py existe")
    all_good &= check(dataset_script.exists(), "source/dataset.py existe")
    all_good &= check(config_script.exists(), "source/config.py existe")
    
    # 4. Verificar dados originais
    print(f"\n{Colors.YELLOW}4. Verificando datasets originais...{Colors.RESET}")
    
    hi_large_trans = external_dir / 'HI-Large_Trans.csv'
    hi_large_accounts = external_dir / 'HI-Large_accounts.csv'
    
    trans_exists = hi_large_trans.exists()
    accounts_exists = hi_large_accounts.exists()
    
    if trans_exists:
        size_mb = hi_large_trans.stat().st_size / (1024**2)
        check(True, f"HI-Large_Trans.csv encontrado ({size_mb:.1f} MB)")
    else:
        check(False, "HI-Large_Trans.csv NÃO encontrado em data/external/")
    
    if accounts_exists:
        size_mb = hi_large_accounts.stat().st_size / (1024**2)
        check(True, f"HI-Large_accounts.csv encontrado ({size_mb:.1f} MB)")
    else:
        check(False, "HI-Large_accounts.csv NÃO encontrado em data/external/")
    
    all_good &= (trans_exists and accounts_exists)
    
    # 5. Verificar se amostra já existe
    print(f"\n{Colors.YELLOW}5. Verificando amostra PySpark...{Colors.RESET}")
    
    sampled_path = interim_dir / 'HI-Large_sampled.csv'
    
    if sampled_path.exists():
        size_mb = sampled_path.stat().st_size / (1024**2)
        print(f"{Colors.GREEN}✓{Colors.RESET} HI-Large_sampled.csv JÁ EXISTE ({size_mb:.1f} MB)")
        print(f"  → Você pode executar diretamente: python source/dataset.py")
    else:
        print(f"{Colors.YELLOW}⚠{Colors.RESET} HI-Large_sampled.csv não encontrado")
        print(f"  → Execute PRIMEIRO: python source/spark_sampler.py")
    
    # 6. Resumo final
    print("\n" + "="*80)
    if all_good:
        print(f"{Colors.GREEN}✓ TUDO PRONTO!{Colors.RESET} A arquitetura híbrida está configurada corretamente.")
        print("\n" + "="*80)
        print(f"{Colors.BLUE}PRÓXIMOS PASSOS:{Colors.RESET}")
        print("="*80)
        
        if not sampled_path.exists() and trans_exists and accounts_exists:
            print(f"\n{Colors.YELLOW}Passo 1:{Colors.RESET} Gerar amostra PySpark")
            print(f"  cd {proj_root}")
            print(f"  python source/spark_sampler.py")
            print(f"\n{Colors.YELLOW}Passo 2:{Colors.RESET} Processar com Pandas")
            print(f"  python source/dataset.py")
        elif sampled_path.exists():
            print(f"\n{Colors.GREEN}Amostra já existe!{Colors.RESET} Execute diretamente:")
            print(f"  cd {proj_root}")
            print(f"  python source/dataset.py")
        else:
            print(f"\n{Colors.RED}Datasets não encontrados.{Colors.RESET}")
            print(f"Copie HI-Large_Trans.csv e HI-Large_accounts.csv para:")
            print(f"  {external_dir}")
    else:
        print(f"{Colors.RED}✗ PROBLEMAS DETECTADOS!{Colors.RESET}")
        print("Corrija os itens marcados com ✗ acima antes de prosseguir.")
    
    print("="*80)

if __name__ == '__main__':
    main()
