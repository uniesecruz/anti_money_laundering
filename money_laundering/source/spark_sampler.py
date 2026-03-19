"""
PySpark Sampler - Amostragem Estatisticamente Validada para Big Data

Este script implementa uma arquitetura híbrida PySpark->Pandas para datasets massivos.
Realiza amostragem estratificada com validação estatística rigorosa (testes de hipótese).

Fluxo:
1. PySpark: Leitura e Join de datasets massivos
2. PySpark: Cálculo de estatísticas populacionais
3. PySpark: Geração de amostras estratificadas candidatas
4. Validação: Testes estatísticos (Z-test, Chi-quadrado) para garantir representatividade
5. Saída: CSV reduzido para processamento Pandas

Autor: TCC - Anti Money Laundering Detection  
Data: Janeiro 2026
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# Lazy loading de PySpark - evita erro de socketserver em Python 3.11+
# Imports globais que serão inicializados via _init_pyspark()
SparkSession = None
DataFrame = None
F = None
StructType = None
StructField = None
StringType = None
DoubleType = None
IntegerType = None

# Para type hints apenas (não executa em runtime)
if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame, functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

def _init_pyspark():
    """Inicializa imports de PySpark com lazy loading para evitar erro de socketserver."""
    global SparkSession, DataFrame, F, StructType, StructField, StringType, DoubleType, IntegerType
    
    if SparkSession is not None:  # Já inicializado
        return
    
    try:
        from pyspark.sql import SparkSession as _SparkSession
        from pyspark.sql import DataFrame as _DataFrame
        from pyspark.sql import functions as _F
        from pyspark.sql.types import (
            StructType as _StructType,
            StructField as _StructField,
            StringType as _StringType,
            DoubleType as _DoubleType,
            IntegerType as _IntegerType
        )
        
        SparkSession = _SparkSession
        DataFrame = _DataFrame
        F = _F
        StructType = _StructType
        StructField = _StructField
        StringType = _StringType
        DoubleType = _DoubleType
        IntegerType = _IntegerType
        
        logger.debug("✅ PySpark imports inicializados com sucesso (lazy loading)")
    except ImportError as e:
        logger.error(f"❌ PySpark não instalado! Execute: pip install pyspark\nErro: {e}")
        raise

# Importar configurações do projeto
sys.path.append(str(Path(__file__).parent.parent))
from source.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROJ_ROOT


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

SAMPLE_FRACTION = 0.10  # 10% dos dados (ajuste conforme necessário)
MAX_ITERATIONS = 10     # Máximo de tentativas para encontrar amostra válida
P_VALUE_THRESHOLD = 0.05  # Nível de significância (alpha = 5%)
RANDOM_SEED = 42

# Colunas críticas para validação estatística
NUMERIC_COLS_TO_VALIDATE = [
    'Amount Received',
    'Amount Paid'
]

CATEGORICAL_COLS_TO_VALIDATE = [
    'Payment Format',
    'From Bank',
    'To Bank'
]


# ============================================================================
# INICIALIZAÇÃO DO SPARK
# ============================================================================

def create_spark_session(app_name: str = "AntiMoneyLaundering-Sampler") -> "SparkSession":
    """
    Cria e configura sessão Spark otimizada para processamento local.
    
    Args:
        app_name: Nome da aplicação Spark
        
    Returns:
        SparkSession configurada
    """
    import os
    
    # Inicializar PySpark com lazy loading
    _init_pyspark()
    
    logger.info("Inicializando Spark Session...")
    
    # Configurar Python para o Spark usar o ambiente virtual correto
    python_path = sys.executable
    logger.info(f"  Python path: {python_path}")
    
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Usar todos os cores disponíveis
        .config("spark.driver.memory", "6g")  # Reduzido para evitar problemas
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "100")  # Reduzido para dataset menor
        .config("spark.default.parallelism", "4")
        .config("spark.driver.maxResultSize", "3g")
        .config("spark.python.worker.timeout", "600")  # 10 minutos timeout
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")
    
    logger.success(f"Spark Session criada: {spark.version}")
    logger.info(f"  Master: {spark.sparkContext.master}")
    logger.info(f"  App Name: {spark.sparkContext.appName}")
    logger.info(f"  Python: {python_path}")
    
    return spark


# ============================================================================
# LEITURA E ENRIQUECIMENTO COM PYSPARK
# ============================================================================

def load_and_join_spark(
    spark: "SparkSession",
    accounts_path: Path,
    trans_path: Path
) -> "DataFrame":
    """
    Carrega e enriquece dados usando PySpark (equivalente ao Pandas do dataset.py).
    
    Args:
        spark: Sessão Spark
        accounts_path: Caminho para HI-Large_accounts.csv
        trans_path: Caminho para HI-Large_Trans.csv
        
    Returns:
        DataFrame Spark enriquecido
    """
    logger.info(f"Carregando dados com PySpark...")
    logger.info(f"  Contas:      {accounts_path}")
    logger.info(f"  Transações:  {trans_path}")
    
    # 1. Carregar CSVs
    accounts_df = spark.read.csv(
        str(accounts_path),
        header=True,
        inferSchema=True
    )
    
    trans_df = spark.read.csv(
        str(trans_path),
        header=True,
        inferSchema=True
    )
    
    logger.info(f"  Shape Contas:      {accounts_df.count()} x {len(accounts_df.columns)}")
    logger.info(f"  Shape Transações:  {trans_df.count()} x {len(trans_df.columns)}")
    
    # 2. Renomear colunas de transações (mesmo esquema do Pandas)
    trans_df = trans_df.toDF(
        'Timestamp', 'From Bank', 'From Account', 'To Bank', 'To Account',
        'Amount Received', 'Receiving Currency', 'Amount Paid',
        'Payment Currency', 'Payment Format', 'Is Laundering'
    )
    
    # 3. Join com contas de ORIGEM (FROM)
    trans_enriched = trans_df.join(
        accounts_df,
        (trans_df['From Bank'] == accounts_df['Bank ID']) &
        (trans_df['From Account'] == accounts_df['Account Number']),
        how='left'
    )
    
    # Renomear colunas FROM
    trans_enriched = trans_enriched \
        .withColumnRenamed('Bank Name', 'From Bank Name') \
        .withColumnRenamed('Entity ID', 'From Entity ID') \
        .withColumnRenamed('Entity Name', 'From Entity Name')
    
    # Dropar colunas duplicadas do join
    trans_enriched = trans_enriched.drop('Bank ID', 'Account Number')
    
    # 4. Join com contas de DESTINO (TO)
    trans_enriched = trans_enriched.join(
        accounts_df,
        (trans_enriched['To Bank'] == accounts_df['Bank ID']) &
        (trans_enriched['To Account'] == accounts_df['Account Number']),
        how='left'
    )
    
    # Renomear colunas TO
    trans_enriched = trans_enriched \
        .withColumnRenamed('Bank Name', 'To Bank Name') \
        .withColumnRenamed('Entity ID', 'To Entity ID') \
        .withColumnRenamed('Entity Name', 'To Entity Name')
    
    # Dropar colunas duplicadas do segundo join
    trans_enriched = trans_enriched.drop('Bank ID', 'Account Number')
    
    logger.success(f"Dados enriquecidos! Shape: {trans_enriched.count()} x {len(trans_enriched.columns)}")
    
    return trans_enriched


# ============================================================================
# ESTATÍSTICAS POPULACIONAIS
# ============================================================================

def compute_population_stats(df_spark: "DataFrame") -> Dict:
    """
    Calcula estatísticas descritivas da população completa.
    
    Args:
        df_spark: DataFrame Spark da população
        
    Returns:
        Dicionário com estatísticas (médias, proporções, contagens)
    """
    logger.info("Calculando estatísticas da POPULAÇÃO...")
    
    stats_dict = {
        'total_count': df_spark.count(),
        'numeric_stats': {},
        'categorical_stats': {},
        'target_distribution': {}
    }
    
    # 1. Estatísticas NUMÉRICAS (médias)
    for col in NUMERIC_COLS_TO_VALIDATE:
        if col in df_spark.columns:
            stats_row = df_spark.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std'),
                F.min(col).alias('min'),
                F.max(col).alias('max')
            ).first()
            
            stats_dict['numeric_stats'][col] = {
                'mean': stats_row['mean'],
                'std': stats_row['std'],
                'min': stats_row['min'],
                'max': stats_row['max']
            }
            
            logger.info(f"  {col}: mean={stats_row['mean']:.2f}, std={stats_row['std']:.2f}")
    
    # 2. Estatísticas CATEGÓRICAS (distribuições)
    for col in CATEGORICAL_COLS_TO_VALIDATE:
        if col in df_spark.columns:
            dist = df_spark.groupBy(col).count().orderBy(F.desc('count')).collect()
            
            # Converter para proporções
            total = stats_dict['total_count']
            stats_dict['categorical_stats'][col] = {
                row[col]: row['count'] / total 
                for row in dist
            }
            
            logger.info(f"  {col}: {len(dist)} categorias únicas")
    
    # 3. Distribuição do TARGET (Is Laundering)
    target_dist = df_spark.groupBy('Is Laundering').count().collect()
    for row in target_dist:
        stats_dict['target_distribution'][row['Is Laundering']] = row['count']
    
    logger.success("Estatísticas populacionais calculadas!")
    logger.info(f"  Total de registros: {stats_dict['total_count']:,}")
    logger.info(f"  Distribuição Target: {stats_dict['target_distribution']}")
    
    return stats_dict


# ============================================================================
# AMOSTRAGEM ESTRATIFICADA
# ============================================================================

def create_stratified_sample(
    df_spark: "DataFrame",
    fraction: float,
    seed: int
) -> "DataFrame":
    """
    Cria amostra estratificada pelo target 'Is Laundering'.
    
    Args:
        df_spark: DataFrame Spark da população
        fraction: Fração da amostra (0.1 = 10%)
        seed: Seed para reprodutibilidade
        
    Returns:
        DataFrame Spark da amostra
    """
    logger.info(f"Criando amostra estratificada ({fraction:.1%})...")
    
    # Estratificar por 'Is Laundering' usando SQL puro do Spark
    # Coletar valores distintos do target primeiro
    target_values = [row['Is Laundering'] for row in df_spark.select('Is Laundering').distinct().collect()]
    
    logger.info(f"  Valores do target: {target_values}")
    
    # Criar frações para cada valor
    fractions_dict = {value: fraction for value in target_values}
    
    # Usar sampleBy que é mais eficiente
    sample_df = df_spark.sampleBy('Is Laundering', fractions=fractions_dict, seed=seed)
    
    sample_count = sample_df.count()
    logger.info(f"  Amostra gerada: {sample_count:,} registros")
    
    return sample_df


# ============================================================================
# VALIDAÇÃO ESTATÍSTICA
# ============================================================================

def validate_sample_statistics(
    population_stats: Dict,
    sample_df_pandas: pd.DataFrame,
    alpha: float = P_VALUE_THRESHOLD
) -> Tuple[bool, Dict]:
    """
    Valida se a amostra é estatisticamente representativa da população.
    
    Aplica:
    - Z-test para variáveis numéricas (médias)
    - Chi-quadrado para variáveis categóricas (proporções)
    
    Args:
        population_stats: Estatísticas da população
        sample_df_pandas: DataFrame Pandas da amostra
        alpha: Nível de significância (0.05 = 5%)
        
    Returns:
        Tuple (is_valid, test_results)
    """
    logger.info("Validando representatividade estatística da amostra...")
    
    test_results = {
        'numeric_tests': {},
        'categorical_tests': {},
        'target_test': {},
        'is_valid': True
    }
    
    n_sample = len(sample_df_pandas)
    
    # 1. TESTES Z para variáveis NUMÉRICAS
    for col, pop_stats in population_stats['numeric_stats'].items():
        if col not in sample_df_pandas.columns:
            continue
        
        sample_mean = sample_df_pandas[col].mean()
        sample_std = sample_df_pandas[col].std()
        
        pop_mean = pop_stats['mean']
        pop_std = pop_stats['std']
        
        # Z-score: (sample_mean - pop_mean) / (pop_std / sqrt(n))
        if pop_std > 0:
            se = pop_std / np.sqrt(n_sample)
            z_score = (sample_mean - pop_mean) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
            
            test_results['numeric_tests'][col] = {
                'pop_mean': pop_mean,
                'sample_mean': sample_mean,
                'z_score': z_score,
                'p_value': p_value,
                'passed': p_value > alpha
            }
            
            status = "✓ PASS" if p_value > alpha else "✗ FAIL"
            logger.info(f"  {col:25s} | Z={z_score:7.3f} | p={p_value:.4f} | {status}")
            
            if p_value <= alpha:
                test_results['is_valid'] = False
    
    # 2. TESTES CHI-QUADRADO para variáveis CATEGÓRICAS
    for col, pop_dist in population_stats['categorical_stats'].items():
        if col not in sample_df_pandas.columns:
            continue
        
        # Distribuição observada na amostra
        sample_counts = sample_df_pandas[col].value_counts().to_dict()
        
        # Frequências esperadas (proporções da população * n_sample)
        expected_counts = {
            category: prop * n_sample 
            for category, prop in pop_dist.items()
        }
        
        # Alinhar categorias (algumas podem estar ausentes na amostra)
        categories = list(pop_dist.keys())
        observed = [sample_counts.get(cat, 0) for cat in categories]
        expected = [expected_counts[cat] for cat in categories]
        
        # Chi-quadrado test
        if sum(expected) > 0:
            chi2_stat, p_value = stats.chisquare(observed, expected)
            
            test_results['categorical_tests'][col] = {
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'passed': p_value > alpha
            }
            
            status = "✓ PASS" if p_value > alpha else "✗ FAIL"
            logger.info(f"  {col:25s} | χ²={chi2_stat:7.3f} | p={p_value:.4f} | {status}")
            
            if p_value <= alpha:
                test_results['is_valid'] = False
    
    # 3. TESTE para distribuição do TARGET
    target_col = 'Is Laundering'
    if target_col in sample_df_pandas.columns:
        pop_target = population_stats['target_distribution']
        sample_target = sample_df_pandas[target_col].value_counts().to_dict()
        
        categories = list(pop_target.keys())
        observed = [sample_target.get(cat, 0) for cat in categories]
        expected = [pop_target[cat] * n_sample / population_stats['total_count'] for cat in categories]
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        test_results['target_test'] = {
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'passed': p_value > alpha
        }
        
        status = "✓ PASS" if p_value > alpha else "✗ FAIL"
        logger.info(f"  {'TARGET (Is Laundering)':25s} | χ²={chi2_stat:7.3f} | p={p_value:.4f} | {status}")
        
        if p_value <= alpha:
            test_results['is_valid'] = False
    
    # Resultado final
    if test_results['is_valid']:
        logger.success("✓ Amostra VÁLIDA! Todos os testes passaram (p > 0.05)")
    else:
        logger.warning("✗ Amostra INVÁLIDA! Alguns testes falharam (p ≤ 0.05)")
    
    return test_results['is_valid'], test_results


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def generate_validated_sample(
    spark: "SparkSession",
    accounts_path: Path,
    trans_path: Path,
    output_path: Path,
    sample_fraction: float = SAMPLE_FRACTION,
    max_iterations: int = MAX_ITERATIONS
) -> bool:
    """
    Pipeline completo: Gera amostra validada estatisticamente.
    
    Args:
        spark: Sessão Spark
        accounts_path: Caminho para contas
        trans_path: Caminho para transações
        output_path: Caminho para salvar amostra
        sample_fraction: Fração da amostra
        max_iterations: Máximo de tentativas
        
    Returns:
        True se amostra válida foi criada, False caso contrário
    """
    logger.info("="*80)
    logger.info("PIPELINE DE AMOSTRAGEM VALIDADA - PYSPARK")
    logger.info("="*80)
    
    # 1. Carregar e enriquecer dados
    df_spark = load_and_join_spark(spark, accounts_path, trans_path)
    
    # 2. Calcular estatísticas populacionais
    population_stats = compute_population_stats(df_spark)
    
    # 3. Loop de geração + validação
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"ITERAÇÃO {iteration}/{max_iterations}")
        logger.info(f"{'='*80}")
        
        # Criar amostra estratificada
        seed = RANDOM_SEED + iteration  # Variar seed a cada tentativa
        sample_spark = create_stratified_sample(df_spark, sample_fraction, seed)
        
        # EVITAR OutOfMemoryError: Salvar direto para CSV com Spark
        logger.info("Salvando amostra para CSV (evitar OOM no toPandas)...")
        temp_output = str(output_path).replace('.csv', '_temp')
        sample_spark.coalesce(1).write.mode('overwrite').option('header', 'true').csv(temp_output)
        
        # Carregar de volta com Pandas para validação (agora é seguro)
        logger.info("Carregando amostra com Pandas para validação...")
        import glob
        csv_files = glob.glob(f"{temp_output}/*.csv")
        if not csv_files:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {temp_output}")
        sample_pandas = pd.read_csv(csv_files[0])
        logger.info(f"Amostra convertida para Pandas: {sample_pandas.shape}")
        
        # Validar estatisticamente
        is_valid, test_results = validate_sample_statistics(population_stats, sample_pandas)
        
        if is_valid:
            logger.success(f"\n{'='*80}")
            logger.success(f"AMOSTRA VÁLIDA ENCONTRADA NA ITERAÇÃO {iteration}!")
            logger.success(f"{'='*80}")
            
            # Mover arquivo temporário para destino final
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Movendo amostra para {output_path.name}...")
            import shutil
            shutil.move(csv_files[0], output_path)
            
            # Limpar diretório temporário
            if Path(temp_output).exists():
                shutil.rmtree(temp_output)
                logger.info("Diretório temporário removido.")
            
            logger.success(f"Amostra salva em: {output_path}")
            logger.info(f"  Shape: {sample_pandas.shape}")
            logger.info(f"  Tamanho: {output_path.stat().st_size / (1024**2):.2f} MB")
            
            return True
        
        else:
            logger.warning(f"Amostra da iteração {iteration} não passou nos testes. Tentando novamente...")
    
    # Se chegou aqui, não encontrou amostra válida
    logger.error(f"\n{'='*80}")
    logger.error(f"FALHA: Nenhuma amostra válida encontrada em {max_iterations} iterações.")
    logger.error(f"{'='*80}")
    logger.error(f"Sugestões:")
    logger.error(f"  1. Aumentar SAMPLE_FRACTION (atualmente {sample_fraction:.1%})")
    logger.error(f"  2. Aumentar MAX_ITERATIONS (atualmente {max_iterations})")
    logger.error(f"  3. Relaxar P_VALUE_THRESHOLD (atualmente {P_VALUE_THRESHOLD})")
    
    return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal do script."""
    logger.info("="*80)
    logger.info("ANTI-MONEY LAUNDERING - AMOSTRAGEM SPARK")
    logger.info("="*80)
    logger.info(f"Projeto Root: {PROJ_ROOT}")
    
    # Caminhos
    accounts_path = EXTERNAL_DATA_DIR / 'HI-Large_accounts.csv'
    trans_path = EXTERNAL_DATA_DIR / 'HI-Large_Trans.csv'
    output_path = INTERIM_DATA_DIR / 'HI-Large_sampled.csv'
    
    # Validar existência dos arquivos
    if not accounts_path.exists():
        logger.error(f"Arquivo não encontrado: {accounts_path}")
        return
    
    if not trans_path.exists():
        logger.error(f"Arquivo não encontrado: {trans_path}")
        return
    
    # Criar sessão Spark
    spark = create_spark_session()
    
    try:
        # Executar pipeline
        success = generate_validated_sample(
            spark=spark,
            accounts_path=accounts_path,
            trans_path=trans_path,
            output_path=output_path,
            sample_fraction=SAMPLE_FRACTION,
            max_iterations=MAX_ITERATIONS
        )
        
        if success:
            logger.success("\n" + "="*80)
            logger.success("PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.success("="*80)
            logger.info("\nPróximos passos:")
            logger.info("  1. Execute o pipeline Pandas normalmente:")
            logger.info("     python source/dataset.py")
            logger.info("  2. O dataset.py detectará automaticamente a amostra")
            logger.info("     e processará apenas os dados reduzidos!")
        else:
            logger.error("\nPipeline falhou. Ajuste os parâmetros e tente novamente.")
    
    finally:
        # Encerrar Spark
        spark.stop()
        logger.info("Spark Session encerrada.")


if __name__ == '__main__':
    main()
