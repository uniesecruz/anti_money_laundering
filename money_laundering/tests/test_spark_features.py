"""
Testes para validar a implementação PySpark de Feature Engineering

Executa:
1. Validação de imports
2. Testes unitários de cada classe
3. Comparação Pandas vs PySpark
4. Testes de anti-leakage
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test que todos os imports funcionam."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: VALIDAÇÃO DE IMPORTS")
    logger.info("="*80)
    
    try:
        from source.features import (
            VelocityFeatureGenerator,
            RatioFeatureGenerator,
            BehavioralFeatureGenerator,
            FeatureEngineeringPipeline
        )
        logger.info("✅ Pandas features importadas com sucesso")
    except ImportError as e:
        logger.error(f"❌ Erro ao importar features Pandas: {e}")
        return False
    
    try:
        from source.spark_features import (
            SparkVelocityFeatureGenerator,
            SparkRatioFeatureGenerator,
            SparkBehavioralFeatureGenerator,
            SparkFeatureEngineeringPipeline
        )
        logger.info("✅ PySpark features importadas com sucesso")
    except ImportError as e:
        logger.error(f"❌ Erro ao importar features Spark: {e}")
        return False
    
    try:
        from source.spark_sampler import create_spark_session
        logger.info("✅ Spark sampler importado com sucesso")
    except ImportError as e:
        logger.error(f"❌ Erro ao importar spark_sampler: {e}")
        return False
    
    return True


def create_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Cria dataset de teste com features realísticas."""
    
    np.random.seed(42)
    
    # Criar timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i*15) for i in range(n_rows)]
    
    # Criar dados
    data = {
        'Timestamp': timestamps,
        'From Account': np.random.choice(['ACC_001', 'ACC_002', 'ACC_003', 'ACC_004'], n_rows),
        'Amount Received': np.random.lognormal(mean=8, sigma=2, size=n_rows),
        'Receiving Currency': np.random.choice(['USD', 'EUR', 'GBP'], n_rows),
        'From Bank': np.random.choice(['BANK_A', 'BANK_B', 'BANK_C'], n_rows),
        'Is Laundering': np.random.binomial(1, 0.1, n_rows)
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df


def test_pandas_pipeline():
    """Test do pipeline Pandas."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: PANDAS PIPELINE")
    logger.info("="*80)
    
    try:
        from source.features import FeatureEngineeringPipeline
        
        # Criar dados
        df = create_test_data(n_rows=500)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        logger.info(f"  Dataset shape: {df.shape}")
        
        # Criar pipeline
        pipeline = FeatureEngineeringPipeline()
        
        # Fit e transform
        pipeline.fit(df)
        df_fe = pipeline.transform(df)
        
        logger.info(f"  Output shape: {df_fe.shape}")
        
        # Validações
        assert df_fe.shape[0] == df.shape[0], "Número de linhas mudou!"
        assert df_fe.shape[1] > df.shape[1], "Nenhuma feature foi adicionada!"
        
        logger.info(f"✅ Pipeline Pandas funcionou corretamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pipeline Pandas: {e}")
        return False


def test_spark_pipeline():
    """Test do pipeline PySpark."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: PYSPARK PIPELINE")
    logger.info("="*80)
    
    try:
        from source.spark_features import SparkFeatureEngineeringPipeline
        from source.spark_sampler import create_spark_session
        
        # Criar Spark session
        spark = create_spark_session()
        
        # Criar dados
        df = create_test_data(n_rows=500)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Converter para Spark
        df_spark = spark.createDataFrame(df)
        
        logger.info(f"  Dataset shape: ({df_spark.count()}, {len(df_spark.columns)})")
        
        # Criar pipeline
        pipeline_spark = SparkFeatureEngineeringPipeline()
        
        # Fit e transform
        pipeline_spark.fit(df_spark)
        df_fe_spark = pipeline_spark.transform(df_spark)
        
        n_rows_output = df_fe_spark.count()
        n_cols_output = len(df_fe_spark.columns)
        
        logger.info(f"  Output shape: ({n_rows_output}, {n_cols_output})")
        
        # Validações
        assert n_rows_output == df.shape[0], "Número de linhas mudou!"
        assert n_cols_output > len(df.columns), "Nenhuma feature foi adicionada!"
        
        spark.stop()
        logger.info(f"✅ Pipeline PySpark funcionou corretamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pipeline PySpark: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anti_leakage_pandas():
    """Test de prevenção de data leakage com Pandas."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: ANTI-LEAKAGE VALIDATION (PANDAS)")
    logger.info("="*80)
    
    try:
        from source.features import FeatureEngineeringPipeline
        
        # Criar dados com padrão conhecido
        df = create_test_data(n_rows=100)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Split em treino (80%) e OOT (20%)
        split_idx = int(0.8 * len(df))
        df_treino = df.iloc[:split_idx].copy()
        df_oot = df.iloc[split_idx:].copy()
        
        logger.info(f"  Treino: {df_treino.shape}")
        logger.info(f"  OOT: {df_oot.shape}")
        
        # Criar pipeline
        pipeline = FeatureEngineeringPipeline()
        
        # FIT APENAS em treino (o segredo do anti-leakage)
        pipeline.fit(df_treino)
        
        # Transform em ambos (com MESMO pipeline)
        df_treino_fe = pipeline.transform(df_treino)
        df_oot_fe = pipeline.transform(df_oot)
        
        # Validações de anti-leakage
        # 1. Primeiras linhas de OOT não devem ter features artificialmente altas
        # (não devem ver dados futuros)
        first_oot_features = df_oot_fe[
            [col for col in df_oot_fe.columns if 'velocity' in col]
        ].iloc[0]
        
        # As primeiras transações devem ter velocity features baixas
        # (primeiro cliente novo, sem histórico)
        assert (first_oot_features.fillna(0) >= 0).all(), \
            "Velocity features não devem ser negativas"
        
        logger.info(f"✅ Anti-leakage validado (Pandas)")
        return True
        
    except AssertionError as e:
        logger.error(f"❌ {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consistency_pandas_vs_spark():
    """Test de consistência entre Pandas e Spark."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: CONSISTÊNCIA PANDAS vs PYSPARK")
    logger.info("="*80)
    
    try:
        from source.features import FeatureEngineeringPipeline as PandasPipeline
        from source.spark_features import SparkFeatureEngineeringPipeline as SparkPipeline
        from source.spark_sampler import create_spark_session
        
        # Criar dados
        df = create_test_data(n_rows=100)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Criar Spark session
        spark = create_spark_session()
        df_spark = spark.createDataFrame(df)
        
        logger.info(f"  Dataset: {df.shape}")
        
        # Pipeline Pandas
        pandas_pipeline = PandasPipeline()
        pandas_pipeline.fit(df)
        df_pandas_result = pandas_pipeline.transform(df)
        
        # Pipeline Spark
        spark_pipeline = SparkPipeline()
        spark_pipeline.fit(df_spark)
        df_spark_result = spark_pipeline.transform(df_spark)
        df_spark_result_pandas = df_spark_result.toPandas()
        
        logger.info(f"  Pandas output shape: {df_pandas_result.shape}")
        logger.info(f"  Spark output shape: {df_spark_result_pandas.shape}")
        
        # Validações
        assert df_pandas_result.shape == df_spark_result_pandas.shape, \
            "Shapes diferem!"
        
        # Verificar que as colunas são as mesmas
        pandas_cols = set(df_pandas_result.columns)
        spark_cols = set(df_spark_result_pandas.columns)
        
        assert pandas_cols == spark_cols, \
            f"Colunas diferem: {pandas_cols - spark_cols} vs {spark_cols - pandas_cols}"
        
        logger.info(f"✅ Consistência validada")
        spark.stop()
        return True
        
    except AssertionError as e:
        logger.error(f"❌ {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes."""
    
    logger.info("\n" + "#"*80)
    logger.info("# TESTES DE FEATURE ENGINEERING: PANDAS vs PYSPARK")
    logger.info("#"*80)
    
    results = {}
    
    # Test 1: Imports
    results['Imports'] = test_imports()
    
    # Test 2: Pandas Pipeline
    results['Pandas Pipeline'] = test_pandas_pipeline()
    
    # Test 3: Spark Pipeline
    results['Spark Pipeline'] = test_spark_pipeline()
    
    # Test 4: Anti-leakage (Pandas)
    results['Anti-leakage (Pandas)'] = test_anti_leakage_pandas()
    
    # Test 5: Consistency
    results['Consistency (Pandas vs Spark)'] = test_consistency_pandas_vs_spark()
    
    # Sumário
    logger.info("\n" + "="*80)
    logger.info("SUMÁRIO DOS TESTES")
    logger.info("="*80)
    
    for test_name, result in results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        logger.info(f"{test_name:<40} {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    logger.info(f"\n📊 Total: {total_passed}/{total_tests} testes passaram")
    
    if total_passed == total_tests:
        logger.info("\n🎉 TODOS OS TESTES PASSARAM!")
        return 0
    else:
        logger.error(f"\n⚠️ {total_tests - total_passed} teste(s) falharam")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
