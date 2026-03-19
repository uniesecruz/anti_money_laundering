"""
Script de Validação Completa - Feature Engineering Pandas vs PySpark

Testa:
1. Imports
2. Feature generation
3. Output validation
4. Performance comparison
"""

import sys
import time
from pathlib import Path

print("\n" + "="*80)
print("VALIDAÇÃO COMPLETA - FEATURE ENGINEERING")
print("="*80)

# ===== TEST 1: IMPORTS =====
print("\n[TEST 1] Testando Imports...")
try:
    import pandas as pd
    import numpy as np
    print("  ✅ Pandas + NumPy")
except ImportError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

try:
    from source.config import get_data_path
    print("  ✅ source.config")
except ImportError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

try:
    from source.features import FeatureEngineeringPipeline
    print("  ✅ source.features (Pandas)")
except ImportError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

try:
    from source.spark_features import SparkFeatureEngineeringPipeline
    print("  ✅ source.spark_features (PySpark - com lazy loading)")
except ImportError as e:
    print(f"  ⚠️  PySpark error: {e}")
    print("     (Não fatal - Pandas ainda funciona)")

# ===== TEST 2: DATA LOADING =====
print("\n[TEST 2] Testando Carregamento de Dados...")
try:
    df_treino = pd.read_csv(get_data_path('df_treino.csv', 'processed'))
    df_oot = pd.read_csv(get_data_path('df_oot.csv', 'processed'))
    
    print(f"  ✅ Treino: {df_treino.shape}")
    print(f"  ✅ OOT: {df_oot.shape}")
    
    # Convert timestamp
    df_treino['Timestamp'] = pd.to_datetime(df_treino['Timestamp'])
    df_oot['Timestamp'] = pd.to_datetime(df_oot['Timestamp'])
    print("  ✅ Timestamps convertidos")
    
except Exception as e:
    print(f"  ❌ Erro ao carregar dados: {e}")
    print("     Certifique-se que df_treino.csv e df_oot.csv existem em processed/")
    sys.exit(1)

# ===== TEST 3: PANDAS PIPELINE =====
print("\n[TEST 3] Testando Feature Engineering (Pandas)...")
try:
    start = time.time()
    
    pipeline = FeatureEngineeringPipeline()
    pipeline.fit(df_treino)
    df_treino_fe = pipeline.transform(df_treino)
    df_oot_fe = pipeline.transform(df_oot)
    
    pandas_time = time.time() - start
    
    print(f"  ✅ Pipeline executado em {pandas_time:.2f}s")
    print(f"  ✅ Treino: {df_treino.shape} → {df_treino_fe.shape}")
    print(f"  ✅ OOT: {df_oot.shape} → {df_oot_fe.shape}")
    
    # Validar features
    original_cols = set(df_treino.columns)
    new_features = [col for col in df_treino_fe.columns if col not in original_cols]
    print(f"  ✅ Features geradas: {len(new_features)}")
    
except Exception as e:
    print(f"  ❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== TEST 4: PYSPARK PIPELINE (Optional) =====
print("\n[TEST 4] Testando Feature Engineering (PySpark)...")
try:
    from source.spark_sampler import create_spark_session
    from pyspark.sql import functions as F
    
    start = time.time()
    
    spark = create_spark_session()
    df_treino_spark = spark.createDataFrame(df_treino)
    df_oot_spark = spark.createDataFrame(df_oot)
    
    pipeline_spark = SparkFeatureEngineeringPipeline()
    pipeline_spark.fit(df_treino_spark)
    df_treino_fe_spark = pipeline_spark.transform(df_treino_spark)
    df_oot_fe_spark = pipeline_spark.transform(df_oot_spark)
    
    spark_time = time.time() - start
    
    print(f"  ✅ Pipeline Spark executado em {spark_time:.2f}s")
    print(f"  ✅ Treino: {df_treino_spark.count()} x {len(df_treino_spark.columns)} → {df_treino_fe_spark.count()} x {len(df_treino_fe_spark.columns)}")
    print(f"  ✅ OOT: {df_oot_spark.count()} x {len(df_oot_spark.columns)} → {df_oot_fe_spark.count()} x {len(df_oot_fe_spark.columns)}")
    
    spark.stop()
    
except Exception as e:
    print(f"  ⚠️  Spark não disponível: {e}")
    print("     (Não fatal - use Pandas para datasets < 2GB)")
    spark_time = None

# ===== SUMMARY =====
print("\n" + "="*80)
print("RESUMO DA VALIDAÇÃO")
print("="*80)

print(f"\n✅ Imports: SUCESSO")
print(f"✅ Dados: CARREGADOS ({df_treino.shape[0]:,} linhas treino)")
print(f"✅ Pandas Pipeline: FUNCIONA ({pandas_time:.2f}s)")

if spark_time:
    print(f"✅ Spark Pipeline: FUNCIONA ({spark_time:.2f}s)")
    if pandas_time < spark_time:
        print(f"   💡 Pandas é {spark_time/pandas_time:.1f}x mais rápido para este dataset")
        print(f"      (Use PySpark apenas para datasets > 2GB)")
    else:
        print(f"   💡 Spark é {pandas_time/spark_time:.1f}x mais rápido")
else:
    print(f"⚠️  Spark Pipeline: NÃO DISPONÍVEL")
    print(f"   💡 Use Pandas para datasets < 2GB")

print(f"\n📊 Features Geradas:")
for feature_type in ['velocity', 'ratio', 'behavioral', 'smurf']:
    count = len([f for f in new_features if feature_type in f])
    if count > 0:
        print(f"   • {feature_type.capitalize()}: {count}")

print(f"\n" + "="*80)
print("🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*80)

print(f"\n💡 Próximos Passos:")
print(f"   1. Abra o Notebook 05 (05_feature_engineering.ipynb)")
print(f"   2. Execute a célula 'TESTE RÁPIDO DE IMPORTS'")
print(f"   3. Execute a célula do seu pipeline preferido (Pandas ou Spark)")
print(f"   4. Vá para Notebook 06 para transformações e encoding")

print(f"\n📖 Documentação:")
print(f"   • SOLUTION_SUMMARY.md - Especificação da solução")
print(f"   • SPARK_FEATURES_GUIDE.md - Documentação técnica")
print(f"   • QUICK_REFERENCE.md - Troubleshooting rápido")
