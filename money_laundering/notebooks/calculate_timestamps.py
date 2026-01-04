from pyspark.sql import SparkSession
from pyspark.sql.functions import min as spark_min, max as spark_max, col
from pathlib import Path
import os

# Criar sessão Spark
spark = SparkSession.builder \
    .appName("Timestamp Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")

# Caminho para a pasta external
path_external = Path(r"c:\Users\win\Desktop\TCC\money_laundering\data\external")

# Listar todos os arquivos CSV
csv_files = [f for f in path_external.glob("*.csv")]

print("=" * 80)
print("ANÁLISE DE TIMESTAMPS DOS ARQUIVOS CSV - PySpark")
print("=" * 80)

results = []
global_min = None
global_max = None
total_records = 0

for csv_file in csv_files:
    print(f"\nProcessando: {csv_file.name}")
    try:
        # Ler o arquivo CSV com Spark
        df = spark.read.csv(str(csv_file), header=True, inferSchema=True)
        
        # Contar registros
        count = df.count()
        total_records += count
        
        # Verificar se há coluna Timestamp
        timestamp_cols = [c for c in df.columns if 'timestamp' in c.lower()]
        
        if timestamp_cols:
            for col_name in timestamp_cols:
                # Calcular min e max
                stats = df.agg(
                    spark_min(col(col_name)).alias("min_ts"),
                    spark_max(col(col_name)).alias("max_ts")
                ).collect()[0]
                
                min_ts = stats['min_ts']
                max_ts = stats['max_ts']
                
                # Atualizar min e max global
                if global_min is None or (min_ts and min_ts < global_min):
                    global_min = min_ts
                if global_max is None or (max_ts and max_ts > global_max):
                    global_max = max_ts
                
                result = {
                    'Arquivo': csv_file.name,
                    'Coluna': col_name,
                    'Min Timestamp': str(min_ts),
                    'Max Timestamp': str(max_ts),
                    'Registros': count
                }
                results.append(result)
                
                print(f"  Coluna: {col_name}")
                print(f"  Min: {min_ts}")
                print(f"  Max: {max_ts}")
                print(f"  Total de registros: {count:,}")
        else:
            print(f"  ⚠️ Nenhuma coluna Timestamp encontrada")
            print(f"  Colunas disponíveis: {', '.join(df.columns)}")
    
    except Exception as e:
        print(f"  ❌ Erro ao processar: {e}")

# Mostrar resultados
if results:
    print("\n" + "=" * 80)
    print("RESUMO GERAL")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['Arquivo']} - {result['Coluna']}")
        print(f"  Min: {result['Min Timestamp']}")
        print(f"  Max: {result['Max Timestamp']}")
        print(f"  Registros: {result['Registros']:,}")
    
    # Min e Max global
    print("\n" + "=" * 80)
    print("MIN E MAX GLOBAL DE TODOS OS ARQUIVOS")
    print("=" * 80)
    print(f"Min Timestamp Global: {global_min}")
    print(f"Max Timestamp Global: {global_max}")
    print(f"Total de registros: {total_records:,}")
else:
    print("\n⚠️ Nenhum timestamp foi encontrado nos arquivos CSV")

# Encerrar sessão Spark
spark.stop()
