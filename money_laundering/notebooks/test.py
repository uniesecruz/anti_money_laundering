# Importar bibliotecas necessárias
from pyspark.sql import SparkSession

# Criar sessão Spark no Colab
spark = SparkSession.builder \
    .appName("Anti Money Laundering - Colab") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")

# Atualizar caminhos para o diretório correto no Google Drive
path_data_raw = r'/content/drive/MyDrive/USP_ESALQ/TCC/money_laundering/data/raw'
path_accounts_df = r"/content/drive/MyDrive/USP_ESALQ/TCC/money_laundering/data/external/LI-Large_accounts.csv"
path_trans_df = r"/content/drive/MyDrive/USP_ESALQ/TCC/money_laundering/data/external/LI-Large_Trans.csv"

# Ler arquivos CSV usando Spark
print("Lendo arquivo de contas...")
accounts_spark_df = spark.read.csv(
    path_accounts_df,
    header=True,
    inferSchema=True
)

print("Lendo arquivo de transações...")
trans_spark_df = spark.read.csv(
    path_trans_df,
    header=True,
    inferSchema=True
)

# Mostrar informações dos DataFrames
print("\n=== Informações do DataFrame de Contas ===")
print(f"Total de registros: {accounts_spark_df.count()}")
print("\nSchema:")
accounts_spark_df.printSchema()
print("\nPrimeiras 5 linhas:")
accounts_spark_df.show(5)

print("\n=== Informações do DataFrame de Transações ===")
print(f"Total de registros: {trans_spark_df.count()}")
print("\nSchema:")
trans_spark_df.printSchema()
print("\nPrimeiras 5 linhas:")
trans_spark_df.show(5)

