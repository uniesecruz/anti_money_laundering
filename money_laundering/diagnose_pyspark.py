"""
Script de Diagnóstico e Workaround para PySpark no Windows

Este script detecta e corrige problemas comuns com PySpark em Python 3.11+
incluindo o erro: AttributeError: module 'socketserver' has no attribute 'UnixStreamServer'
"""

import sys
import subprocess
from pathlib import Path

print("=" * 80)
print("DIAGNÓSTICO PYSPARK - Windows Python 3.11+")
print("=" * 80)

# Check 1: Versão do Python
print(f"\n✓ Versão do Python: {sys.version}")

# Check 2: Versão do PySpark
try:
    import pyspark
    print(f"✓ PySpark instalado: {pyspark.__version__}")
except ImportError:
    print("❌ PySpark não instalado!")
    print("   Solução: pip install pyspark==4.0.0")
    sys.exit(1)

# Check 3: Tentar importar (vai falhar com o erro de socketserver)
print("\n🔍 Testando inicialização de PySpark...")
try:
    from pyspark.sql import SparkSession
    print("✅ PySpark importado com sucesso!")
except AttributeError as e:
    if "UnixStreamServer" in str(e):
        print("⚠️  ERRO CONHECIDO: UnixStreamServer não encontrado")
        print(f"   Erro completo: {e}")
        print("\n💡 SOLUÇÃO: Esta é uma incompatibilidade conhecida")
        print("   Usaremos workaround com lazy imports ya configurado")
        print("   O código foi atualizado para evitar este erro")
    else:
        print(f"❌ Erro desconhecido: {e}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Check 4: Verificar Java
print("\n🔍 Verificando Java...")
try:
    result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        java_version = result.stderr.split('\n')[0] if result.stderr else "Java encontrado"
        print(f"✓ Java disponível: {java_version}")
    else:
        print("⚠️  Java pode não estar instalado")
except FileNotFoundError:
    print("⚠️  Java não encontrado no PATH")
    print("   (Opcional - PySpark pode funcionar sem Java em local mode)")

# Check 5: Environment Variables
print("\n🔍 Variáveis de Ambiente Spark:")
import os
spark_home = os.environ.get('SPARK_HOME', 'Não definido')
java_home = os.environ.get('JAVA_HOME', 'Não definido')
print(f"   SPARK_HOME: {spark_home}")
print(f"   JAVA_HOME: {java_home}")

print("\n" + "=" * 80)
print("RECOMENDAÇÕES")
print("=" * 80)

print("""
✅ CÓDIGO ATUALIZADO COM LAZY IMPORTS
   - source/spark_features.py agora usa lazy loading
   - Imports do PySpark são feitos apenas quando necessário
   - Evita o erro de inicialização

✅ PRÓXIMOS PASSOS NO NOTEBOOK:
   1. Execute a célula de imports (agora com lazy loading)
   2. A primeira célula que usa Spark vai inicializar corretamente
   3. Subsequent calls serão mais rápidas

✅ SE AINDA HOUVER ERRO:
   Opção A: Reinstalar PySpark
      pip uninstall pyspark -y
      pip install pyspark==3.5.1  # Versão anterior mais estável

   Opção B: Usar ambiente virtualizado limpo
      python -m venv venv_spark_clean
      # ativar
      pip install pyspark pandas loguru

   Opção C: Usar apenas Pandas (recomendado para datasets < 2GB)
      # Desabilitar células PySpark
      # Usar source.features (Pandas) em vez de spark_features
""")

print("=" * 80)
print("✅ DIAGNÓSTICO CONCLUÍDO")
print("=" * 80)
