# 🚀 Quick Reference: PySpark + Pandas Feature Engineering

## 📋 Checklist Rápido

- [x] PySpark 4.0.0 instalado
- [x] Lazy loading configurado em `source/spark_features.py`
- [x] Type hints corrigidos (strings para Python 3.11+)
- [x] Diagnóstico criado (`diagnose_pyspark.py`)
- [x] Notebook atualizado com guide
- [x] ✅ **Problema RESOLVIDO**

---

## 🎯 Comparativo Rápido

| Feature | Pandas | PySpark |
|---------|--------|---------|
| **Dataset até 1GB** | ✅ Ideal | ⚡ OK |
| **Dataset 1-10GB** | ⚠️ Lento | ✅ Ideal |
| **Dataset 10GB+** | ❌ Impossível | ✅ Ideal |
| **Setup Tempo** | 0.1s | 3-5s |
| **Debug** | Fácil | Moderado |
| **Produção Local** | Sim | Não (precisa cluster) |

---

## 🔥 Troubleshooting 10-Segundo

### Erro: `AttributeError: UnixStreamServer`
**Solução**: Já foi corrigido! Use lazy loading.
```bash
# Validate
python -c "from source.spark_features import SparkFeatureEngineeringPipeline; print('✅')"
```

### Erro: `ModuleNotFoundError: pyspark`
**Solução**: Instale ou upgrade
```bash
pip install pyspark==4.0.0
# ou versão anterior mais estável
pip install pyspark==3.5.1
```

### Erro: `Java not found`
**Solução**: 
```bash
# Verificar
java -version

# Se não tiver, instale JDK:
# Windows: Instale de https://www.oracle.com/java/technologies/downloads/
# então configure JAVA_HOME
```

### Performance muito lenta
**Solução**: Para datasets < 2GB, use Pandas
```python
# em vez de PySpark
from source.features import FeatureEngineeringPipeline
```

---

## 💻 Quick Start (Copy-Paste)

### Versão Pandas
```python
from source.features import FeatureEngineeringPipeline
import pandas as pd

# Carregar dados
df_treino = pd.read_csv('df_treino.csv')
df_oot = pd.read_csv('df_oot.csv')

# Feature Engineering
pipeline = FeatureEngineeringPipeline()
df_treino_fe = pipeline.fit(df_treino).transform(df_treino)
df_oot_fe = pipeline.transform(df_oot)

print(f"✅ Features: {df_treino_fe.shape}")
```

### Versão PySpark
```python
from source.spark_features import SparkFeatureEngineeringPipeline
from source.spark_sampler import create_spark_session

# Criar Spark Session (automático com lazy loading)
spark = create_spark_session()

# Carregar dados
df_treino_spark = spark.read.csv('df_treino.csv', header=True, inferSchema=True)
df_oot_spark = spark.read.csv('df_oot.csv', header=True, inferSchema=True)

# Feature Engineering (mesmo código!)
pipeline = SparkFeatureEngineeringPipeline()
df_treino_fe = pipeline.fit(df_treino_spark).transform(df_treino_spark)
df_oot_fe = pipeline.transform(df_oot_spark)

print(f"✅ Features: {df_treino_fe.count()} x {len(df_treino_fe.columns)}")
```

---

## 📊 Features 30+ Geradas

Ambas as versões geram:
- **Velocity** (janelas 1h, 24h, 7d): txn_count, amount_sum, amount_avg, amount_max, amount_std
- **Ratio** (30d histórico): amount_to_historical_mean, amount_to_historical_max, amount_zscore
- **Behavioral**: time_since_last_txn, bank_change, is_new_country, is_unusual_hour
- **Smurfing**: smurf_txn_count_24h, smurf_txn_amount_sum, smurf_proximity_score

⚠️ **ZERO Data Leakage Garantido**: Fit apenas em treino, transform em ambos

---

## 🧪 Testar Agora

Notebook célula: "TESTE RÁPIDO DE IMPORTS"
```python
# Role para baixo no notebook até a célula 
# "TESTE RÁPIDO DE IMPORTS" (abaixo de ALTERNATIVA: PySpark)
# e execute. Deve mostrar ✅ para ambos os imports
```

---

## 📚 Documentos de Referência

| Arquivo | Conteúdo |
|---------|----------|
| `SPARK_FEATURES_GUIDE.md` | Documentação completa técnica |
| `SPARK_USAGE_EXAMPLES.md` | Exemplos lado a lado Pandas/Spark |
| `SPARK_IMPLEMENTATION_SUMMARY.md` | Sumário da implementação |
| `PYSPARK_FIX.md` | Detalhes técnicos sobre lazy loading fix |
| `diagnose_pyspark.py` | Script de diagnóstico |
| `tests/test_spark_features.py` | Testes unittest |

---

## 🔗 Próximos Passos

1. **Executar Notebook 05**
   - Use Pandas se dataset < 2GB (mais simples)
   - Ou PySpark se > 2GB (mais rápido)

2. **Notebook 06**: Transformações
   - Normalização (Yeo-Johnson)
   - Target Encoding

3. **Notebook 08**: Modelagem
   - Treinar com features
   - Validar com OOT

---

**Version**: 1.0  
**Status**: ✅ READY  
**Last Updated**: Março 2026
