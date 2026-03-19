# 📚 Índice de Documentação - Feature Engineering

## 🎯 Problema & Solução

### Resumo Executivo
- **[README_MAIN.md](README_MAIN.md)** ⭐ START HERE
  - ✅ Status final
  - 🚀 Quick start (copy-paste)
  - 📋 Checklist

- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)**
  - 🔍 O que foi corrigido
  - ✅ Validação
  - 💡 Próximos passos

### Documentação Técnica

- **[PYSPARK_FIX.md](PYSPARK_FIX.md)**
  - 📝 Detalhes técnicos do fix
  - 🔧 Lazy loading pattern
  - ⚠️ Type hints em strings

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
  - ⚡ Quick troubleshooting
  - 💻 Copy-paste solutions
  - 🧪 Speedtest

---

## 🚀 Feature Engineering (Pandas & PySpark)

### Guias Principais

- **[SPARK_FEATURES_GUIDE.md](SPARK_FEATURES_GUIDE.md)** 
  - 📖 Documentação completa
  - 📊 Benchmarks
  - 🏗️ Arquitetura Window Functions
  - ⚠️ Garantias anti-leakage

- **[SPARK_USAGE_EXAMPLES.md](SPARK_USAGE_EXAMPLES.md)**
  - 💻 Exemplos lado a lado
  - Pandas vs PySpark
  - 🔄 Conversão Spark ↔ Pandas

- **[SPARK_IMPLEMENTATION_SUMMARY.md](SPARK_IMPLEMENTATION_SUMMARY.md)**
  - 📋 O que foi implementado
  - ✅ Testes realizados
  - 🎯 Performance

---

## 🧪 Scripts de Teste & Diagnóstico

### Executar Testes

```bash
# Diagnóstico rápido
python diagnose_pyspark.py

# Validação completa
python test_feature_engineering.py

# Unit tests
python tests/test_spark_features.py
```

### Descrição dos Scripts

| Script | Propósito | Tempo |
|--------|-----------|-------|
| `diagnose_pyspark.py` | Verificar Python, PySpark, Java | <1s |
| `test_feature_engineering.py` | Validar Pandas + Spark | ~20s |
| `tests/test_spark_features.py` | Unit tests anti-leakage | ~30s |

---

## 📁 Estrutura de Código

### Módulos Atualizados

```
source/
├── spark_features.py ⭐ MODIFICADO
│   ├── SparkVelocityFeatureGenerator
│   ├── SparkRatioFeatureGenerator
│   ├── SparkBehavioralFeatureGenerator
│   ├── SparkFeatureEngineeringPipeline
│   └── _init_pyspark() [NEW - lazy loading]
│
└── features.py [EXISTENTE]
    ├── VelocityFeatureGenerator
    ├── RatioFeatureGenerator
    ├── BehavioralFeatureGenerator
    └── FeatureEngineeringPipeline
```

### Notebooks Atualizados

```
notebooks/
└── 05_feature_engineering.ipynb ⭐ MODIFICADO
    ├── Célula: TESTE RÁPIDO DE IMPORTS [NEW]
    ├── Célula: Nota sobre PySpark [NEW]
    ├── Célula: Setup PySpark
    ├── Célula: Carregamento em Spark
    ├── Célula: Feature Engineering Spark
    ├── Célula: Análise Comparativa
    └── Célula: Guia de Uso
```

---

## 🔄 Workflow Recomendado

### 1. Diagnóstico (~1 min)
```bash
python diagnose_pyspark.py
# Verificar Python 3.11+, PySpark 4.0+, Java 17
```

### 2. Validação (~20 min)
```bash
python test_feature_engineering.py
# Validar imports, dados, Pandas, Spark
```

### 3. Abrir Notebook
```
notebooks/05_feature_engineering.ipynb
│
├─ Célula: "TESTE RÁPIDO DE IMPORTS" ✅ Run
├─ Célula: "SETUP PYSPARK" ✅ Run
├─ Célula: "FEATURE ENGINEERING PANDAS" ou "SPARK" ✅ Run
└─ Próximo: Notebook 06
```

### 4. Verificar Output
```
dados/processed/
├── df_treino_fe.csv (Pandas)
├── df_oot_fe.csv (Pandas)
├── df_treino_fe.parquet (Spark)
├── df_oot_fe.parquet (Spark)
└── ✅ Features salvos
```

---

## 🎯 Para Cada Caso de Uso

### "Quero usar Pandas (< 2GB)"
📖 Leia: `QUICK_REFERENCE.md` → "Pandas Quick Start"
💻 Para ver exemplos: `SPARK_USAGE_EXAMPLES.md` → "VERSÃO PANDAS"

### "Quero usar PySpark (> 2GB)"
📖 Leia: `SPARK_FEATURES_GUIDE.md` → "Como Usar"
💻 Para ver exemplos: `SPARK_USAGE_EXAMPLES.md` → "VERSÃO PYSPARK"

### "Tenho erro ao importar"
⚡ Leia: `QUICK_REFERENCE.md` → "Troubleshooting 10-Segundo"
🔧 Execute: `python diagnose_pyspark.py`

### "Quero entender a solução técnica"
🔬 Leia: `PYSPARK_FIX.md` → "A Solução Aplicada"
📊 Leia: `SOLUTION_SUMMARY.md` → "Solução Aplicada"

### "Quero benchmarks e performance"
📈 Leia: `SPARK_FEATURES_GUIDE.md` → "Performance Benchmarks"
📊 Leia: `SPARK_IMPLEMENTATION_SUMMARY.md` → "Performance"

---

## 📊 Referência Rápida

### Features Geradas (30+)

| Categoria | Features | Window | Leakage-Safe |
|-----------|----------|--------|--------------|
| **Velocity** | txn_count, amount_sum, etc | 1h,24h,7d | ✅ .shift()/rangeBetween |
| **Ratio** | amount_to_mean, amount_to_max | 30d | ✅ closed='left' |
| **Behavioral** | time_since_last, bank_change | lag/row_num | ✅ shift(1) |
| **Smurfing** | smurf_count, smurf_sum | 24h | ✅ rangeBetween |

### Performance

| Dataset | Pandas | Spark | Recomendação |
|---------|--------|-------|--------------|
| 100 MB | 2s | 5s | Pandas |
| 1 GB | 10s | 8s | Spark** |
| 10 GB | ❌ Falha | 30s | Spark ✅ |
| 100 GB | ❌ Impossível | 3min | Spark obrigatório |

---

## 🛠️ Troubleshooting Index

| Problema | Solução | Arquivo |
|----------|---------|---------|
| UnixStreamServer error | Lazy loading | `PYSPARK_FIX.md` |
| ModuleNotFoundError | pip install | `QUICK_REFERENCE.md` |
| Java not found | Install JDK | `diagnose_pyspark.py` run outcome |
| Performance lenta | Use Pandas | `SPARK_FEATURES_GUIDE.md` |
| Import fails | `python diagnose_pyspark.py` | diagnose_pyspark.py |

---

## 📋 Arquivos Criados/Modificados

### ✏️ Código Modificado
- `source/spark_features.py` (850+ linhas, lazy loading)

### 📄 Documentação Criada
1. `README_MAIN.md` - **START HERE**
2. `SOLUTION_SUMMARY.md` - Detalhes da solução
3. `PYSPARK_FIX.md` - Fix técnico
4. `QUICK_REFERENCE.md` - Troubleshooting rápido
5. `SPARK_FEATURES_GUIDE.md` - Guia completo
6. `SPARK_USAGE_EXAMPLES.md` - Exemplos
7. `SPARK_IMPLEMENTATION_SUMMARY.md` - Implementação
8. `INDEX.md` - Este arquivo

### 🧪 Testes Criados
- `diagnose_pyspark.py` - Diagnóstico
- `test_feature_engineering.py` - Validação
- `tests/test_spark_features.py` - Unit tests

### 📓 Notebook Atualizado
- `notebooks/05_feature_engineering.ipynb` - 2 células novas

---

## ✅ Checklist para Deploy

- [x] Problema identificado
- [x] Solução implementada
- [x] Codigo testado
- [x] Documentação descrita
- [x] Exemplos fornecidos
- [x] Troubleshooting criado
- [x] Scripts de diagnóstico
- [x] Notebook atualizado

---

## 🎓 Recomendação de Leitura

**Se tem 5 min**:
→ Leia `README_MAIN.md`

**Se tem 15 min**:
→ Leia `README_MAIN.md` + `QUICK_REFERENCE.md`

**Se tem 1 hora**:
→ Leia tudo em ordem:
1. `README_MAIN.md`
2. `SOLUTION_SUMMARY.md`
3. `PYSPARK_FIX.md`
4. `SPARK_FEATURES_GUIDE.md`

---

**Last Updated**: Março 2026  
**Status**: ✅ COMPLETE  
**Version**: 1.0-final

---

## 🚀 Ready to Start?

→ Open `README_MAIN.md` now!
