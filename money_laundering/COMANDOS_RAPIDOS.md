# Comandos Rápidos - Arquitetura Híbrida

## 📋 Comandos Essenciais

### 1. Validar Setup (Primeira Vez)
```bash
cd money_laundering
python validate_hybrid_setup.py
```

### 2. Ver Instruções de Uso
```bash
python quick_start.py
```

### 3. Gerar Amostra PySpark (Uma Vez para HI-Large)
```bash
python source/spark_sampler.py
```

### 4. Processar com Pandas (Sempre)
```bash
python source/dataset.py
```

---

## 🎯 Workflows Comuns

### Workflow A: Primeira Execução (HI-Large)
```bash
# 1. Validar
python validate_hybrid_setup.py

# 2. Instalar dependências (se necessário)
pip install pyspark pyarrow scipy loguru

# 3. Gerar amostra
python source/spark_sampler.py   # 10-15 min

# 4. Processar
python source/dataset.py          # 30 seg - 2 min

# 5. Continuar com notebooks
jupyter lab notebooks/05_feature_engineering.ipynb
```

---

### Workflow B: Execuções Subsequentes
```bash
# A amostra já existe, basta processar
python source/dataset.py

# Seguir para feature engineering
jupyter lab
```

---

### Workflow C: Datasets Pequenos (LI-Medium)
```bash
# NÃO precisa do Spark! Use Pandas direto
# Editar source/dataset.py linha ~235:
#   main(dataset='LI-Medium', use_spark_sample=False)

python source/dataset.py
```

---

## 🔧 Troubleshooting

### Amostra inválida após 10 iterações
```bash
# Editar source/spark_sampler.py
# Aumentar: SAMPLE_FRACTION = 0.15 (de 0.10)
# Aumentar: MAX_ITERATIONS = 20 (de 10)

python source/spark_sampler.py
```

### OutOfMemoryError no Spark
```bash
# Editar source/spark_sampler.py linhas ~86-87
# Aumentar: spark.driver.memory = "12g" (de "8g")

python source/spark_sampler.py
```

### PySpark não instalado
```bash
pip install pyspark>=3.5.0 pyarrow>=14.0.0
```

### Dataset.py ignora amostra
```bash
# Verificar se arquivo existe
ls data/interim/HI-Large_sampled.csv

# Verificar configuração em dataset.py
# Linha ~235: use_spark_sample=True
```

---

## 📊 Verificar Resultados

### Ver estatísticas da amostra
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/interim/HI-Large_sampled.csv')
print(f'Shape: {df.shape}')
print(f'Target: {df[\"Is Laundering\"].value_counts().to_dict()}')
"
```

### Ver estatísticas de treino/OOT
```bash
python -c "
import pandas as pd
treino = pd.read_csv('data/processed/df_treino.csv')
oot = pd.read_csv('data/processed/df_oot.csv')
print(f'Treino: {treino.shape} | Target: {treino[\"Is Laundering\"].value_counts().to_dict()}')
print(f'OOT: {oot.shape} | Target: {oot[\"Is Laundering\"].value_counts().to_dict()}')
"
```

---

## 🧪 Testar Validação Estatística

### Ver output detalhado do Spark Sampler
```bash
python source/spark_sampler.py 2>&1 | tee spark_sampler_log.txt
```

O output mostrará:
```
Amount Received           | Z=  0.123 | p=0.9021 | ✓ PASS
Amount Paid               | Z= -0.456 | p=0.6485 | ✓ PASS
Payment Format            | χ²= 2.341 | p=0.5043 | ✓ PASS
...
✓ Amostra VÁLIDA! Todos os testes passaram (p > 0.05)
```

---

## 📚 Documentação

### Ver README da arquitetura
```bash
cat ARQUITETURA_HIBRIDA.md
# ou no Windows:
type ARQUITETURA_HIBRIDA.md
```

### Ver sumário técnico
```bash
cat SUMARIO_TECNICO.md
```

### Abrir tutorial interativo
```bash
jupyter lab notebooks/00_arquitetura_hibrida_tutorial.ipynb
```

---

## 🔄 Reprocessar do Zero

### Deletar amostra e reprocessar
```bash
# Windows
del data\interim\HI-Large_sampled.csv
python source\spark_sampler.py

# Linux/Mac
rm data/interim/HI-Large_sampled.csv
python source/spark_sampler.py
```

### Deletar treino/OOT e reprocessar
```bash
# Windows
del data\processed\df_treino.csv
del data\processed\df_oot.csv
python source\dataset.py

# Linux/Mac
rm data/processed/df_treino.csv data/processed/df_oot.csv
python source/dataset.py
```

---

## 🎓 Citação para TCC

Se usar esta arquitetura no TCC, cite:

```bibtex
@misc{anti_money_laundering_hybrid,
  title={Arquitetura Híbrida PySpark-Pandas para Detecção de Lavagem de Dinheiro},
  author={[Seu Nome]},
  year={2026},
  note={Implementação de amostragem estatisticamente validada 
        usando testes de hipótese (Z-test, Chi-quadrado) 
        para processamento de datasets massivos.}
}
```

**Referências:**
- Cochran, W. G. (1977). *Sampling Techniques*
- Montgomery, D. C. (2017). *Design and Analysis of Experiments*
- Zaharia et al. (2016). *Apache Spark: A Unified Engine for Big Data Processing*

---

## ✅ Checklist de Validação

Antes de considerar concluído, verifique:

- [ ] `validate_hybrid_setup.py` mostra "TUDO PRONTO"
- [ ] `HI-Large_sampled.csv` gerado com sucesso
- [ ] Logs mostram todos os testes estatísticos com p > 0.05
- [ ] `df_treino.csv` e `df_oot.csv` criados
- [ ] Proporções do target mantidas (estratificação funcionou)
- [ ] Notebooks conseguem carregar os dados processados

---

**Sucesso! 🚀**
