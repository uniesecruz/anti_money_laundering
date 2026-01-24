# Instruções de Refatoração dos Notebooks (01-10)

## 🎯 Objetivo

Atualizar TODOS os notebooks existentes para:
1. **Eliminar caminhos absolutos** (hardcoded)
2. **Usar `source/config.py`** para todos os paths
3. **Garantir reprodutibilidade** multiplataforma

---

## 📋 Notebooks a Atualizar

1. `01_dataprep.ipynb` - Preparação de dados
2. `01_harmonization.ipynb` - Harmonização
3. `02_EDA.ipynb` - Análise exploratória
4. `03_data_quality.ipynb` - Qualidade de dados
5. `04_divisao_treino_e_oot.ipynb` - Divisão treino/OOT
6. `05_analise_parametrica.ipynb` - Análise paramétrica
7. `06_analise_nao_parametrica.ipynb` - Análise não-paramétrica
8. `07_transformacoes.ipynb` - Transformações
9. `08_EDA_parte2.ipynb` - EDA parte 2
10. `09_transformacoes_das_variaveis*.ipynb` - Feature engineering
11. `10_treinamento_modelo.ipynb` - Treinamento

---

## 🔧 Mudanças Necessárias

### Para TODOS os notebooks

#### 1. Adicionar Cell de Setup no Início

```python
# ===== CELL 1: Setup de Paths (ADICIONAR NO INÍCIO) =====

# Adicionar source ao path
import sys
from pathlib import Path

# Detectar se estamos em notebook (notebooks/) ou root
if Path.cwd().name == 'notebooks':
    sys.path.insert(0, str(Path.cwd().parent))
else:
    sys.path.insert(0, str(Path.cwd()))

# Importar configuração de paths
from source.config import (
    PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    EXTERNAL_DATA_DIR, MODELS_DIR, FIGURES_DIR,
    get_data_path, get_model_path, get_figure_path
)

from loguru import logger

logger.info(f"📁 Projeto: {PROJ_ROOT}")
logger.info(f"📁 Diretório atual: {Path.cwd()}")
```

#### 2. Substituir Caminhos Hardcoded

**❌ ANTES (ERRADO):**
```python
path_data_raw = r'C:\Users\win\Desktop\TCC\money_laundering\data\raw'
path_accounts_df = r"C:\Users\win\Desktop\TCC\money_laundering\data\external\LI-Medium_accounts.csv"
df = pd.read_csv(r'C:\Users\win\Desktop\TCC\money_laundering\data\processed\df_treino.csv')
```

**✅ DEPOIS (CORRETO):**
```python
# Usar helpers do config.py
accounts_path = get_data_path('LI-Medium_accounts.csv', 'external')
df_treino_path = get_data_path('df_treino.csv', 'processed')

# OU usar diretórios diretamente
accounts_path = EXTERNAL_DATA_DIR / 'LI-Medium_accounts.csv'
df = pd.read_csv(PROCESSED_DATA_DIR / 'df_treino.csv')
```

---

## 📝 Mudanças Específicas por Notebook

### 01_dataprep.ipynb

**Linhas a Modificar:**

**ANTES:**
```python
path_data_raw = r'C:\Users\win\Desktop\TCC\money_laundering\data\raw'
path_accounts_df = r"C:\Users\win\Desktop\TCC\money_laundering\data\external\LI-Medium_accounts.csv"
path_trans_df = r"C:\Users\win\Desktop\TCC\money_laundering\data\external\LI-Medium_Trans.csv"

trans_enriched_df.to_csv(r'C:\Users\win\Desktop\TCC\money_laundering\data\raw\trans_enriched.csv', index=False)
```

**DEPOIS:**
```python
# Carregar dados usando config
accounts_path = get_data_path('LI-Medium_accounts.csv', 'external')
trans_path = get_data_path('LI-Medium_Trans.csv', 'external')

accounts_df = pd.read_csv(accounts_path)
trans_df = pd.read_csv(trans_path)

# Salvar usando config
output_path = get_data_path('trans_enriched.csv', 'raw')
trans_enriched_df.to_csv(output_path, index=False)
logger.success(f"✅ Dados salvos em: {output_path}")
```

---

### 02_EDA.ipynb

**Linhas a Modificar:**

**ANTES:**
```python
# Carregar dados (exemplo genérico)
df = pd.read_csv('../data/processed/df_treino.csv')
```

**DEPOIS:**
```python
# Carregar dados usando config
df_treino_path = get_data_path('df_treino.csv', 'processed')
df = pd.read_csv(df_treino_path)

logger.info(f"Dados carregados de: {df_treino_path}")
```

**Para Salvar Figuras:**

**ANTES:**
```python
plt.savefig('../reports/figures/distribuicao.png')
```

**DEPOIS:**
```python
fig_path = get_figure_path('distribuicao.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
logger.info(f"Figura salva em: {fig_path}")
```

---

### 04_divisao_treino_e_oot.ipynb

**Linhas a Modificar:**

**ANTES:**
```python
df = pd.read_csv('../data/raw/trans_enriched.csv')

# Salvar
df_treino.to_csv('../data/processed/df_treino.csv', index=False)
df_oot.to_csv('../data/processed/df_oot.csv', index=False)
```

**DEPOIS:**
```python
# Carregar
input_path = get_data_path('trans_enriched.csv', 'raw')
df = pd.read_csv(input_path)

# Salvar
df_treino.to_csv(get_data_path('df_treino.csv', 'processed'), index=False)
df_oot.to_csv(get_data_path('df_oot.csv', 'processed'), index=False)

logger.success("✅ Dados salvos em PROCESSED_DATA_DIR")
```

---

### 07_transformacoes.ipynb e 09_transformacoes_das_variaveis*.ipynb

**Linhas a Modificar:**

**ANTES:**
```python
df_treino = pd.read_csv('../data/processed/df_treino.csv')

# Salvar
df_treino_transformed.to_csv('../data/processed/df_treino_transformed.csv', index=False)
```

**DEPOIS:**
```python
# Carregar
df_treino = pd.read_csv(get_data_path('df_treino.csv', 'processed'))

# Salvar (se necessário - mas considere usar pipeline ao invés de intermediários)
output_path = get_data_path('df_treino_transformed.csv', 'processed')
df_treino_transformed.to_csv(output_path, index=False)
```

**⚠️ IMPORTANTE:** Considere **NÃO salvar** arquivos intermediários (`df_treino_transformed.csv`). Use o `pipeline_final.ipynb` ao invés disso.

---

### 10_treinamento_modelo.ipynb

**Linhas a Modificar:**

**ANTES:**
```python
X_train = pd.read_csv('../data/processed/X_train.csv', index_col=0)
y_train_df = pd.read_csv('../data/processed/y_train.csv')

# Salvar modelo
joblib.dump(pipeline, '../models/logistic_regression.pkl')
```

**DEPOIS:**
```python
# Carregar dados
X_train = pd.read_csv(get_data_path('X_train.csv', 'processed'), index_col=0)
y_train_df = pd.read_csv(get_data_path('y_train.csv', 'processed'))

# Salvar modelo
model_path = get_model_path('logistic_regression.pkl')
joblib.dump(pipeline, model_path)
logger.success(f"✅ Modelo salvo em: {model_path}")
```

---

## 🚨 Padrões a Eliminar

### ❌ NÃO FAZER (Paths Absolutos):

```python
r'C:\Users\win\Desktop\...'
'C:/Users/win/Desktop/...'
'/home/user/project/...'
'../../../data/...'  # Relativo demais
```

### ✅ FAZER (Paths Dinâmicos):

```python
get_data_path('filename.csv', 'processed')
PROCESSED_DATA_DIR / 'filename.csv'
EXTERNAL_DATA_DIR / 'dataset.csv'
get_model_path('model.pkl')
get_figure_path('plot.png')
```

---

## 📊 Template de Cell de Setup (Copiar e Colar)

```python
# ========================================================================
# SETUP DE PATHS - USAR source/config.py (NÃO HARDCODED PATHS!)
# ========================================================================

import sys
from pathlib import Path

# Adicionar source ao path
notebook_dir = Path.cwd()
if notebook_dir.name == 'notebooks':
    project_root = notebook_dir.parent
else:
    project_root = notebook_dir

sys.path.insert(0, str(project_root))

# Importar configuração
from source.config import (
    PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODELS_DIR, FIGURES_DIR, get_data_path, get_model_path, get_figure_path
)

from loguru import logger

logger.info(f"📁 Projeto: {PROJ_ROOT}")
logger.info(f"📁 Dados: {DATA_DIR}")
logger.info(f"📁 Modelos: {MODELS_DIR}")

# ========================================================================
```

---

## ✅ Checklist de Validação

Após modificar cada notebook, verificar:

- [ ] Cell de setup adicionada no início
- [ ] TODOS os caminhos usam `config.py` (nenhum hardcoded)
- [ ] Imports relativos removidos (`../data/...`)
- [ ] Caminhos absolutos removidos (`C:\Users\...`)
- [ ] Código executa sem erros
- [ ] Outputs são salvos nos diretórios corretos
- [ ] Logs informativos usando logger

---

## 🎯 Resultado Esperado

Depois das mudanças, o notebook deve:

1. ✅ Funcionar em **qualquer máquina** (Windows/Linux/Mac)
2. ✅ Funcionar em **qualquer diretório** (não depende de path específico)
3. ✅ **Logs claros** indicando onde arquivos estão sendo lidos/salvos
4. ✅ **Zero warnings** de paths não encontrados

---

## 🚀 Execução

Para cada notebook:

1. Abrir no VS Code
2. Adicionar cell de setup no topo
3. Buscar por `r'C:\` ou `'../` no código
4. Substituir por `get_data_path()` ou variáveis do config
5. Executar todas as células
6. Verificar se executa sem erros
7. Commit com mensagem: `refactor: use config.py paths in notebook XX`

---

## 📞 Suporte

Se tiver dúvidas:
- Consulte `source/config.py` para ver funções disponíveis
- Veja `pipeline_final.ipynb` como exemplo de uso correto
- Veja `validate_pipeline.py` para validação

---

**Status:** Pronto para Refatoração  
**Prioridade:** ALTA (Necessário para integridade do TCC)
