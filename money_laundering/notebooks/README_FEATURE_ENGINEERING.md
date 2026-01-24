# Engenharia de Atributos - Variáveis Categóricas

## 📋 Visão Geral

Este documento descreve o processo completo de engenharia de atributos aplicado às variáveis categóricas do projeto de detecção de lavagem de dinheiro.

## 🎯 Objetivo

Transformar variáveis categóricas e temporais em features numéricas adequadas para treinamento de modelos de machine learning, mantendo a integridade temporal entre os conjuntos de treino e out-of-time (OOT).

## 📊 Pipeline de Processamento

### 1. Carregamento dos Dados
- **Input**: `df_treino.csv` e `df_oot.csv`
- Dataset de treino já com transformações numéricas aplicadas (se disponível)
- Divisão temporal respeitada (80% treino / 20% OOT)

### 2. Identificação de Variáveis
Classificação automática das variáveis em:
- **Numéricas**: Já processadas nas etapas anteriores
- **Categóricas**: Variáveis tipo `object`
- **Datetime**: Variáveis de data/hora (incluindo conversão automática)
- **Target**: Identificação automática da variável alvo

### 3. Análise de Cardinalidade
Para cada variável categórica, determina-se:
- Número de valores únicos
- Percentual de valores ausentes
- Técnica de encoding recomendada baseada em cardinalidade:
  - **Binária (2 valores)** → Label Encoding
  - **Baixa (3-10 valores)** → One-Hot Encoding
  - **Média (11-50 valores)** → Target Encoding
  - **Alta (>50 valores)** → Frequency Encoding

## 🔧 Técnicas de Encoding Aplicadas

### 3.1 Label Encoding (Variáveis Binárias)
**Quando usar**: Variáveis com exatamente 2 categorias

**Implementação**:
```python
LabelEncoder()
```

**Características**:
- Transforma categorias em 0 e 1
- Simples e eficiente
- Valores desconhecidos no OOT recebem valor -1

**Exemplo**:
- `["Sim", "Não"]` → `[1, 0]`

### 3.2 One-Hot Encoding (Baixa Cardinalidade)
**Quando usar**: Variáveis com 3 a 10 categorias

**Implementação**:
```python
pd.get_dummies(drop_first=False)
```

**Características**:
- Cria uma coluna binária para cada categoria
- Não assume ordem entre categorias
- Alinha automaticamente colunas entre treino e OOT
- Categorias ausentes no OOT recebem valor 0

**Exemplo**:
- `["A", "B", "C"]` → 3 colunas: `var_A`, `var_B`, `var_C`

### 3.3 Target Encoding (Média Cardinalidade)
**Quando usar**: Variáveis com 11 a 50 categorias

**Implementação**:
```python
TargetEncoder(smoothing=1.0, min_samples_leaf=10)
```

**Características**:
- Substitui categoria pela média do target naquela categoria
- Reduz dimensionalidade mantendo poder preditivo
- Smoothing para evitar overfitting em categorias raras
- Categorias desconhecidas no OOT usam média global

**Vantagens**:
- Mantém dimensionalidade baixa
- Captura relação direta com target
- Eficiente para alta cardinalidade

**Cuidados**:
- Pode vazar informação do target (mitigado com smoothing)
- Requer validação cruzada apropriada

### 3.4 Frequency Encoding (Alta Cardinalidade)
**Quando usar**: Variáveis com mais de 50 categorias

**Implementação**:
```python
value_counts(normalize=True)
```

**Características**:
- Substitui categoria por sua frequência relativa
- Não depende do target (evita data leakage)
- Categorias desconhecidas no OOT recebem frequência 0

**Exemplo**:
- Categoria que aparece em 15% dos casos → valor 0.15

## 📅 Feature Engineering Temporal

### 4. Extração de Features Datetime

Para cada variável datetime, são extraídas **19 features**:

#### Features Básicas (9)
- `year`: Ano
- `month`: Mês (1-12)
- `day`: Dia do mês
- `dayofweek`: Dia da semana (0=Segunda, 6=Domingo)
- `hour`: Hora (0-23)
- `minute`: Minuto (0-59)
- `quarter`: Trimestre (1-4)
- `dayofyear`: Dia do ano (1-365)
- `weekofyear`: Semana do ano (1-52)

#### Features Cíclicas (6)
Capturam a periodicidade natural do tempo:
- `month_sin`, `month_cos`: Sazonalidade mensal
- `dayofweek_sin`, `dayofweek_cos`: Padrão semanal
- `hour_sin`, `hour_cos`: Padrão diário

**Por que usar features cíclicas?**
- O valor 23h é próximo de 0h (mas numericamente distante)
- Sin/cos capturam essa proximidade circular
- Melhora performance de modelos baseados em distância

#### Features de Negócio (4)
- `is_weekend`: 1 se sábado/domingo, 0 caso contrário
- `is_business_hours`: 1 se entre 9h-17h, 0 caso contrário
- `is_night`: 1 se entre 22h-5h, 0 caso contrário

**Variáveis datetime originais são removidas após extração.**

## 🔄 Tratamento de Missing Values

### Estratégia de Imputação
- **Variáveis numéricas**: Imputação com mediana do treino
- **Variáveis categóricas codificadas**: 
  - Label/One-Hot: Categoria "Unknown" ou valor padrão
  - Target: Média global do target
  - Frequency: Frequência 0

### Princípios
- Imputação sempre baseada apenas no conjunto de treino
- Mesmos valores aplicados no OOT para evitar data leakage
- Registro de todas as estatísticas de imputação

## 🗂️ Estrutura dos Dados Finais

### 7. Remoção de Variáveis Originais
Após encoding, as variáveis originais categóricas e datetime são removidas, mantendo apenas:
- Variáveis numéricas transformadas
- Features extraídas de datetime
- Variáveis categóricas codificadas
- Target (se disponível)

### 8. Separação Features e Target

#### Arquivos Completos
- `df_treino_final.csv`: Dataset completo de treino (features + target)
- `df_oot_final.csv`: Dataset completo OOT (features + target)

#### Arquivos Separados (Pronto para ML)
- `X_train.csv`: Features de treino (apenas numéricas)
- `y_train.csv`: Target de treino
- `X_oot.csv`: Features OOT (apenas numéricas)
- `y_oot.csv`: Target OOT

#### Arquivos de Metadados
- `feature_engineering_info.json`: Informações sobre transformações aplicadas
- `feature_list.txt`: Lista de todas as features finais

## 📈 Outputs Gerados

```
data/processed/
├── df_treino_final.csv          # Dataset completo de treino
├── df_oot_final.csv              # Dataset completo OOT
├── X_train.csv                   # Features de treino
├── y_train.csv                   # Target de treino
├── X_oot.csv                     # Features OOT
├── y_oot.csv                     # Target OOT
├── feature_engineering_info.json # Metadados das transformações
└── feature_list.txt              # Lista de features
```

## ✅ Validações Realizadas

1. **Alinhamento de colunas**: OOT tem exatamente as mesmas features que treino
2. **Tipos de dados**: Todas as features são numéricas
3. **Missing values**: Nenhum valor ausente nos dados finais
4. **Distribuição do target**: Verificação de balanceamento
5. **Integridade temporal**: Sem vazamento de informação do futuro

## 🎓 Boas Práticas Implementadas

### Prevenção de Data Leakage
- ✅ Encoding calculado apenas no treino
- ✅ Target encoding com smoothing adequado
- ✅ Mesmos encoders aplicados no OOT
- ✅ Sem look-ahead nas features temporais

### Escalabilidade
- ✅ Processamento eficiente para datasets grandes
- ✅ Otimização de memória
- ✅ Armazenamento de transformações para produção

### Reprodutibilidade
- ✅ Seeds fixados onde aplicável
- ✅ Documentação completa de transformações
- ✅ Versão das bibliotecas registrada

## 🚀 Próximos Passos

### 1. Feature Selection (Opcional)
Se o número de features for muito alto:
- Análise de correlação
- Importância de features (tree-based)
- Recursive Feature Elimination (RFE)
- PCA para redução de dimensionalidade

### 2. Normalização/Padronização (Condicional)
Dependendo do modelo escolhido:
- **Requer normalização**: SVM, KNN, Redes Neurais, Regressão Logística
- **Não requer**: Árvores de Decisão, Random Forest, XGBoost, LightGBM

Técnicas:
```python
StandardScaler()  # z-score: média=0, std=1
MinMaxScaler()    # Escala para [0,1]
RobustScaler()    # Menos sensível a outliers
```

### 3. Treinamento de Modelos
Os dados estão prontos para:
- Regressão Logística
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Redes Neurais
- SVM

### 4. Validação Temporal
- Treinar no conjunto de treino
- Validar no conjunto OOT
- Avaliar métricas: AUC-ROC, Precision, Recall, F1-Score
- Analisar drift de dados entre treino e OOT

## 📚 Referências

### Técnicas de Encoding
- Micci-Barreca, D. (2001). "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems"
- Cerda, P., & Varoquaux, G. (2020). "Encoding high-cardinality string categorical variables"

### Feature Engineering
- Kuhn, M., & Johnson, K. (2019). "Feature Engineering and Selection"
- Zheng, A., & Casari, A. (2018). "Feature Engineering for Machine Learning"

### Temporal Features
- Taieb, S. B., & Hyndman, R. J. (2014). "Forecasting time series with multiple seasonal patterns"

## 🤝 Contribuindo

Para adicionar novas técnicas de encoding ou feature engineering:
1. Documentar a técnica neste README
2. Implementar no notebook `09_transformacoes_das_variaveis.ipynb`
3. Atualizar os testes de validação
4. Verificar impacto no desempenho dos modelos

---

**Última atualização**: Janeiro 2026
**Notebook**: `09_transformacoes_das_variaveis.ipynb`
**Autor**: Projeto TCC - Detecção de Lavagem de Dinheiro
