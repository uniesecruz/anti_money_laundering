# Processed Data

Dados finais, limpos e prontos para modelagem.

## Características

- **Dados limpos**: Sem valores nulos/inconsistentes
- **Features prontas**: Todas as transformações aplicadas
- **Formato otimizado**: Parquet para performance
- **Documentados**: Schema e metadados incluídos

## Datasets Principais

### Para Modelagem
- `final_training_set.parquet` - Conjunto de treino
- `final_test_set.parquet` - Conjunto de teste
- `final_validation_set.parquet` - Conjunto de validação

### Para Análise
- `clean_transactions.parquet` - Transações limpas
- `customer_features.parquet` - Features de clientes
- `risk_indicators.parquet` - Indicadores de risco

## Schema Documentation

Cada dataset deve ter um arquivo `{nome}_schema.json` com:
- Descrição das colunas
- Tipos de dados
- Ranges válidos
- Transformações aplicadas

## Qualidade dos Dados

- ✅ Sem valores nulos em colunas críticas
- ✅ Tipos de dados consistentes
- ✅ Encoding padronizado (UTF-8)
- ✅ Valores dentro de ranges esperados
- ✅ Duplicatas removidas
