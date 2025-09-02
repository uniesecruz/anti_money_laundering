# Interim Data

Dados em processo de transformação - estados intermediários.

## Propósito

- Dados parcialmente processados
- Resultados de etapas de limpeza
- Features engineering intermediárias
- Dados com transformações aplicadas

## Workflow Típico

1. **raw/** → **interim/** (limpeza básica)
2. **interim/** → **interim/** (transformações)
3. **interim/** → **processed/** (finalização)

## Convenções de Nomenclatura

```
{dataset}_{step}_{version}.{ext}

Exemplos:
- accounts_cleaned_v1.parquet
- transactions_features_v2.parquet
- merged_data_normalized_v3.parquet
```

## Tipos de Transformação

- Limpeza de dados nulos
- Normalização de formatos
- Feature engineering
- Joins e agregações
- Filtros e seleções
