# Data Directory Structure

Este diretório contém todos os dados do projeto Anti-Money Laundering.

## Estrutura

```
data/
├── external/       # Dados externos baixados (Kaggle, APIs, etc.)
├── raw/           # Dados brutos, imutáveis
├── interim/       # Dados intermediários transformados
└── processed/     # Dados finais, prontos para modelagem
```

## Diretrizes

- **external/**: Dados originais de fontes externas
- **raw/**: Primeira cópia dos dados externos, sem modificações
- **interim/**: Dados em processo de limpeza e transformação
- **processed/**: Dados limpos e prontos para análise/modelagem

## Formatos Suportados

- CSV para datasets menores (< 100MB)
- Parquet para datasets maiores (otimizado para Spark)
- JSON para metadados e configurações

## Política de Versionamento

- Grandes datasets não são versionados (listados em .gitignore)
- Metadados e amostras pequenas são versionados
- Use DVC ou similar para versionamento de dados grandes
