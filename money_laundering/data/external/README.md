# External Data Sources

Dados obtidos de fontes externas.

## Fontes Principais

### Kaggle Datasets
- Anti Money Laundering datasets
- Financial fraud detection datasets
- Banking transaction datasets

### APIs Públicas
- Central Bank APIs
- Financial regulatory data
- Economic indicators

## Download e Configuração

Use os scripts em `/scripts/data_download/` para baixar automaticamente.

### Kaggle Setup
1. Configure `kaggle.json` (não versionado)
2. Execute: `python scripts/download_kaggle_data.py`

## Estrutura Típica

```
external/
├── kaggle/
│   ├── anti-money-laundering/
│   └── financial-fraud/
├── apis/
│   ├── central-bank/
│   └── regulatory/
└── manual/
    └── pdfs/
```
