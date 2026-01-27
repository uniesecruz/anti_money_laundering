# 📚 Índice de Arquivos - Arquitetura Híbrida PySpark → Pandas

## 🎯 Arquivos Criados/Modificados

### ⭐ Scripts Principais

| Arquivo | Status | Descrição | Linha de Comando |
|---------|--------|-----------|------------------|
| **source/spark_sampler.py** | 🆕 NOVO | Amostragem PySpark com validação estatística | `python source/spark_sampler.py` |
| **source/dataset.py** | ✏️ MODIFICADO | Detecção automática de amostra + integração | `python source/dataset.py` |

---

### 🔧 Utilitários

| Arquivo | Status | Descrição | Linha de Comando |
|---------|--------|-----------|------------------|
| **validate_hybrid_setup.py** | 🆕 NOVO | Valida instalação e configuração | `python validate_hybrid_setup.py` |
| **quick_start.py** | 🆕 NOVO | Guia rápido de uso | `python quick_start.py` |
| **config_profiles.py** | 🆕 NOVO | Perfis de configuração para diferentes cenários | `python config_profiles.py STANDARD` |
| **analyze_sample_quality.py** | 🆕 NOVO | Análise de qualidade estatística da amostra | `python analyze_sample_quality.py` |

---

### 📚 Documentação

| Arquivo | Status | Descrição | Quando Ler |
|---------|--------|-----------|------------|
| **ARQUITETURA_HIBRIDA.md** | 🆕 NOVO | Documentação completa da arquitetura | Leia PRIMEIRO |
| **SUMARIO_TECNICO.md** | 🆕 NOVO | Detalhes técnicos e justificativa acadêmica | Para entender a metodologia |
| **COMANDOS_RAPIDOS.md** | 🆕 NOVO | Comandos essenciais e workflows | Consulta rápida |
| **GUIA_COMPLETO.md** | 🆕 NOVO | Guia definitivo com tudo | Referência completa |
| **INDEX.md** | 🆕 NOVO | Este arquivo (índice de todos os arquivos) | Navegação |

---

### 📓 Notebooks

| Arquivo | Status | Descrição | Como Abrir |
|---------|--------|-----------|------------|
| **notebooks/00_arquitetura_hibrida_tutorial.ipynb** | 🆕 NOVO | Tutorial interativo completo | `jupyter lab notebooks/00_arquitetura_hibrida_tutorial.ipynb` |

---

### ⚙️ Configuração

| Arquivo | Status | Descrição | Observação |
|---------|--------|-----------|------------|
| **requirements.txt** | ✏️ MODIFICADO | Dependências atualizadas (pyspark, pyarrow) | `pip install -r requirements.txt` |

---

## 🗺️ Mapa de Navegação

### Sou novo aqui. Por onde começar?

```
1. Leia:    GUIA_COMPLETO.md
2. Execute: python validate_hybrid_setup.py
3. Leia:    quick_start.py (ou execute para ver instruções)
4. Execute: python source/spark_sampler.py
5. Execute: python source/dataset.py
6. Explore: notebooks/00_arquitetura_hibrida_tutorial.ipynb
```

---

### Quero entender a metodologia estatística

```
1. Leia: ARQUITETURA_HIBRIDA.md (seção "Validação Estatística")
2. Leia: SUMARIO_TECNICO.md (seção "Metodologia Estatística")
3. Execute: python analyze_sample_quality.py
4. Veja: Logs do spark_sampler.py (mostra p-values)
```

---

### Preciso ajustar configurações

```
1. Execute: python config_profiles.py
2. Escolha perfil apropriado (STANDARD, HIGH_PERFORMANCE, etc.)
3. Edite: source/spark_sampler.py com as configurações sugeridas
4. Veja: ARQUITETURA_HIBRIDA.md (seção "Configuração")
```

---

### Tenho problemas/erros

```
1. Consulte: COMANDOS_RAPIDOS.md (seção "Troubleshooting")
2. Consulte: ARQUITETURA_HIBRIDA.md (seção "Troubleshooting")
3. Execute: python validate_hybrid_setup.py
4. Veja: config_profiles.py para perfis alternativos
```

---

### Quero comandos rápidos

```
1. Veja: COMANDOS_RAPIDOS.md
2. Execute: python quick_start.py
```

---

## 📖 Descrição Detalhada dos Arquivos

### 1. source/spark_sampler.py ⭐

**O QUE FAZ:**
- Lê datasets massivos (5+ GB) com PySpark
- Faz join de transações com contas
- Calcula estatísticas da população total
- Gera amostra estratificada (10%)
- **Valida com testes estatísticos** (Z-test, Chi-quadrado)
- Loop iterativo até encontrar amostra válida
- Salva `data/interim/HI-Large_sampled.csv`

**QUANDO USAR:**
- Datasets grandes (HI-Large) que não cabem na memória
- Primeira vez processando dados massivos
- Quando precisar gerar nova amostra

**LINHAS DE CÓDIGO:** ~600 linhas
**TEMPO DE EXECUÇÃO:** 5-15 minutos (máquina padrão)

**CONFIGURAÇÕES PRINCIPAIS:**
```python
SAMPLE_FRACTION = 0.10          # 10% dos dados
MAX_ITERATIONS = 10             # Tentativas
P_VALUE_THRESHOLD = 0.05        # Nível de significância
RANDOM_SEED = 42                # Reprodutibilidade
```

---

### 2. source/dataset.py ✏️

**O QUE MUDOU:**
- ✅ Nova função: `load_sampled_data()`
- ✅ Lógica híbrida na `main()`: detecta amostra automaticamente
- ✅ Novo parâmetro: `use_spark_sample=True`
- ✅ Import atualizado: `INTERIM_DATA_DIR`

**COMO FUNCIONA:**
```python
if use_spark_sample and sampled_path.exists():
    # Carrega amostra PySpark (200 MB)
    df_enriched = load_sampled_data(sampled_path)
else:
    # Carrega originais (5 GB) - pode falhar!
    df_enriched = load_and_enrich_data(...)
```

**QUANDO USAR:**
- Sempre! (após gerar amostra)
- Para fazer split Treino/OOT
- Para salvar dados processados

---

### 3. validate_hybrid_setup.py 🔧

**O QUE FAZ:**
- ✅ Verifica dependências (pandas, pyspark, scipy, loguru)
- ✅ Verifica estrutura de diretórios
- ✅ Verifica scripts principais
- ✅ Verifica datasets originais
- ✅ Verifica se amostra já existe

**OUTPUT ESPERADO:**
```
✓ pandas 2.0.0 instalado
✓ pyspark 3.5.0 instalado
✓ data/ existe
✓ HI-Large_Trans.csv encontrado (4.8 GB)
✓ TUDO PRONTO!
```

**QUANDO USAR:**
- Primeira vez (antes de tudo)
- Após instalar dependências
- Quando tiver problemas

---

### 4. quick_start.py 📋

**O QUE FAZ:**
- Mostra checklist de validação
- Mostra instruções passo a passo
- Mostra dicas úteis
- Detecta estado atual (amostra existe? datasets existem?)

**OUTPUT:**
```
📋 CHECKLIST:
✓ HI-Large_Trans.csv encontrado (4.8 GB)
✗ HI-Large_sampled.csv não encontrado

🚀 PASSO 1: Gerar Amostra PySpark
  python source/spark_sampler.py
  ...
```

**QUANDO USAR:**
- Quando estiver perdido
- Para lembrar próximos passos
- Para ver status do projeto

---

### 5. config_profiles.py ⚙️

**O QUE FAZ:**
- Define 6 perfis de configuração
- Gera código Python para copiar
- Mostra guia de escolha

**PERFIS DISPONÍVEIS:**
1. **STANDARD:** Máquina comum (8 GB RAM)
2. **HIGH_PERFORMANCE:** Workstation (16+ GB RAM)
3. **LOW_RESOURCES:** Máquina limitada (4 GB RAM)
4. **DISTRIBUTED:** Cloud/Cluster
5. **FAST_TEST:** Teste rápido (5% dados)
6. **ACADEMIC:** Rigor máximo (alpha=0.01)

**EXEMPLO DE USO:**
```bash
python config_profiles.py ACADEMIC
```

**OUTPUT:**
```python
# Copie estas linhas para spark_sampler.py:
SAMPLE_FRACTION = 0.15
MAX_ITERATIONS = 30
P_VALUE_THRESHOLD = 0.01
```

---

### 6. analyze_sample_quality.py 📊

**O QUE FAZ:**
- Carrega amostra gerada
- Mostra estatísticas descritivas
- Compara com população (se disponível)
- Executa testes estatísticos
- Mostra p-values detalhados

**QUANDO USAR:**
- Após gerar amostra
- Para verificar qualidade
- Para incluir no TCC (mostrar validação)

---

### 7. ARQUITETURA_HIBRIDA.md 📚

**CONTEÚDO:**
- Visão geral da arquitetura
- Como usar (passo a passo)
- Validação estatística explicada
- Configuração e ajustes
- Troubleshooting
- Comparação de performance
- Justificativa acadêmica

**TAMANHO:** ~400 linhas
**QUANDO LER:** PRIMEIRO! (visão geral completa)

---

### 8. SUMARIO_TECNICO.md 📋

**CONTEÚDO:**
- Detalhes de cada componente criado
- Fundamentação teórica (Z-test, Chi-quadrado)
- Fórmulas matemáticas
- Exemplo de output de validação
- Métricas de sucesso
- Estrutura de arquivos
- Checklist de implementação

**TAMANHO:** ~500 linhas
**QUANDO LER:** Para entender a fundo (metodologia)

---

### 9. COMANDOS_RAPIDOS.md ⚡

**CONTEÚDO:**
- Comandos essenciais
- Workflows comuns (A, B, C)
- Troubleshooting com soluções
- Como verificar resultados
- Como reprocessar

**TAMANHO:** ~200 linhas
**QUANDO LER:** Consulta rápida (cheatsheet)

---

### 10. GUIA_COMPLETO.md 📖

**CONTEÚDO:**
- Sumário de tudo
- O que foi implementado
- Fluxo de uso (3 passos)
- Validação estatística
- Configuração rápida
- Troubleshooting
- Métricas
- Estrutura final
- Checklist

**TAMANHO:** ~400 linhas
**QUANDO LER:** Referência completa (tudo em um lugar)

---

### 11. notebooks/00_arquitetura_hibrida_tutorial.ipynb 📓

**CONTEÚDO:**
- Tutorial interativo
- Células executáveis
- Explicações passo a passo
- Justificativa estatística com LaTeX
- Configurações avançadas
- Diagnóstico de problemas

**QUANDO USAR:**
- Aprender interativamente
- Executar passo a passo
- Ver outputs reais

---

## 🎯 Fluxo Recomendado de Leitura

### Para Iniciantes:
```
1. GUIA_COMPLETO.md              (visão geral)
2. python quick_start.py         (instruções)
3. ARQUITETURA_HIBRIDA.md        (detalhes)
4. python validate_hybrid_setup.py
5. python source/spark_sampler.py
6. python source/dataset.py
```

---

### Para Entendimento Técnico:
```
1. SUMARIO_TECNICO.md            (metodologia)
2. source/spark_sampler.py       (código comentado)
3. python analyze_sample_quality.py
4. notebooks/00_arquitetura_hibrida_tutorial.ipynb
```

---

### Para Resolver Problemas:
```
1. COMANDOS_RAPIDOS.md (seção Troubleshooting)
2. python validate_hybrid_setup.py
3. python config_profiles.py
4. ARQUITETURA_HIBRIDA.md (seção Troubleshooting)
```

---

## 📊 Estatísticas dos Arquivos

| Tipo | Quantidade | Linhas Totais (aprox.) |
|------|------------|------------------------|
| Scripts Python | 6 | ~2500 |
| Documentação Markdown | 5 | ~1500 |
| Notebooks | 1 | ~50 células |
| Modificações | 2 | ~100 linhas |
| **TOTAL** | **14** | **~4100** |

---

## ✅ Verificação Final

Use este checklist para garantir que tudo está ok:

- [ ] Todos os 14 arquivos existem
- [ ] `validate_hybrid_setup.py` mostra "TUDO PRONTO"
- [ ] Documentação está acessível (markdown renderiza)
- [ ] Scripts executam sem erros de import
- [ ] Notebook abre no Jupyter
- [ ] Requirements.txt atualizado
- [ ] Estrutura de diretórios correta

---

## 📞 Ajuda Rápida

| Situação | Arquivo para Consultar |
|----------|------------------------|
| Não sei por onde começar | `GUIA_COMPLETO.md` ou `python quick_start.py` |
| Erro ao executar script | `python validate_hybrid_setup.py` |
| Amostra inválida | `COMANDOS_RAPIDOS.md` (Troubleshooting) |
| Entender testes estatísticos | `SUMARIO_TECNICO.md` (Metodologia) |
| Ajustar configurações | `python config_profiles.py` |
| Ver exemplos práticos | `notebooks/00_arquitetura_hibrida_tutorial.ipynb` |
| Comandos rápidos | `COMANDOS_RAPIDOS.md` |

---

**🎉 Índice completo! Use este arquivo para navegar em toda a implementação.**
