"""
Configurações Recomendadas - Arquitetura Híbrida

Este arquivo contém perfis de configuração para diferentes cenários.
Copie as configurações apropriadas para source/spark_sampler.py.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

# ============================================================================
# PERFIL 1: MÁQUINA PADRÃO (8 GB RAM, 4 cores)
# ============================================================================
# Uso: Laptop/Desktop comum
# Dataset: HI-Large (~5 GB)
# Tempo esperado: 10-15 minutos

PROFILE_STANDARD = {
    "spark_config": {
        "spark.driver.memory": "6g",
        "spark.executor.memory": "6g",
        "spark.driver.maxResultSize": "3g",
        "spark.sql.shuffle.partitions": "100",
        "spark.default.parallelism": "4"
    },
    "sampling": {
        "SAMPLE_FRACTION": 0.10,      # 10% dos dados
        "MAX_ITERATIONS": 10,
        "P_VALUE_THRESHOLD": 0.05,
        "RANDOM_SEED": 42
    }
}


# ============================================================================
# PERFIL 2: MÁQUINA POTENTE (16+ GB RAM, 8+ cores)
# ============================================================================
# Uso: Workstation, servidor
# Dataset: HI-Large ou maior
# Tempo esperado: 5-10 minutos

PROFILE_HIGH_PERFORMANCE = {
    "spark_config": {
        "spark.driver.memory": "12g",
        "spark.executor.memory": "12g",
        "spark.driver.maxResultSize": "8g",
        "spark.sql.shuffle.partitions": "200",
        "spark.default.parallelism": "8"
    },
    "sampling": {
        "SAMPLE_FRACTION": 0.15,      # 15% dos dados (amostra maior)
        "MAX_ITERATIONS": 15,
        "P_VALUE_THRESHOLD": 0.05,
        "RANDOM_SEED": 42
    }
}


# ============================================================================
# PERFIL 3: MÁQUINA LIMITADA (4 GB RAM, 2 cores)
# ============================================================================
# Uso: Laptop antigo, máquina virtual básica
# Dataset: HI-Large (processamento lento mas funcional)
# Tempo esperado: 20-30 minutos

PROFILE_LOW_RESOURCES = {
    "spark_config": {
        "spark.driver.memory": "3g",
        "spark.executor.memory": "3g",
        "spark.driver.maxResultSize": "2g",
        "spark.sql.shuffle.partitions": "50",
        "spark.default.parallelism": "2"
    },
    "sampling": {
        "SAMPLE_FRACTION": 0.08,      # 8% dos dados (amostra menor)
        "MAX_ITERATIONS": 10,
        "P_VALUE_THRESHOLD": 0.10,    # Mais permissivo
        "RANDOM_SEED": 42
    }
}


# ============================================================================
# PERFIL 4: CLOUD/CLUSTER (Databricks, EMR, Dataproc)
# ============================================================================
# Uso: Ambiente distribuído real
# Dataset: Qualquer tamanho
# Tempo esperado: Muito rápido (< 5 min)

PROFILE_DISTRIBUTED = {
    "spark_config": {
        "spark.driver.memory": "16g",
        "spark.executor.memory": "16g",
        "spark.executor.instances": "4",
        "spark.driver.maxResultSize": "8g",
        "spark.sql.shuffle.partitions": "400",
        "spark.default.parallelism": "16",
        "spark.dynamicAllocation.enabled": "true"
    },
    "sampling": {
        "SAMPLE_FRACTION": 0.20,      # 20% dos dados (maior poder computacional)
        "MAX_ITERATIONS": 20,
        "P_VALUE_THRESHOLD": 0.01,    # Mais rigoroso
        "RANDOM_SEED": 42
    }
}


# ============================================================================
# PERFIL 5: TESTE RÁPIDO (Desenvolvimento)
# ============================================================================
# Uso: Testar código rapidamente
# Dataset: Qualquer
# Tempo esperado: 1-3 minutos

PROFILE_FAST_TEST = {
    "spark_config": {
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.driver.maxResultSize": "2g",
        "spark.sql.shuffle.partitions": "50",
        "spark.default.parallelism": "4"
    },
    "sampling": {
        "SAMPLE_FRACTION": 0.05,      # 5% dos dados (bem pequeno)
        "MAX_ITERATIONS": 5,          # Poucas tentativas
        "P_VALUE_THRESHOLD": 0.10,    # Mais permissivo
        "RANDOM_SEED": 42
    }
}


# ============================================================================
# PERFIL 6: RIGOR ESTATÍSTICO MÁXIMO (Pesquisa Acadêmica)
# ============================================================================
# Uso: TCC, dissertação, paper
# Dataset: HI-Large
# Tempo esperado: 15-30 minutos (mais iterações)

PROFILE_ACADEMIC = {
    "spark_config": {
        "spark.driver.memory": "10g",
        "spark.executor.memory": "10g",
        "spark.driver.maxResultSize": "6g",
        "spark.sql.shuffle.partitions": "200",
        "spark.default.parallelism": "8"
    },
    "sampling": {
        "SAMPLE_FRACTION": 0.15,      # 15% dos dados
        "MAX_ITERATIONS": 30,         # Muitas tentativas
        "P_VALUE_THRESHOLD": 0.01,    # Muito rigoroso (alpha=1%)
        "RANDOM_SEED": 42
    },
    "validation": {
        # Variáveis adicionais para validar
        "NUMERIC_COLS_TO_VALIDATE": [
            'Amount Received',
            'Amount Paid'
        ],
        "CATEGORICAL_COLS_TO_VALIDATE": [
            'Payment Format',
            'From Bank',
            'To Bank',
            'Receiving Currency',
            'Payment Currency'
        ]
    }
}


# ============================================================================
# COMO USAR ESTES PERFIS
# ============================================================================

def apply_profile(profile_name: str = "STANDARD"):
    """
    Gera código Python para aplicar um perfil.
    
    Exemplo de uso:
        python config_profiles.py STANDARD
        python config_profiles.py HIGH_PERFORMANCE
    """
    profiles = {
        "STANDARD": PROFILE_STANDARD,
        "HIGH_PERFORMANCE": PROFILE_HIGH_PERFORMANCE,
        "LOW_RESOURCES": PROFILE_LOW_RESOURCES,
        "DISTRIBUTED": PROFILE_DISTRIBUTED,
        "FAST_TEST": PROFILE_FAST_TEST,
        "ACADEMIC": PROFILE_ACADEMIC
    }
    
    if profile_name not in profiles:
        print(f"❌ Perfil '{profile_name}' não encontrado.")
        print(f"Perfis disponíveis: {list(profiles.keys())}")
        return
    
    profile = profiles[profile_name]
    
    print("="*80)
    print(f"CONFIGURAÇÃO: {profile_name}")
    print("="*80)
    print("\nCopie e cole as linhas abaixo em source/spark_sampler.py:\n")
    
    # Configuração Spark
    print("# Configuração Spark (linhas ~86-92)")
    print("spark = (")
    print("    SparkSession.builder")
    print("    .appName('AntiMoneyLaundering-Sampler')")
    print("    .master('local[*]')")
    
    for key, value in profile["spark_config"].items():
        print(f"    .config('{key}', '{value}')")
    
    print("    .getOrCreate()")
    print(")\n")
    
    # Configuração de amostragem
    print("# Configuração de amostragem (linhas ~37-41)")
    for key, value in profile["sampling"].items():
        if isinstance(value, str):
            print(f"{key} = '{value}'")
        else:
            print(f"{key} = {value}")
    
    print("\n" + "="*80)
    print("✅ Configuração gerada!")
    print("="*80)


# ============================================================================
# GUIA DE ESCOLHA DE PERFIL
# ============================================================================

PROFILE_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║                     GUIA DE ESCOLHA DE PERFIL                            ║
╚══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│ 1. STANDARD - Recomendado para a maioria dos usuários                 │
│    RAM: 8 GB | Cores: 4 | Tempo: 10-15 min                            │
│    Use quando: Laptop/Desktop comum, primeira execução                │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 2. HIGH_PERFORMANCE - Para máquinas potentes                          │
│    RAM: 16+ GB | Cores: 8+ | Tempo: 5-10 min                          │
│    Use quando: Workstation, servidor, múltiplas execuções             │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 3. LOW_RESOURCES - Para máquinas limitadas                            │
│    RAM: 4 GB | Cores: 2 | Tempo: 20-30 min                            │
│    Use quando: Hardware antigo, VM básica                             │
│    ⚠️  Pode falhar se RAM < 4 GB                                       │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 4. DISTRIBUTED - Para ambientes cloud                                 │
│    RAM: Ilimitada | Cores: Muitos | Tempo: < 5 min                    │
│    Use quando: Databricks, AWS EMR, Google Dataproc                   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 5. FAST_TEST - Para desenvolvimento                                   │
│    RAM: Qualquer | Cores: Qualquer | Tempo: 1-3 min                   │
│    Use quando: Testar código, debug, CI/CD                            │
│    ⚠️  Amostra pode não ser válida (5% apenas)                        │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 6. ACADEMIC - Para rigor científico máximo                            │
│    RAM: 10+ GB | Cores: 8+ | Tempo: 15-30 min                         │
│    Use quando: TCC, dissertação, paper acadêmico                      │
│    ✅ Alpha = 0.01 (mais rigoroso que 0.05)                           │
│    ✅ 30 iterações (vs. 10 padrão)                                    │
└────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════╗
║ DICA: Comece com STANDARD. Se falhar, tente LOW_RESOURCES.              ║
║       Se tiver hardware potente, use HIGH_PERFORMANCE.                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print(PROFILE_GUIDE)
    
    if len(sys.argv) > 1:
        profile_name = sys.argv[1].upper()
        apply_profile(profile_name)
    else:
        print("\n💡 USO:")
        print("  python config_profiles.py STANDARD")
        print("  python config_profiles.py HIGH_PERFORMANCE")
        print("  python config_profiles.py ACADEMIC")
        print("\nPerfis disponíveis:")
        print("  - STANDARD")
        print("  - HIGH_PERFORMANCE")
        print("  - LOW_RESOURCES")
        print("  - DISTRIBUTED")
        print("  - FAST_TEST")
        print("  - ACADEMIC")
