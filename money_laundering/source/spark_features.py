"""
Módulo de Feature Engineering para PySpark - Detecção de Lavagem de Dinheiro

Este módulo implementa features avançadas em PySpark com garantia de ZERO data leakage:
- Velocity Features: Contagens e agregações em janelas deslizantes
- Ratio Features: Comparações com histórico do usuário
- Behavioral Features: Desvios de padrão temporal

REGRA DE OURO: Todas as features são calculadas APÓS ordenação temporal
e ANTES de qualquer scaling/normalização.

Utilizamos Window Functions do Spark para eficiência máxima em Big Data.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import sys

import pandas as pd
import numpy as np
from loguru import logger

# Type hints - importar durante type checking apenas
if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame, Window
    from pyspark.sql import functions as F

# PySpark imports - LAZY LOADING para evitar erro de socketserver
# Este padrão evita a falha de inicialização do PySpark em Python 3.11+
SparkSession = None
DataFrame = None
Window = None
F = None

def _init_pyspark():
    """Inicializa imports do PySpark apenas quando necessário."""
    global SparkSession, DataFrame, Window, F
    
    if SparkSession is not None:
        return  # Já inicializado
    
    try:
        from pyspark.sql import SparkSession as _SparkSession
        from pyspark.sql import DataFrame as _DataFrame
        from pyspark.sql import Window as _Window
        from pyspark.sql import functions as _F
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
        
        SparkSession = _SparkSession
        DataFrame = _DataFrame
        Window = _Window
        F = _F
        
        logger.success("✅ PySpark inicializado com sucesso")
    except (ImportError, AttributeError) as e:
        logger.error(f"❌ Erro ao inicializar PySpark: {e}")
        logger.error("Execute: pip install pyspark --upgrade")
        raise


class SparkVelocityFeatureGenerator:
    """
    Gera features de velocidade usando Window Functions do Spark.
    
    Features geradas por usuário/conta:
    - Contagem de transações nas últimas 1h, 24h, 7d
    - Soma de valores nas últimas 1h, 24h, 7d
    - Média de valores nas últimas 1h, 24h, 7d
    - Máximo de valores nas últimas 1h, 24h, 7d
    - Std de valores nas últimas 7d
    
    ⚠️ CRÍTICO - Prevenção de Data Leakage:
    1. Dados DEVEM estar ordenados por timestamp ANTES
    2. Usa Window com unbounded preceding e 1 second preceding (exclui a transação atual)
    3. Sem aprende parâmetros (stateless)
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'From Account',
        amount_col: str = 'Amount Received',
        windows: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação
            windows: Dict de janelas temporais {'nome': 'seconds'}
                    Ex: {'1h': 3600, '24h': 86400, '7d': 604800}
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        self.windows = windows or {
            '1h': 3600,
            '24h': 86400,
            '7d': 604800
        }
    
    def fit(self, df_spark: "DataFrame"):
        """Não aprende parâmetros, apenas valida."""
        _init_pyspark()
        self._validate_input(df_spark)
        return self
    
    def transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Gera features de velocidade usando Window Functions."""
        _init_pyspark()
        logger.info("🔄 Gerando Velocity Features em Spark...")
        
        # Validação
        self._validate_input(df_spark)
        
        # Garantir timestamp em datetime
        if df_spark.schema[self.timestamp_col].dataType != TimestampType():
            df_spark = df_spark.withColumn(
                self.timestamp_col,
                F.to_timestamp(self.timestamp_col)
            )
        
        # Ordenar por conta e timestamp
        df_spark = df_spark.orderBy(self.account_col, self.timestamp_col)
        
        # Gerar features para cada janela temporal
        for window_name, window_seconds in self.windows.items():
            logger.info(f"   Gerando janela: {window_name} ({window_seconds}s)")
            df_spark = self._generate_window_features(
                df_spark, 
                window_name, 
                window_seconds
            )
        
        # Preencher NaN com 0
        velocity_cols = [col for col in df_spark.columns if '_velocity_' in col or 'sma_' in col]
        for col in velocity_cols:
            df_spark = df_spark.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
        
        logger.success(f"✅ {len(velocity_cols)} features de velocidade geradas")
        
        return df_spark
    
    def fit_transform(self, df_spark: DataFrame) -> DataFrame:
        """Fit e transform em uma única chamada."""
        return self.fit(df_spark).transform(df_spark)
    
    def _validate_input(self, df_spark: DataFrame):
        """Valida colunas necessárias."""
        required_cols = [self.timestamp_col, self.account_col, self.amount_col]
        missing_cols = [col for col in required_cols if col not in df_spark.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
    
    def _generate_window_features(
        self, 
        df_spark: DataFrame,
        window_name: str,
        window_seconds: int
    ) -> DataFrame:
        """Gera features para uma janela temporal específica usando Window Functions."""
        
        # Definir a window specification
        # unbounded preceding até 1 segundo antes (exclui a transação atual)
        window_spec = Window.partitionBy(self.account_col).orderBy(
            F.col(self.timestamp_col).cast("long")
        ).rangeBetween(-window_seconds, -1)  # Exclui a transação atual
        
        # Contagem de transações
        df_spark = df_spark.withColumn(
            f"txn_count_{window_name}_velocity_",
            F.count(F.col(self.account_col)).over(window_spec)
        )
        
        # Soma de valores
        df_spark = df_spark.withColumn(
            f"amount_sum_{window_name}_velocity_",
            F.sum(F.col(self.amount_col)).over(window_spec)
        )
        
        # Média de valores
        df_spark = df_spark.withColumn(
            f"amount_avg_{window_name}_velocity_",
            F.avg(F.col(self.amount_col)).over(window_spec)
        )
        
        # Máximo de valores
        df_spark = df_spark.withColumn(
            f"amount_max_{window_name}_velocity_",
            F.max(F.col(self.amount_col)).over(window_spec)
        )
        
        # Standard deviation (apenas para 7 dias)
        if window_name == '7d':
            df_spark = df_spark.withColumn(
                f"amount_std_{window_name}_velocity_",
                F.stddev(F.col(self.amount_col)).over(window_spec)
            )
        
        return df_spark


class SparkRatioFeatureGenerator:
    """
    Gera features de ratio comparando transação atual com histórico do usuário.
    
    Features geradas:
    - Ratio: Valor atual / Média dos últimos 30 dias
    - Ratio: Valor atual / Máximo dos últimos 30 dias
    - Desvio: (Valor atual - Média) / Std dos últimos 30 dias
    
    ⚠️ CRÍTICO - Prevenção de Data Leakage:
    Usa janelas históricas que EXCLUEM a transação atual
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'From Account',
        amount_col: str = 'Amount Received',
        window_seconds: int = 2592000  # 30 dias em segundos
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação
            window_seconds: Janela temporal em segundos (padrão: 30 dias = 2592000)
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        self.window_seconds = window_seconds
    
    def fit(self, df_spark: "DataFrame"):
        """Não aprende parâmetros."""
        return self
    
    def transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Gera features de ratio usando Window Functions."""
        _init_pyspark()
        logger.info("🔄 Gerando Ratio Features em Spark...")
        
        # Garantir timestamp em datetime
        if df_spark.schema[self.timestamp_col].dataType != TimestampType():
            df_spark = df_spark.withColumn(
                self.timestamp_col,
                F.to_timestamp(self.timestamp_col)
            )
        
        # Ordenar por conta e timestamp
        df_spark = df_spark.orderBy(self.account_col, self.timestamp_col)
        
        # Window specification para histórico de 30 dias (exclui a transação atual)
        window_spec = Window.partitionBy(self.account_col).orderBy(
            F.col(self.timestamp_col).cast("long")
        ).rangeBetween(-self.window_seconds, -1)
        
        # Calcular estatísticas históricas
        df_spark = df_spark.withColumn(
            "hist_mean_30d",
            F.avg(F.col(self.amount_col)).over(window_spec)
        )
        
        df_spark = df_spark.withColumn(
            "hist_max_30d",
            F.max(F.col(self.amount_col)).over(window_spec)
        )
        
        df_spark = df_spark.withColumn(
            "hist_std_30d",
            F.stddev(F.col(self.amount_col)).over(window_spec)
        )
        
        # Calcular ratios (com proteção contra divisão por zero)
        df_spark = df_spark.withColumn(
            "amount_to_historical_mean_ratio",
            F.when(
                F.col("hist_mean_30d") > 0,
                F.col(self.amount_col) / F.col("hist_mean_30d")
            ).otherwise(0)
        )
        
        df_spark = df_spark.withColumn(
            "amount_to_historical_max_ratio",
            F.when(
                F.col("hist_max_30d") > 0,
                F.col(self.amount_col) / F.col("hist_max_30d")
            ).otherwise(0)
        )
        
        # Z-score (com proteção contra divisão por zero)
        df_spark = df_spark.withColumn(
            "amount_zscore_historical",
            F.when(
                F.col("hist_std_30d") > 0,
                (F.col(self.amount_col) - F.col("hist_mean_30d")) / F.col("hist_std_30d")
            ).otherwise(0)
        )
        
        # Remover colunas temporárias de histórico
        df_spark = df_spark.drop("hist_mean_30d", "hist_max_30d", "hist_std_30d")
        
        # Preencher infinitos com 0
        ratio_cols = [
            'amount_to_historical_mean_ratio',
            'amount_to_historical_max_ratio',
            'amount_zscore_historical'
        ]
        
        for col in ratio_cols:
            df_spark = df_spark.withColumn(
                col,
                F.when(F.isnan(F.col(col)) | F.isinf(F.col(col)), 0).otherwise(F.col(col))
            )
        
        logger.success(f"✅ {len(ratio_cols)} features de ratio geradas")
        
        return df_spark
    
    def fit_transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Fit e transform em uma única chamada."""
        return self.fit(df_spark).transform(df_spark)


class SparkBehavioralFeatureGenerator:
    """
    Gera features comportamentais avançadas em Spark.
    
    Features geradas:
    - Tempo desde última transação (em segundos)
    - Mudança de banco (flag se banco diferente da última transação)
    - Transação em novo país (flag)
    - Hora incomum (flag se fora do horário comercial)
    - Smurfing Features: Detecta transações consecutivas próximas de $10.000
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'From Account',
        amount_col: str = 'Amount Received',
        bank_col: Optional[str] = 'Receiving Currency',
        country_col: Optional[str] = 'From Bank',
        smurf_threshold: float = 10000.0,
        smurf_time_window: int = 86400  # 24 horas em segundos
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Coluna de identificação da conta
            amount_col: Coluna de valor da transação
            bank_col: Nome da coluna de banco (opcional)
            country_col: Nome da coluna de país (opcional)
            smurf_threshold: Limite para classificar como pequena transação
            smurf_time_window: Janela temporal para detectar smurfing em segundos
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        self.bank_col = bank_col
        self.country_col = country_col
        self.smurf_threshold = smurf_threshold
        self.smurf_time_window = smurf_time_window
    
    def fit(self, df_spark: "DataFrame"):
        """Não aprende parâmetros."""
        return self
    
    def transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Gera features comportamentais."""
        _init_pyspark()
        logger.info("🔄 Gerando Behavioral Features em Spark...")
        
        # Garantir timestamp em datetime
        if df_spark.schema[self.timestamp_col].dataType != TimestampType():
            df_spark = df_spark.withColumn(
                self.timestamp_col,
                F.to_timestamp(self.timestamp_col)
            )
        
        # Ordenar por conta e timestamp
        df_spark = df_spark.orderBy(self.account_col, self.timestamp_col)
        
        # Feature 1: Tempo desde última transação (segundos)
        window_spec = Window.partitionBy(self.account_col).orderBy(
            F.col(self.timestamp_col).cast("long")
        )
        
        df_spark = df_spark.withColumn(
            "previous_timestamp",
            F.lag(F.col(self.timestamp_col).cast("long")).over(window_spec)
        )
        
        df_spark = df_spark.withColumn(
            "time_since_last_txn_seconds",
            F.when(
                F.col("previous_timestamp").isNotNull(),
                F.col(self.timestamp_col).cast("long") - F.col("previous_timestamp")
            ).otherwise(0)
        )
        
        df_spark = df_spark.drop("previous_timestamp")
        
        # Feature 2: Mudança de banco
        if self.bank_col and self.bank_col in df_spark.columns:
            window_spec = Window.partitionBy(self.account_col).orderBy(
                F.col(self.timestamp_col).cast("long")
            )
            
            df_spark = df_spark.withColumn(
                "previous_bank",
                F.lag(F.col(self.bank_col)).over(window_spec)
            )
            
            df_spark = df_spark.withColumn(
                "bank_change",
                F.when(
                    F.col("previous_bank").isNotNull() & 
                    (F.col(self.bank_col) != F.col("previous_bank")),
                    1
                ).otherwise(0)
            )
            
            df_spark = df_spark.drop("previous_bank")
        
        # Feature 3: Novo país/banco
        if self.country_col and self.country_col in df_spark.columns:
            # Primeira transação de um determinado país para cada conta
            window_spec_country = Window.partitionBy(
                self.account_col, self.country_col
            ).orderBy(F.col(self.timestamp_col).cast("long"))
            
            df_spark = df_spark.withColumn(
                "country_rank",
                F.row_number().over(window_spec_country)
            )
            
            df_spark = df_spark.withColumn(
                "is_new_country",
                F.when(F.col("country_rank") == 1, 1).otherwise(0)
            )
            
            df_spark = df_spark.drop("country_rank")
        
        # Feature 4: Hora incomum (fora do horário comercial)
        df_spark = df_spark.withColumn(
            "hour_of_day",
            F.hour(F.col(self.timestamp_col))
        )
        
        df_spark = df_spark.withColumn(
            "is_unusual_hour",
            F.when(
                (F.col("hour_of_day") < 6) | (F.col("hour_of_day") > 22),
                1
            ).otherwise(0)
        )
        
        # Features 5-7: Smurfing Detection
        logger.info("   Detectando padrões de Smurfing...")
        df_spark = self._detect_smurfing(df_spark)
        
        logger.success(f"✅ Features comportamentais geradas")
        
        return df_spark
    
    def _detect_smurfing(self, df_spark: DataFrame) -> DataFrame:
        """
        Detecta padrões de Smurfing (structured transactions).
        
        Smurfing é quando uma entidade realiza múltiplas transações pequenas
        para evitar detectabilidade. Métricas:
        
        1. smurf_txn_count: Contagem de transações "pequenas" em 24h
        2. smurf_txn_amount_sum: Soma total de transações "pequenas" em 24h
        3. smurf_proximity_score: Proximidade ao threshold
        """
        try:
            # Flag: transação é pequena? (entre $8k e $10k, típico de smurfing)
            df_spark = df_spark.withColumn(
                "is_small_amount",
                F.when(
                    (F.col(self.amount_col) >= 8000) & 
                    (F.col(self.amount_col) <= self.smurf_threshold),
                    1
                ).otherwise(0)
            )
            
            # Window para smurfing: últimas 24h, mesma conta
            window_spec_smurf = Window.partitionBy(self.account_col).orderBy(
                F.col(self.timestamp_col).cast("long")
            ).rangeBetween(-self.smurf_time_window, -1)
            
            # Contagem de transações pequenas em 24h
            df_spark = df_spark.withColumn(
                "smurf_txn_count_24h",
                F.sum(F.col("is_small_amount")).over(window_spec_smurf)
            )
            
            # Soma total de transações pequenas em 24h
            df_spark = df_spark.withColumn(
                "smurf_txn_amount_sum_24h",
                F.sum(
                    F.when(F.col("is_small_amount") == 1, F.col(self.amount_col)).otherwise(0)
                ).over(window_spec_smurf)
            )
            
            # Score de proximidade (quanto mais próximo de $10k, maior o score)
            df_spark = df_spark.withColumn(
                "smurf_proximity_score",
                F.when(
                    F.col("is_small_amount") == 1,
                    (self.smurf_threshold - F.col(self.amount_col)) / (self.smurf_threshold - 8000)
                ).otherwise(0)
            )
            
            # Remover coluna temporária
            df_spark = df_spark.drop("is_small_amount")
            
            return df_spark
            
        except Exception as e:
            logger.error(f"❌ Erro ao detectar smurfing: {e}")
            return df_spark
    
    def fit_transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Fit e transform em uma única chamada."""
        return self.fit(df_spark).transform(df_spark)


class SparkFeatureEngineeringPipeline:
    """
    Pipeline completo de Feature Engineering em PySpark anti-leakage.
    
    Ordem de execução (CRÍTICA):
    1. Ordenação temporal (SEMPRE primeiro)
    2. Velocity Features (janelas deslizantes 24h, 7d, 30d)
    3. Ratio Features (comparação com histórico 30d)
    4. Behavioral Features (mudanças de padrão + Smurfing Detection)
    
    ⚠️ REGRA DE OURO: Este pipeline deve ser executado ANTES de qualquer
    transformação (scaling, encoding) e APÓS a divisão treino/OOT.
    
    Otimizado para Big Data usando Window Functions do Spark.
    """
    
    def __init__(
        self,
        timestamp_col: str = 'Timestamp',
        account_col: str = 'From Account',
        amount_col: str = 'Amount Received',
        bank_col: Optional[str] = 'Receiving Currency',
        country_col: Optional[str] = 'From Bank',
        velocity_windows: Optional[Dict[str, int]] = None,
        ratio_window_seconds: int = 2592000,  # 30 dias
        smurf_threshold: float = 10000.0,
        smurf_time_window: int = 86400  # 24 horas
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Coluna de identificação da conta
            amount_col: Coluna de valor da transação
            bank_col: Coluna de banco (opcional)
            country_col: Coluna de país (opcional)
            velocity_windows: Dict de janelas em segundos
                             Default: {'1h': 3600, '24h': 86400, '7d': 604800}
            ratio_window_seconds: Janela para ratio features em segundos
            smurf_threshold: Limite para detectar smurfing
            smurf_time_window: Janela temporal para smurfing em segundos
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        
        # Inicializar transformadores
        self.velocity_generator = SparkVelocityFeatureGenerator(
            timestamp_col=timestamp_col,
            account_col=account_col,
            amount_col=amount_col,
            windows=velocity_windows or {
                '1h': 3600,
                '24h': 86400,
                '7d': 604800
            }
        )
        
        self.ratio_generator = SparkRatioFeatureGenerator(
            timestamp_col=timestamp_col,
            account_col=account_col,
            amount_col=amount_col,
            window_seconds=ratio_window_seconds
        )
        
        self.behavioral_generator = SparkBehavioralFeatureGenerator(
            timestamp_col=timestamp_col,
            account_col=account_col,
            amount_col=amount_col,
            bank_col=bank_col,
            country_col=country_col,
            smurf_threshold=smurf_threshold,
            smurf_time_window=smurf_time_window
        )
    
    def fit(self, df_spark: DataFrame):
        """Fit todos os geradores (na prática, não aprendem parâmetros)."""
        _init_pyspark()
        logger.info("="*80)
        logger.info("INICIANDO FEATURE ENGINEERING PIPELINE (SPARK)")
        logger.info("="*80)
        
        self.velocity_generator.fit(df_spark)
        self.ratio_generator.fit(df_spark)
        self.behavioral_generator.fit(df_spark)
        
        logger.success("✅ Pipeline de Feature Engineering ajustado")
        return self
    
    def transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Executa todas as transformações na ordem correta."""
        logger.info(f"📊 Input shape: {df_spark.count()} x {len(df_spark.columns)}")
        
        # ETAPA 1: Garantir ordenação temporal
        if not df_spark.schema[self.timestamp_col]:
            logger.warning(f"⚠️ Timestamp column not found, validating...")
        
        # ETAPA 2: Velocity Features
        logger.info("🔄 Gerando Velocity Features...")
        df_spark = self.velocity_generator.transform(df_spark)
        
        # ETAPA 3: Ratio Features
        logger.info("🔄 Gerando Ratio Features...")
        df_spark = self.ratio_generator.transform(df_spark)
        
        # ETAPA 4: Behavioral Features
        logger.info("🔄 Gerando Behavioral Features...")
        df_spark = self.behavioral_generator.transform(df_spark)
        
        logger.info(f"📊 Output shape: {df_spark.count()} x {len(df_spark.columns)}")
        logger.success(f"✅ Feature Engineering concluído! {len(df_spark.columns) - 21} novas features")
        
        return df_spark
    
    def fit_transform(self, df_spark: "DataFrame") -> "DataFrame":
        """Fit e transform em uma única chamada."""
        return self.fit(df_spark).transform(df_spark)
    
    def get_feature_names(self, df_spark: "DataFrame") -> List[str]:
        """Retorna lista de todas as features geradas."""
        # Para Spark, retornamos todas as colunas geradas
        return df_spark.columns


# ==================== FUNÇÕES AUXILIARES ====================

def apply_feature_engineering_spark(
    df_treino_spark: "DataFrame",
    df_oot_spark: "DataFrame",
    timestamp_col: str = 'Timestamp',
    account_col: str = 'From Account',
    amount_col: str = 'Amount Received',
    save_path_train: Optional[Path] = None,
    save_path_oot: Optional[Path] = None
) -> Tuple["DataFrame", "DataFrame", List[str]]:
    """
    Aplica feature engineering em treino e OOT de forma consistente usando PySpark.
    
    Args:
        df_treino_spark: DataFrame Spark de treino
        df_oot_spark: DataFrame Spark de OOT
        timestamp_col: Nome da coluna de timestamp
        account_col: Nome da coluna de conta
        amount_col: Nome da coluna de valor
        save_path_train: Caminho para salvar treino (opcional)
        save_path_oot: Caminho para salvar OOT (opcional)
    
    Returns:
        Tuple[df_treino_fe_spark, df_oot_fe_spark, feature_names]
    """
    logger.info("="*80)
    logger.info("APLICANDO FEATURE ENGINEERING EM TREINO E OOT (SPARK)")
    logger.info("="*80)
    
    # Inicializar pipeline
    pipeline = SparkFeatureEngineeringPipeline(
        timestamp_col=timestamp_col,
        account_col=account_col,
        amount_col=amount_col
    )
    
    # Fit no treino
    pipeline.fit(df_treino_spark)
    
    # Transform em treino
    logger.info("\n📊 Aplicando em TREINO...")
    df_treino_fe = pipeline.transform(df_treino_spark)
    
    # Transform em OOT
    logger.info("\n📊 Aplicando em OOT...")
    df_oot_fe = pipeline.transform(df_oot_spark)
    
    # Listar features geradas
    original_cols = set(df_treino_spark.columns)
    new_features = [col for col in df_treino_fe.columns if col not in original_cols]
    
    logger.success(f"\n✅ Feature Engineering concluído!")
    logger.info(f"   - Treino: {df_treino_spark.count()} x {df_treino_spark.columns.__len__()} → {df_treino_fe.count()} x {len(df_treino_fe.columns)}")
    logger.info(f"   - OOT: {df_oot_spark.count()} x {df_oot_spark.columns.__len__()} → {df_oot_fe.count()} x {len(df_oot_fe.columns)}")
    logger.info(f"   - Novas features: {len(new_features)}")
    
    # Salvar informações das features
    if save_path_train or save_path_oot:
        feature_info = {
            'timestamp_col': timestamp_col,
            'account_col': account_col,
            'amount_col': amount_col,
            'new_features': new_features,
            'total_features': len(df_treino_fe.columns),
            'train_rows': df_treino_fe.count(),
            'oot_rows': df_oot_fe.count()
        }
        
        import json
        info_path = Path(str(save_path_train).replace('.parquet', '_info.json'))
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.success(f"📁 Informações salvas em {info_path}")
    
    return df_treino_fe, df_oot_fe, new_features
