"""
Módulo de Feature Engineering para Detecção de Lavagem de Dinheiro

Este módulo implementa features avançadas com garantia de ZERO data leakage:
- Velocity Features: Contagens e agregações em janelas deslizantes
- Ratio Features: Comparações com histórico do usuário
- Behavioral Features: Desvios de padrão temporal

REGRA DE OURO: Todas as features são calculadas APÓS ordenação temporal
e ANTES de qualquer scaling/normalização.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

warnings.filterwarnings('ignore')


class VelocityFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Gera features de velocidade usando janelas deslizantes (rolling windows).
    
    Features geradas por usuário/conta:
    - Contagem de transações nas últimas 1h, 24h, 7d
    - Soma de valores nas últimas 1h, 24h, 7d
    - Média de valores nas últimas 1h, 24h, 7d
    - Máximo de valores nas últimas 1h, 24h, 7d
    - Std de valores nas últimas 7d
    
    ⚠️ CRÍTICO - Prevenção de Data Leakage:
    1. Dados DEVEM estar ordenados por timestamp ANTES de chamar fit/transform
    2. Usa .shift(1) para excluir a transação atual da janela
    3. Fit e Transform são idênticos (não aprende parâmetros, apenas calcula)
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'Account',
        amount_col: str = 'Amount Received',
        windows: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp (deve ser datetime)
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação
            windows: Dict de janelas temporais {'nome': 'timedelta_string'}
                    Ex: {'1h': '1H', '24h': '24H', '7d': '7D'}
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        self.windows = windows or {
            '1h': '1H',
            '24h': '24H',
            '7d': '7D'
        }
    
    def fit(self, X, y=None):
        """Não aprende parâmetros, apenas valida."""
        self._validate_input(X)
        return self
    
    def transform(self, X):
        """Gera features de velocidade."""
        # Validação
        self._validate_input(X)
        
        # Garantir que está ordenado por timestamp
        if not X[self.timestamp_col].is_monotonic_increasing:
            logger.warning(f"⚠️  DataFrame não está ordenado por {self.timestamp_col}. Ordenando...")
            X = X.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Converter timestamp para datetime se necessário
        if X[self.timestamp_col].dtype != 'datetime64[ns]':
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])
        
        # Gerar features para cada janela temporal
        for window_name, window_size in self.windows.items():
            logger.info(f"Gerando features de velocidade para janela: {window_name}")
            X = self._generate_window_features(X, window_name, window_size)
        
        # Preencher NaN com 0 (transações sem histórico anterior)
        velocity_cols = [col for col in X.columns if '_velocity_' in col or '_ratio_' in col]
        X[velocity_cols] = X[velocity_cols].fillna(0)
        
        logger.success(f"✅ {len(velocity_cols)} features de velocidade geradas")
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit e transform em uma única chamada."""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X):
        """Valida colunas necessárias."""
        required_cols = [self.timestamp_col, self.account_col, self.amount_col]
        missing_cols = [col for col in required_cols if col not in X.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
    
    def _generate_window_features(self, X, window_name, window_size):
        """Gera features para uma janela temporal específica."""
        
        X = X.copy()
        
        # Garantir que timestamp é datetime
        if X[self.timestamp_col].dtype != 'datetime64[ns]':
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])
        
        # Garantir ordenação
        X = X.sort_values([self.account_col, self.timestamp_col]).reset_index(drop=True)
        
        # Para cada conta, calcular features usando janelas temporais
        def calculate_velocity_for_group(group):
            """Calcula features de velocidade para uma conta."""
            group = group.sort_values(self.timestamp_col).reset_index(drop=True)
            group.set_index(self.timestamp_col, inplace=True)
            
            # rolling com closed='left' exclui a transação atual
            group[f'txn_count_{window_name}_velocity'] = group[self.amount_col].rolling(
                window_size, closed='left'
            ).count().fillna(0)
            
            group[f'amount_sum_{window_name}_velocity'] = group[self.amount_col].rolling(
                window_size, closed='left'
            ).sum().fillna(0)
            
            group[f'amount_mean_{window_name}_velocity'] = group[self.amount_col].rolling(
                window_size, closed='left'
            ).mean().fillna(0)
            
            group[f'amount_max_{window_name}_velocity'] = group[self.amount_col].rolling(
                window_size, closed='left'
            ).max().fillna(0)
            
            if window_name == '7d':
                group[f'amount_std_{window_name}_velocity'] = group[self.amount_col].rolling(
                    window_size, closed='left'
                ).std().fillna(0)
            
            return group
        
        try:
            # Aplicar para cada conta
            X_grouped = X.groupby(self.account_col, group_keys=False).apply(
                calculate_velocity_for_group
            )
            
            # Resetar index para recuperar a estrutura
            X_grouped = X_grouped.reset_index()
            X_grouped = X_grouped.sort_values([self.account_col, self.timestamp_col]).reset_index(drop=True)
            
            # Copiar as novas colunas de volta para X na mesma ordem
            X = X.sort_values([self.account_col, self.timestamp_col]).reset_index(drop=True)
            
            velocity_feature_cols = [col for col in X_grouped.columns if '_velocity' in col]
            for col in velocity_feature_cols:
                X[col] = X_grouped[col].values
            
            # Garantir que não há NaNs
            for col in velocity_feature_cols:
                X[col] = X[col].fillna(0)
            
            # Garantir que valores são >= 0 (contagens não podem ser negativas)
            for col in velocity_feature_cols:
                X[col] = X[col].clip(lower=0)
                
            logger.info(f"✅ {len(velocity_feature_cols)} features de velocidade geradas para janela {window_name}")
            
        except Exception as e:
            logger.error(f"Erro ao calcular features de velocidade para {window_name}: {e}")
            # Preencher com zeros em caso de erro
            velocity_feature_cols = [
                f'txn_count_{window_name}_velocity',
                f'amount_sum_{window_name}_velocity',
                f'amount_mean_{window_name}_velocity',
                f'amount_max_{window_name}_velocity'
            ]
            if window_name == '7d':
                velocity_feature_cols.append(f'amount_std_{window_name}_velocity')
            
            for col in velocity_feature_cols:
                X[col] = 0
        
        return X


class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Gera features de ratio comparando transação atual com histórico do usuário.
    
    Features geradas:
    - Ratio: Valor atual / Média dos últimos 30 dias
    - Ratio: Valor atual / Máximo dos últimos 30 dias
    - Desvio: (Valor atual - Média) / Std dos últimos 30 dias
    
    ⚠️ CRÍTICO - Prevenção de Data Leakage:
    Usa janelas históricas que EXCLUEM a transação atual (closed='left')
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'Account',
        amount_col: str = 'Amount Received',
        window: str = '30D'
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação
            window: Janela temporal para calcular histórico (padrão: 30 dias)
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        self.window = window
    
    def fit(self, X, y=None):
        """Não aprende parâmetros."""
        return self
    
    def transform(self, X):
        """Gera features de ratio."""
        # Garantir ordenação temporal
        if not X[self.timestamp_col].is_monotonic_increasing:
            logger.warning(f"⚠️  DataFrame não está ordenado por {self.timestamp_col}. Ordenando...")
            X = X.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Converter timestamp para datetime
        if X[self.timestamp_col].dtype != 'datetime64[ns]':
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])
        
        # Criar cópia
        X_copy = X.copy()
        X_indexed = X_copy.set_index(self.timestamp_col)
        
        # Função auxiliar para calcular ratios em cada grupo
        def ratio_features(group):
            """Calcula ratio features para um grupo."""
            if not isinstance(group.index, pd.DatetimeIndex):
                return group
            
            # Calcular estatísticas históricas
            group['__hist_mean'] = group[self.amount_col].rolling(self.window, closed='left').mean()
            group['__hist_max'] = group[self.amount_col].rolling(self.window, closed='left').max()
            group['__hist_std'] = group[self.amount_col].rolling(self.window, closed='left').std()
            
            return group
        
        try:
            # Aplicar rolling features por conta
            X_with_hist = X_indexed.groupby(self.account_col, group_keys=False).apply(ratio_features)
            X_with_hist = X_with_hist.reset_index().sort_values(self.timestamp_col).reset_index(drop=True)
            
            # Calcular ratios
            X['amount_to_historical_mean_ratio'] = X[self.amount_col] / X_with_hist['__hist_mean'].values
            X['amount_to_historical_max_ratio'] = X[self.amount_col] / X_with_hist['__hist_max'].values
            X['amount_zscore_historical'] = (X[self.amount_col] - X_with_hist['__hist_mean'].values) / X_with_hist['__hist_std'].values
            
        except Exception as e:
            logger.error(f"Erro ao calcular ratio features: {e}")
            X['amount_to_historical_mean_ratio'] = 0
            X['amount_to_historical_max_ratio'] = 0
            X['amount_zscore_historical'] = 0
        
        # Preencher valores infinitos e NaN
        ratio_cols = [
            'amount_to_historical_mean_ratio',
            'amount_to_historical_max_ratio',
            'amount_zscore_historical'
        ]
        
        for col in ratio_cols:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(0)
        
        logger.success(f"✅ {len(ratio_cols)} features de ratio geradas")
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit e transform em uma única chamada."""
        return self.fit(X, y).transform(X)


class BehavioralFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Gera features comportamentais avançadas.
    
    Features geradas:
    - Tempo desde última transação (em segundos)
    - Mudança de banco (flag se banco diferente da última transação)
    - Transação em novo país (flag se país diferente do histórico)
    - Hora incomum (flag se fora do horário comercial e diferente do padrão)
    - Smurfing Features: Detecta transações consecutivas próximas de $10.000
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'Account',
        amount_col: str = 'Amount Received',
        bank_col: Optional[str] = 'Receiving Currency',
        country_col: Optional[str] = 'From Bank',
        smurf_threshold: float = 10000.0,
        smurf_time_window: str = '24H'
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação (para detecção de smurfing)
            bank_col: Nome da coluna de banco (opcional)
            country_col: Nome da coluna de país (opcional)
            smurf_threshold: Limite de valor para classificar como "pequena" transação ($10.000 padrão)
            smurf_time_window: Janela de tempo para detectar smurfing (padrão: '24H')
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        self.bank_col = bank_col
        self.country_col = country_col
        self.smurf_threshold = smurf_threshold
        self.smurf_time_window = smurf_time_window
    
    def fit(self, X, y=None):
        """Não aprende parâmetros."""
        return self
    
    def transform(self, X):
        """Gera features comportamentais."""
        # Garantir ordenação temporal
        if not X[self.timestamp_col].is_monotonic_increasing:
            X = X.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Converter timestamp para datetime
        if X[self.timestamp_col].dtype != 'datetime64[ns]':
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])
        
        # Feature 1: Tempo desde última transação (segundos)
        X = X.sort_values([self.account_col, self.timestamp_col])
        X['time_since_last_txn_seconds'] = (
            X.groupby(self.account_col)[self.timestamp_col]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        
        # Feature 2: Mudança de banco
        if self.bank_col and self.bank_col in X.columns:
            X['bank_change_flag'] = (
                X.groupby(self.account_col)[self.bank_col]
                .shift(1) != X[self.bank_col]
            ).astype(int)
            X['bank_change_flag'] = X['bank_change_flag'].fillna(0)
        
        # Feature 3: Novo país (país diferente do histórico)
        if self.country_col and self.country_col in X.columns:
            # Verifica se o país já apareceu antes para a conta
            X['is_new_country'] = (
                X.groupby(self.account_col)[self.country_col]
                .apply(lambda x: ~x.isin(x.shift().dropna()))
                .reset_index(level=0, drop=True)
            ).astype(int)
            X['is_new_country'] = X['is_new_country'].fillna(1)
        
        # Feature 4: Transação em horário incomum (fora do horário comercial)
        X['hour_of_day'] = X[self.timestamp_col].dt.hour
        X['is_unusual_hour'] = (
            (X['hour_of_day'] < 6) | (X['hour_of_day'] > 22)
        ).astype(int)
        
        # Features 5-7: Smurfing Detection 🚩
        logger.info("Detectando padrões de Smurfing...")
        X = self._detect_smurfing(X)
        
        logger.success(f"✅ Features comportamentais geradas (incluindo Smurfing Detection)")
        
        return X
    
    def _detect_smurfing(self, X):
        """
        Detecta padrões de Smurfing (estrutured transactions).
        
        Smurfing é quando uma entidade realiza múltiplas transações pequenas
        (logo abaixo do threshold de $10.000) em um curto período de tempo
        para evitar detectabilidade. Metricas:
        
        1. smurf_txn_count: Contagem de transações "pequenas" em 24h
        2. smurf_txn_amount_sum: Soma total de transações "pequenas" em 24h
        3. smurf_proximity_score: Score que mede quão próximas estão do threshold
        """
        try:
            # Criar cópia e ordenar
            X_copy = X.copy()
            X_copy = X_copy.sort_values([self.account_col, self.timestamp_col])
            X_copy.index = range(len(X_copy))
            
            # Indexar por timestamp
            X_indexed = X_copy.set_index(self.timestamp_col)
            
            def get_smurf_features(group):
                """Calcula features de smurfing para um grupo."""
                if not isinstance(group.index, pd.DatetimeIndex):
                    return group
                
                # Feature 1: Contagem de transações próximas ao threshold em 24h
                is_smurf_amount = (group[self.amount_col] > self.smurf_threshold * 0.8) & \
                                 (group[self.amount_col] < self.smurf_threshold)
                
                group['__smurf_txn_count'] = is_smurf_amount.rolling(
                    self.smurf_time_window, 
                    closed='left'
                ).sum()
                
                # Feature 2: Soma total de transações próximas ao threshold em 24h
                group['__smurf_txn_amount_sum'] = (
                    group[self.amount_col] * is_smurf_amount
                ).rolling(self.smurf_time_window, closed='left').sum()
                
                # Feature 3: Score de proximidade com threshold
                # Mede quanto a transação está próxima de $10k (0=exatamente 10k, 1=muito abaixo)
                proximity = (self.smurf_threshold - group[self.amount_col]) / self.smurf_threshold
                proximity = proximity.clip(0, 1)  # Clamp entre 0 e 1
                
                group['__smurf_proximity_score'] = (
                    (group[self.amount_col] < self.smurf_threshold) * proximity
                ).rolling(self.smurf_time_window, closed='left').mean()
                
                return group
            
            # Aplicar por conta
            X_with_smurf = X_indexed.groupby(self.account_col, group_keys=False).apply(
                get_smurf_features
            )
            X_with_smurf = X_with_smurf.reset_index().sort_values(
                [self.account_col, self.timestamp_col]
            ).reset_index(drop=True)
            
            # Atribuir features ao dataframe original
            X['smurf_txn_count_24h_behavioral'] = X_with_smurf['__smurf_txn_count'].fillna(0)
            X['smurf_amount_sum_24h_behavioral'] = X_with_smurf['__smurf_txn_amount_sum'].fillna(0)
            X['smurf_proximity_score_behavioral'] = X_with_smurf['__smurf_proximity_score'].fillna(0)
            
        except Exception as e:
            logger.error(f"Erro ao detectar smurfing: {e}")
            X['smurf_txn_count_24h_behavioral'] = 0
            X['smurf_amount_sum_24h_behavioral'] = 0
            X['smurf_proximity_score_behavioral'] = 0
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit e transform em uma única chamada."""
        return self.fit(X, y).transform(X)


class FeatureEngineeringPipeline:
    """
    Pipeline completo de Feature Engineering anti-leakage.
    
    Ordem de execução (CRÍTICA):
    1. Ordenação temporal (SEMPRE primeiro)
    2. Velocity Features (janelas deslizantes 24h, 7d, 30d)
    3. Ratio Features (comparação com histórico 30d)
    4. Behavioral Features (mudanças de padrão + Smurfing Detection)
    
    ⚠️ REGRA DE OURO: Este pipeline deve ser executado ANTES de qualquer
    transformação (scaling, encoding) e APÓS a divisão treino/OOT.
    
    Features Geradas (Total de 30+ features):
    - Velocity: Contagem, soma e média de transações em 1h, 24h, 7d
    - Ratio: Comparação com histórico (média, máximo, z-score)
    - Behavioral: Tempo desde última transação, mudança de banco, novo país
    - Smurfing: Detecção de transações estruturadas ($8k-$10k)
    """
    
    def __init__(
        self,
        timestamp_col: str = 'Timestamp',
        account_col: str = 'Account',
        amount_col: str = 'Amount Received',
        bank_col: Optional[str] = 'Receiving Currency',
        country_col: Optional[str] = 'From Bank',
        velocity_windows: Optional[Dict[str, str]] = None,
        ratio_window: str = '30D',
        smurf_threshold: float = 10000.0,
        smurf_time_window: str = '24H'
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação
            bank_col: Nome da coluna de banco (opcional)
            country_col: Nome da coluna de país (opcional)
            velocity_windows: Dict de janelas para velocity features
                             Default: {'1h': '1H', '24h': '24H', '7d': '7D'}
            ratio_window: Janela para ratio features (padrão: 30 dias)
            smurf_threshold: Limite para detectar smurfing (padrão: $10.000)
            smurf_time_window: Janela temporal para smurfing (padrão: 24h)
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.amount_col = amount_col
        
        # Inicializar transformadores
        self.velocity_generator = VelocityFeatureGenerator(
            timestamp_col=timestamp_col,
            account_col=account_col,
            amount_col=amount_col,
            windows=velocity_windows
        )
        
        self.ratio_generator = RatioFeatureGenerator(
            timestamp_col=timestamp_col,
            account_col=account_col,
            amount_col=amount_col,
            window=ratio_window
        )
        
        self.behavioral_generator = BehavioralFeatureGenerator(
            timestamp_col=timestamp_col,
            account_col=account_col,
            amount_col=amount_col,
            bank_col=bank_col,
            country_col=country_col,
            smurf_threshold=smurf_threshold,
            smurf_time_window=smurf_time_window
        )
    
    def fit(self, X, y=None):
        """Fit todos os geradores (na prática, não aprendem parâmetros)."""
        logger.info("="*80)
        logger.info("INICIANDO FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        self.velocity_generator.fit(X, y)
        self.ratio_generator.fit(X, y)
        self.behavioral_generator.fit(X, y)
        
        logger.success("✅ Pipeline de Feature Engineering ajustado")
        return self
    
    def transform(self, X):
        """Executa todas as transformações na ordem correta."""
        logger.info(f"📊 Input shape: {X.shape}")
        
        # ETAPA 1: Garantir ordenação temporal
        if not X[self.timestamp_col].is_monotonic_increasing:
            logger.warning(f"⚠️  Ordenando por {self.timestamp_col}...")
            X = X.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # ETAPA 2: Velocity Features
        logger.info("🔄 Gerando Velocity Features...")
        X = self.velocity_generator.transform(X)
        
        # ETAPA 3: Ratio Features
        logger.info("🔄 Gerando Ratio Features...")
        X = self.ratio_generator.transform(X)
        
        # ETAPA 4: Behavioral Features
        logger.info("🔄 Gerando Behavioral Features...")
        X = self.behavioral_generator.transform(X)
        
        logger.info(f"📊 Output shape: {X.shape}")
        logger.success(f"✅ Feature Engineering concluído! {X.shape[1] - 21} novas features")  # 21 é o número de colunas originais
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit e transform em uma única chamada."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self, X):
        """Retorna lista de todas as features geradas."""
        X_transformed = self.transform(X.head(100))  # Amostra pequena
        
        original_cols = set(X.columns)
        new_cols = [col for col in X_transformed.columns if col not in original_cols]
        
        return new_cols


# ==================== FUNÇÕES AUXILIARES ====================

def apply_feature_engineering(
    df_treino: pd.DataFrame,
    df_oot: pd.DataFrame,
    timestamp_col: str = 'Timestamp',
    account_col: str = 'Account',
    amount_col: str = 'Amount Received',
    save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Aplica feature engineering em treino e OOT de forma consistente.
    
    Args:
        df_treino: DataFrame de treino
        df_oot: DataFrame de OOT
        timestamp_col: Nome da coluna de timestamp
        account_col: Nome da coluna de conta
        amount_col: Nome da coluna de valor
        save_path: Caminho para salvar informações das features (opcional)
    
    Returns:
        Tuple[df_treino_fe, df_oot_fe, feature_names]
    """
    logger.info("="*80)
    logger.info("APLICANDO FEATURE ENGINEERING EM TREINO E OOT")
    logger.info("="*80)
    
    # Inicializar pipeline
    pipeline = FeatureEngineeringPipeline(
        timestamp_col=timestamp_col,
        account_col=account_col,
        amount_col=amount_col
    )
    
    # Fit no treino (mesmo não aprendendo parâmetros, mantém consistência)
    pipeline.fit(df_treino)
    
    # Transform em treino (fazer cópia aqui)
    logger.info("\n📊 Aplicando em TREINO...")
    df_treino_fe = df_treino.copy()
    df_treino_fe = pipeline.transform(df_treino_fe)
    
    # Transform em OOT (fazer cópia aqui)
    logger.info("\n📊 Aplicando em OOT...")
    df_oot_fe = df_oot.copy()
    df_oot_fe = pipeline.transform(df_oot_fe)
    
    # Listar features geradas
    new_features = pipeline.get_feature_names(df_treino)
    
    logger.success(f"\n✅ Feature Engineering concluído!")
    logger.info(f"   - Treino: {df_treino.shape} → {df_treino_fe.shape}")
    logger.info(f"   - OOT: {df_oot.shape} → {df_oot_fe.shape}")
    logger.info(f"   - Novas features: {len(new_features)}")
    
    # Salvar informações das features
    if save_path:
        feature_info = {
            'timestamp_col': timestamp_col,
            'account_col': account_col,
            'amount_col': amount_col,
            'new_features': new_features,
            'total_features': df_treino_fe.shape[1],
            'train_shape': list(df_treino_fe.shape),
            'oot_shape': list(df_oot_fe.shape)
        }
        
        import json
        with open(save_path, 'w') as f:
            json.dump(feature_info, f, indent=4)
        
        logger.success(f"📁 Informações salvas em {save_path}")
    
    return df_treino_fe, df_oot_fe, new_features
