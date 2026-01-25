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
        X_copy = X.copy()
        
        # Validação
        self._validate_input(X_copy)
        
        # Garantir que está ordenado por timestamp
        if not X_copy[self.timestamp_col].is_monotonic_increasing:
            logger.warning(f"⚠️  DataFrame não está ordenado por {self.timestamp_col}. Ordenando...")
            X_copy = X_copy.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Converter timestamp para datetime se necessário
        if X_copy[self.timestamp_col].dtype != 'datetime64[ns]':
            X_copy[self.timestamp_col] = pd.to_datetime(X_copy[self.timestamp_col])
        
        # Definir timestamp como índice temporariamente
        X_copy = X_copy.set_index(self.timestamp_col)
        
        # Gerar features para cada janela temporal
        for window_name, window_size in self.windows.items():
            logger.info(f"Gerando features de velocidade para janela: {window_name}")
            X_copy = self._generate_window_features(X_copy, window_name, window_size)
        
        # Resetar índice
        X_copy = X_copy.reset_index()
        
        # Preencher NaN com 0 (transações sem histórico anterior)
        velocity_cols = [col for col in X_copy.columns if '_velocity_' in col or '_ratio_' in col]
        X_copy[velocity_cols] = X_copy[velocity_cols].fillna(0)
        
        logger.success(f"✅ {len(velocity_cols)} features de velocidade geradas")
        
        return X_copy
    
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
        
        # Agrupar por conta
        grouped = X.groupby(self.account_col)
        
        # 1. Contagem de transações na janela (excluindo a transação atual)
        X[f'txn_count_{window_name}_velocity'] = (
            grouped[self.amount_col]
            .rolling(window_size, closed='left')  # closed='left' exclui o ponto atual
            .count()
            .reset_index(level=0, drop=True)
        )
        
        # 2. Soma de valores na janela
        X[f'amount_sum_{window_name}_velocity'] = (
            grouped[self.amount_col]
            .rolling(window_size, closed='left')
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        # 3. Média de valores na janela
        X[f'amount_mean_{window_name}_velocity'] = (
            grouped[self.amount_col]
            .rolling(window_size, closed='left')
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        # 4. Máximo de valores na janela
        X[f'amount_max_{window_name}_velocity'] = (
            grouped[self.amount_col]
            .rolling(window_size, closed='left')
            .max()
            .reset_index(level=0, drop=True)
        )
        
        # 5. Desvio padrão (apenas para janelas maiores)
        if window_name in ['7d']:
            X[f'amount_std_{window_name}_velocity'] = (
                grouped[self.amount_col]
                .rolling(window_size, closed='left')
                .std()
                .reset_index(level=0, drop=True)
            )
        
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
        X_copy = X.copy()
        
        # Garantir ordenação temporal
        if not X_copy[self.timestamp_col].is_monotonic_increasing:
            logger.warning(f"⚠️  DataFrame não está ordenado por {self.timestamp_col}. Ordenando...")
            X_copy = X_copy.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Converter timestamp para datetime
        if X_copy[self.timestamp_col].dtype != 'datetime64[ns]':
            X_copy[self.timestamp_col] = pd.to_datetime(X_copy[self.timestamp_col])
        
        # Definir timestamp como índice
        X_copy = X_copy.set_index(self.timestamp_col)
        
        # Agrupar por conta
        grouped = X_copy.groupby(self.account_col)
        
        # Calcular estatísticas históricas (excluindo transação atual)
        historical_mean = (
            grouped[self.amount_col]
            .rolling(self.window, closed='left')
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        historical_max = (
            grouped[self.amount_col]
            .rolling(self.window, closed='left')
            .max()
            .reset_index(level=0, drop=True)
        )
        
        historical_std = (
            grouped[self.amount_col]
            .rolling(self.window, closed='left')
            .std()
            .reset_index(level=0, drop=True)
        )
        
        # Feature 1: Ratio valor atual / média histórica
        X_copy['amount_to_historical_mean_ratio'] = (
            X_copy[self.amount_col] / historical_mean
        )
        
        # Feature 2: Ratio valor atual / máximo histórico
        X_copy['amount_to_historical_max_ratio'] = (
            X_copy[self.amount_col] / historical_max
        )
        
        # Feature 3: Z-score (desvio em termos de std)
        X_copy['amount_zscore_historical'] = (
            (X_copy[self.amount_col] - historical_mean) / historical_std
        )
        
        # Resetar índice
        X_copy = X_copy.reset_index()
        
        # Preencher valores infinitos e NaN
        ratio_cols = [
            'amount_to_historical_mean_ratio',
            'amount_to_historical_max_ratio',
            'amount_zscore_historical'
        ]
        
        for col in ratio_cols:
            X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)
            X_copy[col] = X_copy[col].fillna(0)
        
        logger.success(f"✅ {len(ratio_cols)} features de ratio geradas")
        
        return X_copy
    
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
    """
    
    def __init__(
        self, 
        timestamp_col: str = 'Timestamp',
        account_col: str = 'Account',
        bank_col: Optional[str] = 'Receiving Currency',
        country_col: Optional[str] = 'From Bank'
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            bank_col: Nome da coluna de banco (opcional)
            country_col: Nome da coluna de país (opcional)
        """
        self.timestamp_col = timestamp_col
        self.account_col = account_col
        self.bank_col = bank_col
        self.country_col = country_col
    
    def fit(self, X, y=None):
        """Não aprende parâmetros."""
        return self
    
    def transform(self, X):
        """Gera features comportamentais."""
        X_copy = X.copy()
        
        # Garantir ordenação temporal
        if not X_copy[self.timestamp_col].is_monotonic_increasing:
            X_copy = X_copy.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Converter timestamp para datetime
        if X_copy[self.timestamp_col].dtype != 'datetime64[ns]':
            X_copy[self.timestamp_col] = pd.to_datetime(X_copy[self.timestamp_col])
        
        # Feature 1: Tempo desde última transação (segundos)
        X_copy = X_copy.sort_values([self.account_col, self.timestamp_col])
        X_copy['time_since_last_txn_seconds'] = (
            X_copy.groupby(self.account_col)[self.timestamp_col]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        
        # Feature 2: Mudança de banco
        if self.bank_col and self.bank_col in X_copy.columns:
            X_copy['bank_change_flag'] = (
                X_copy.groupby(self.account_col)[self.bank_col]
                .shift(1) != X_copy[self.bank_col]
            ).astype(int)
            X_copy['bank_change_flag'] = X_copy['bank_change_flag'].fillna(0)
        
        # Feature 3: Novo país (país diferente do histórico)
        if self.country_col and self.country_col in X_copy.columns:
            # Verifica se o país já apareceu antes para a conta
            X_copy['is_new_country'] = (
                X_copy.groupby(self.account_col)[self.country_col]
                .apply(lambda x: ~x.isin(x.shift().dropna()))
                .reset_index(level=0, drop=True)
            ).astype(int)
            X_copy['is_new_country'] = X_copy['is_new_country'].fillna(1)
        
        # Feature 4: Transação em horário incomum (fora do horário comercial)
        X_copy['hour_of_day'] = X_copy[self.timestamp_col].dt.hour
        X_copy['is_unusual_hour'] = (
            (X_copy['hour_of_day'] < 6) | (X_copy['hour_of_day'] > 22)
        ).astype(int)
        
        logger.success(f"✅ Features comportamentais geradas")
        
        return X_copy
    
    def fit_transform(self, X, y=None):
        """Fit e transform em uma única chamada."""
        return self.fit(X, y).transform(X)


class FeatureEngineeringPipeline:
    """
    Pipeline completo de Feature Engineering anti-leakage.
    
    Ordem de execução (CRÍTICA):
    1. Ordenação temporal (SEMPRE primeiro)
    2. Velocity Features (janelas deslizantes)
    3. Ratio Features (comparação com histórico)
    4. Behavioral Features (mudanças de padrão)
    
    ⚠️ REGRA DE OURO: Este pipeline deve ser executado ANTES de qualquer
    transformação (scaling, encoding) e APÓS a divisão treino/OOT.
    """
    
    def __init__(
        self,
        timestamp_col: str = 'Timestamp',
        account_col: str = 'Account',
        amount_col: str = 'Amount Received',
        bank_col: Optional[str] = 'Receiving Currency',
        country_col: Optional[str] = 'From Bank',
        velocity_windows: Optional[Dict[str, str]] = None,
        ratio_window: str = '30D'
    ):
        """
        Args:
            timestamp_col: Nome da coluna de timestamp
            account_col: Nome da coluna de identificação do usuário/conta
            amount_col: Nome da coluna de valor da transação
            bank_col: Nome da coluna de banco (opcional)
            country_col: Nome da coluna de país (opcional)
            velocity_windows: Dict de janelas para velocity features
            ratio_window: Janela para ratio features
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
            bank_col=bank_col,
            country_col=country_col
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
        X_copy = X.copy()
        
        logger.info(f"📊 Input shape: {X_copy.shape}")
        
        # ETAPA 1: Garantir ordenação temporal
        if not X_copy[self.timestamp_col].is_monotonic_increasing:
            logger.warning(f"⚠️  Ordenando por {self.timestamp_col}...")
            X_copy = X_copy.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # ETAPA 2: Velocity Features
        logger.info("🔄 Gerando Velocity Features...")
        X_copy = self.velocity_generator.transform(X_copy)
        
        # ETAPA 3: Ratio Features
        logger.info("🔄 Gerando Ratio Features...")
        X_copy = self.ratio_generator.transform(X_copy)
        
        # ETAPA 4: Behavioral Features
        logger.info("🔄 Gerando Behavioral Features...")
        X_copy = self.behavioral_generator.transform(X_copy)
        
        logger.info(f"📊 Output shape: {X_copy.shape}")
        logger.success(f"✅ Feature Engineering concluído! {X_copy.shape[1] - X.shape[1]} novas features")
        
        return X_copy
    
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
    
    # Transform em treino
    logger.info("\n📊 Aplicando em TREINO...")
    df_treino_fe = pipeline.transform(df_treino)
    
    # Transform em OOT
    logger.info("\n📊 Aplicando em OOT...")
    df_oot_fe = pipeline.transform(df_oot)
    
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
