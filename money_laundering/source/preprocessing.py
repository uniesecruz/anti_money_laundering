"""
Módulo de Pré-processamento de Dados para Detecção de Lavagem de Dinheiro

Este módulo implementa um pipeline completo e reprodutível de transformação de dados,
garantindo zero data leakage através de fit apenas em treino e transform em OOT.

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from loguru import logger

warnings.filterwarnings('ignore')


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrai features temporais de colunas datetime.
    
    Features extraídas:
    - Componentes: year, month, day, dayofweek, hour, minute, quarter, dayofyear, weekofyear
    - Features cíclicas: sin/cos de month, dayofweek, hour
    - Features de negócio: is_weekend, is_business_hours, is_night
    """
    
    def __init__(self, datetime_cols: List[str]):
        self.datetime_cols = datetime_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in self.datetime_cols:
            if col not in X_copy.columns:
                continue
                
            # Garantir que é datetime
            if X_copy[col].dtype != 'datetime64[ns]':
                X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')
            
            # Componentes temporais básicos
            X_copy[f'{col}_year'] = X_copy[col].dt.year
            X_copy[f'{col}_month'] = X_copy[col].dt.month
            X_copy[f'{col}_day'] = X_copy[col].dt.day
            X_copy[f'{col}_dayofweek'] = X_copy[col].dt.dayofweek
            X_copy[f'{col}_hour'] = X_copy[col].dt.hour
            X_copy[f'{col}_minute'] = X_copy[col].dt.minute
            X_copy[f'{col}_quarter'] = X_copy[col].dt.quarter
            X_copy[f'{col}_dayofyear'] = X_copy[col].dt.dayofyear
            X_copy[f'{col}_weekofyear'] = X_copy[col].dt.isocalendar().week.astype(int)
            
            # Features cíclicas (captura periodicidade)
            X_copy[f'{col}_month_sin'] = np.sin(2 * np.pi * X_copy[f'{col}_month'] / 12)
            X_copy[f'{col}_month_cos'] = np.cos(2 * np.pi * X_copy[f'{col}_month'] / 12)
            X_copy[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X_copy[f'{col}_dayofweek'] / 7)
            X_copy[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X_copy[f'{col}_dayofweek'] / 7)
            X_copy[f'{col}_hour_sin'] = np.sin(2 * np.pi * X_copy[f'{col}_hour'] / 24)
            X_copy[f'{col}_hour_cos'] = np.cos(2 * np.pi * X_copy[f'{col}_hour'] / 24)
            
            # Features de negócio
            X_copy[f'{col}_is_weekend'] = X_copy[f'{col}_dayofweek'].isin([5, 6]).astype(int)
            X_copy[f'{col}_is_business_hours'] = X_copy[f'{col}_hour'].between(9, 17).astype(int)
            X_copy[f'{col}_is_night'] = (
                X_copy[f'{col}_hour'].between(22, 23) | 
                X_copy[f'{col}_hour'].between(0, 5)
            ).astype(int)
            
            # Remover coluna datetime original
            X_copy = X_copy.drop(columns=[col])
        
        return X_copy


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Codifica variáveis categóricas de alta cardinalidade usando frequência.
    
    Fit: Aprende frequências no conjunto de treino
    Transform: Aplica frequências (0 para categorias desconhecidas)
    """
    
    def __init__(self):
        self.freq_maps_ = {}
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        
        for col in X_copy.columns:
            freq_map = X_copy[col].value_counts(normalize=True).to_dict()
            self.freq_maps_[col] = freq_map
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in X_copy.columns:
            if col in self.freq_maps_:
                X_copy[col] = X_copy[col].map(self.freq_maps_[col]).fillna(0)
            else:
                X_copy[col] = 0
        
        return X_copy


class OneHotEncoderSafe(BaseEstimator, TransformerMixin):
    """
    One-Hot Encoding seguro que garante alinhamento entre treino e OOT.
    
    Fit: Aprende categorias do treino
    Transform: Alinha colunas (adiciona colunas faltantes com 0, remove extras)
    """
    
    def __init__(self):
        self.columns_ = None
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        
        # Converter todas as colunas para string
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].astype(str)
        
        # Gerar dummies
        dummies = pd.get_dummies(X_copy, drop_first=False)
        self.columns_ = list(dummies.columns)
        self.feature_names_ = self.columns_
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Converter todas as colunas para string
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].astype(str)
        
        # Gerar dummies
        dummies = pd.get_dummies(X_copy, drop_first=False)
        
        # Adicionar colunas faltantes com 0
        for col in self.columns_:
            if col not in dummies.columns:
                dummies[col] = 0
        
        # Remover colunas extras e reordenar
        dummies = dummies[self.columns_]
        
        return dummies
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class ImputerWithStrategy(BaseEstimator, TransformerMixin):
    """
    Imputador customizado que usa mediana para numéricas.
    
    Fit: Aprende medianas do treino
    Transform: Aplica medianas aprendidas
    """
    
    def __init__(self):
        self.medians_ = {}
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        
        for col in X_copy.columns:
            if X_copy[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                self.medians_[col] = X_copy[col].median()
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in X_copy.columns:
            if col in self.medians_:
                X_copy[col] = X_copy[col].fillna(self.medians_[col])
        
        return X_copy


class AMLPreprocessor:
    """
    Pipeline completo de pré-processamento para detecção de lavagem de dinheiro.
    
    Garante zero data leakage através de:
    - Fit apenas nos dados de treino
    - Transform em treino e OOT usando parâmetros aprendidos do treino
    
    Transformações aplicadas:
    1. Extração de features datetime
    2. Encoding de variáveis categóricas (One-Hot, Target, Frequency)
    3. Transformações numéricas (Yeo-Johnson)
    4. Imputação de valores ausentes
    5. Normalização
    """
    
    def __init__(
        self, 
        target_col: str = 'Is Laundering',
        datetime_cols: Optional[List[str]] = None,
        onehot_cols: Optional[List[str]] = None,
        target_encoding_cols: Optional[List[str]] = None,
        frequency_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        transform_cols: Optional[Dict[str, str]] = None
    ):
        """
        Inicializa o preprocessador.
        
        Args:
            target_col: Nome da coluna target
            datetime_cols: Colunas datetime para extração de features
            onehot_cols: Colunas para One-Hot Encoding (baixa cardinalidade)
            target_encoding_cols: Colunas para Target Encoding (média cardinalidade)
            frequency_cols: Colunas para Frequency Encoding (alta cardinalidade)
            numeric_cols: Colunas numéricas para transformação
            transform_cols: Dict mapeando coluna -> tipo de transformação (yeojohnson, log, sqrt)
        """
        self.target_col = target_col
        self.datetime_cols = datetime_cols or []
        self.onehot_cols = onehot_cols or []
        self.target_encoding_cols = target_encoding_cols or []
        self.frequency_cols = frequency_cols or []
        self.numeric_cols = numeric_cols or []
        self.transform_cols = transform_cols or {}
        
        self.pipeline_ = None
        self.feature_names_ = []
        
    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Identifica automaticamente os tipos de colunas se não fornecidos."""
        
        if not self.datetime_cols:
            # Identificar colunas datetime
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                    if df[col].dtype == 'object':
                        try:
                            pd.to_datetime(df[col])
                            self.datetime_cols.append(col)
                        except:
                            pass
                    elif df[col].dtype == 'datetime64[ns]':
                        self.datetime_cols.append(col)
        
        # Identificar colunas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in self.datetime_cols and col != self.target_col]
        
        if not self.onehot_cols and not self.target_encoding_cols and not self.frequency_cols:
            # Categorizar por cardinalidade
            for col in categorical_cols:
                n_unique = df[col].nunique()
                
                if n_unique <= 10:
                    self.onehot_cols.append(col)
                elif n_unique <= 50:
                    self.target_encoding_cols.append(col)
                else:
                    self.frequency_cols.append(col)
        
        # Identificar colunas numéricas
        if not self.numeric_cols:
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.numeric_cols = [col for col in self.numeric_cols if col != self.target_col]
        
        logger.info(f"Colunas identificadas:")
        logger.info(f"  - Datetime: {len(self.datetime_cols)}")
        logger.info(f"  - One-Hot: {len(self.onehot_cols)}")
        logger.info(f"  - Target Encoding: {len(self.target_encoding_cols)}")
        logger.info(f"  - Frequency: {len(self.frequency_cols)}")
        logger.info(f"  - Numéricas: {len(self.numeric_cols)}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'AMLPreprocessor':
        """
        Ajusta o pipeline nos dados de treino.
        
        Args:
            X: DataFrame de features
            y: Series target (necessário para Target Encoding)
        
        Returns:
            self
        """
        X_copy = X.copy()
        
        # Identificar colunas automaticamente se necessário
        self._identify_columns(X_copy)
        
        # 1. Extração de features datetime
        if self.datetime_cols:
            datetime_extractor = DateTimeFeatureExtractor(self.datetime_cols)
            X_copy = datetime_extractor.fit_transform(X_copy)
        
        # 2. Preparar transformadores categóricos
        transformers = []
        
        # One-Hot Encoding
        if self.onehot_cols:
            onehot_cols_present = [col for col in self.onehot_cols if col in X_copy.columns]
            if onehot_cols_present:
                transformers.append((
                    'onehot',
                    OneHotEncoderSafe(),
                    onehot_cols_present
                ))
        
        # Target Encoding
        if self.target_encoding_cols and y is not None:
            target_cols_present = [col for col in self.target_encoding_cols if col in X_copy.columns]
            if target_cols_present:
                transformers.append((
                    'target',
                    TargetEncoder(smoothing=1.0, min_samples_leaf=10),
                    target_cols_present
                ))
        
        # Frequency Encoding
        if self.frequency_cols:
            freq_cols_present = [col for col in self.frequency_cols if col in X_copy.columns]
            if freq_cols_present:
                transformers.append((
                    'frequency',
                    FrequencyEncoder(),
                    freq_cols_present
                ))
        
        # 3. Transformações numéricas (Yeo-Johnson para variáveis especificadas)
        transform_cols_present = [col for col in self.transform_cols.keys() if col in X_copy.columns]
        if transform_cols_present:
            transformers.append((
                'power',
                PowerTransformer(method='yeo-johnson', standardize=False),
                transform_cols_present
            ))
        
        # Colunas numéricas restantes (passthrough)
        remaining_numeric = [
            col for col in self.numeric_cols 
            if col in X_copy.columns and col not in transform_cols_present
        ]
        if remaining_numeric:
            transformers.append((
                'passthrough',
                'passthrough',
                remaining_numeric
            ))
        
        # 4. Criar ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # 5. Pipeline completo
        self.pipeline_ = Pipeline([
            ('preprocessor', preprocessor),
            ('imputer', ImputerWithStrategy()),
            ('scaler', StandardScaler())
        ])
        
        # Fit do pipeline
        if y is not None and self.target_encoding_cols:
            # Para Target Encoding, precisamos passar y
            self.pipeline_.named_steps['preprocessor'].fit(X_copy, y)
            X_transformed = self.pipeline_.named_steps['preprocessor'].transform(X_copy)
            
            # Converter para DataFrame para imputer e scaler
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            
            feature_names = self._get_feature_names(self.pipeline_.named_steps['preprocessor'])
            X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_copy.index)
            
            self.pipeline_.named_steps['imputer'].fit(X_df)
            X_imputed = self.pipeline_.named_steps['imputer'].transform(X_df)
            self.pipeline_.named_steps['scaler'].fit(X_imputed)
        else:
            self.pipeline_.fit(X_copy)
        
        # Armazenar nomes das features
        self.feature_names_ = self._get_feature_names(self.pipeline_.named_steps['preprocessor'])
        
        logger.success(f"Pipeline ajustado com sucesso! Features: {len(self.feature_names_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma os dados usando o pipeline ajustado.
        
        Args:
            X: DataFrame de features
        
        Returns:
            DataFrame transformado
        """
        if self.pipeline_ is None:
            raise ValueError("Pipeline não foi ajustado. Execute fit() primeiro.")
        
        X_copy = X.copy()
        
        # 1. Extração de features datetime
        if self.datetime_cols:
            datetime_extractor = DateTimeFeatureExtractor(self.datetime_cols)
            X_copy = datetime_extractor.transform(X_copy)
        
        # 2. Aplicar pipeline
        X_transformed = self.pipeline_.transform(X_copy)
        
        # 3. Converter para DataFrame
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        X_df = pd.DataFrame(
            X_transformed, 
            columns=self.feature_names_,
            index=X_copy.index
        )
        
        return X_df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit e transform em uma única chamada."""
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self, column_transformer) -> List[str]:
        """Extrai nomes das features do ColumnTransformer."""
        feature_names = []
        
        for name, transformer, columns in column_transformer.transformers_:
            if name == 'remainder':
                continue
            
            if transformer == 'drop':
                continue
            
            if transformer == 'passthrough':
                feature_names.extend(columns)
            elif hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                except:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        
        return feature_names
    
    def save(self, filepath: Path) -> None:
        """Salva o preprocessador em disco."""
        import joblib
        joblib.dump(self, filepath)
        logger.success(f"Preprocessador salvo em {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'AMLPreprocessor':
        """Carrega o preprocessador do disco."""
        import joblib
        preprocessor = joblib.load(filepath)
        logger.success(f"Preprocessador carregado de {filepath}")
        return preprocessor
