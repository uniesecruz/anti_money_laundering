"""
ETAPA 3: Tuning com Bayesian Optimization (Optuna) + Isolation Forest

Implementa:
1. Bayesian Optimization para hyperparameter tuning (especialmente scale_pos_weight)
2. Isolation Forest para anomaly detection como feature
3. Integração com XGBoost/LightGBM cost-aware (ETAPA 2)

Author: TCC - Anti Money Laundering Detection
Date: Marco 2026
"""

import warnings
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score
)
from loguru import logger

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna não disponível. Instale com: pip install optuna")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost não disponível")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM não disponível")


class AnomalyDetector:
    """
    Usa Isolation Forest para detectar anomalias nas transações.
    
    O anomaly_score é usado como feature no modelo principal para
    melhorar detecção de padrões inusitados que podem indicar lavagem.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Args:
            contamination: Proporção esperada de anomalias (padrão: 5%)
            n_estimators: Número de árvores no IF
            random_state: Seed para reprodutibilidade
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """
        Treina Isolation Forest nos dados.
        
        Args:
            X: Features
        
        Returns:
            Self para method chaining
        """
        logger.info(f"Treinando Isolation Forest com {len(X)} amostras...")
        self.model.fit(X)
        self.is_fitted = True
        logger.success("Isolation Forest treinado!")
        return self
    
    def detect(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detecta anomalias (-1 = outlier, 1 = normal).
        
        Args:
            X: Features
        
        Returns:
            Array com classifications (-1 ou 1)
        """
        if not self.is_fitted:
            logger.warning("IF não foi ajustado. Executando fit primeiro...")
            self.fit(X)
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna anomaly scores (quanto mais alto, mais anômalo).
        Útil como feature para o modelo principal.
        
        Args:
            X: Features
        
        Returns:
            Array com anomaly scores normalizados [0, 1]
        """
        if not self.is_fitted:
            logger.warning("IF não foi ajustado. Executando fit primeiro...")
            self.fit(X)
        
        scores = self.model.score_samples(X)
        
        # Normalizar para [0, 1] (onde 1 = muito anômalo)
        # scores.score_samples retorna: quanto mais negativo, mais anômalo
        # Queremos: quanto mais alto, mais anômalo
        # Então: invertemos e normalizamos
        
        # Encontrar min e max para normalizar
        score_min = scores.min()
        score_max = scores.max()
        
        # Normalizar: score_min (mais anômalo) → 1, score_max (mais normal) → 0
        if score_min == score_max:
            anomaly_score_normalized = np.zeros(len(scores))
        else:
            anomaly_score_normalized = 1.0 - (scores - score_min) / (score_max - score_min)
        
        return anomaly_score_normalized
    
    def get_anomaly_ratio(self, X: pd.DataFrame) -> float:
        """
        Retorna proporção de amostras detectadas como anomalias.
        """
        predictions = self.detect(X)
        anomaly_ratio = (predictions == -1).mean()
        return anomaly_ratio


class OptunaHyperparametersOptimizer:
    """
    Otimiza hiperparâmetros usando Bayesian Optimization (Optuna).
    
    Foca especialmente em scale_pos_weight (diferenciação de custo entre classes)
    que é crítico para datasets desbalanceados.
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_trials: int = 50,
        random_state: int = 42,
        timeout: Optional[int] = None,
        direction: str = 'maximize'
    ):
        """
        Args:
            model_type: 'xgboost' ou 'lightgbm'
            n_trials: Número de trials de otimização
            random_state: Seed
            timeout: Timeout em segundos (None = sem limite)
            direction: 'maximize' ou 'minimize' (métrica principal)
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.random_state = random_state
        self.timeout = timeout
        self.direction = direction
        
        self.best_params = None
        self.best_trial = None
        self.study = None
    
    def _objective_xgboost(
        self,
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tscv: TimeSeriesSplit
    ) -> float:
        """Objetivo para otimização XGBoost."""
        
        # Hiperparâmetros para otimizar
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100, log=True),
        }
        
        # Criar modelo com esses parâmetros
        model = xgb.XGBClassifier(
            **params,
            n_estimators=100,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        
        # Validação temporal (CV)
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            
            # Métrica: F1 (equilibra precision e recall)
            score = f1_score(y_fold_val, (y_pred_proba >= 0.5).astype(int), zero_division=0)
            scores.append(score)
            
            # Pruning: parar early se não melhorar
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def _objective_lightgbm(
        self,
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tscv: TimeSeriesSplit
    ) -> float:
        """Objetivo para otimização LightGBM."""
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100, log=True),
        }
        
        model = lgb.LGBMClassifier(
            **params,
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        # Validação temporal
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            
            score = f1_score(y_fold_val, (y_pred_proba >= 0.5).astype(int), zero_division=0)
            scores.append(score)
            
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tscv: Optional[TimeSeriesSplit] = None
    ) -> Dict[str, Any]:
        """
        Executa otimização Bayesiana.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação
            y_val: Target de validação
            tscv: TimeSeriesSplit para CV (se None, usar X_val)
        
        Returns:
            Dict com best_params e study info
        """
        
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna não disponível!")
            return {}
        
        logger.info(f"Iniciando otimização Bayesiana com {self.n_trials} trials...")
        logger.info(f"Modelo: {self.model_type}")
        logger.info(f"Métrica: F1-Score (validation)")
        
        # Criar sampler TPE (Optuna default)
        sampler = TPESampler(seed=self.random_state)
        
        # Criar study
        self.study = optuna.create_study(
            sampler=sampler,
            direction=self.direction
        )
        
        # Objetivo baseado no modelo
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            objective = lambda trial: self._objective_xgboost(
                trial, X_train, y_train, X_val, y_val, tscv
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            objective = lambda trial: self._objective_lightgbm(
                trial, X_train, y_train, X_val, y_val, tscv
            )
        else:
            logger.error(f"Modelo {self.model_type} não disponível!")
            return {}
        
        # Executar otimização
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Extrair best trial
        self.best_trial = self.study.best_trial
        self.best_params = self.best_trial.params
        
        # Log resultados
        logger.success(f"Otimização completa!")
        logger.info(f"Best F1-Score: {self.best_trial.value:.4f}")
        logger.info(f"Best Params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_trial.value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def get_best_model(self) -> Any:
        """Retorna modelo com best params."""
        if self.best_params is None:
            logger.error("Otimização não foi executada!")
            return None
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                **self.best_params,
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = lgb.LGBMClassifier(
                **self.best_params,
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            return None
        
        return model


class AMLTunerPipeline:
    """
    Pipeline completo que integra:
    1. Anomaly Detection (Isolation Forest)
    2. Bayesian Optimization (Optuna)
    3. Model training com best params
    4. Feature engineering com anomaly scores
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_optuna_trials: int = 50,
        anomaly_contamination: float = 0.05,
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'xgboost' ou 'lightgbm'
            n_optuna_trials: Número de trials Optuna
            anomaly_contamination: Proporção para IF
            random_state: Seed
        """
        self.model_type = model_type
        self.n_optuna_trials = n_optuna_trials
        self.anomaly_contamination = anomaly_contamination
        self.random_state = random_state
        
        self.anomaly_detector = AnomalyDetector(
            contamination=anomaly_contamination,
            random_state=random_state
        )
        
        self.optimizer = OptunaHyperparametersOptimizer(
            model_type=model_type,
            n_trials=n_optuna_trials,
            random_state=random_state
        )
        
        self.best_model = None
        self.X_train_augmented = None
        self.X_val_augmented = None
    
    def add_anomaly_features(
        self,
        X: pd.DataFrame,
        anomaly_detector: AnomalyDetector,
        feature_name: str = 'anomaly_score'
    ) -> pd.DataFrame:
        """
        Adiciona anomaly scores como feature.
        
        Args:
            X: Features originais
            anomaly_detector: Fitted AnomalyDetector
            feature_name: Nome da nova feature
        
        Returns:
            X com anomaly feature adicionada
        """
        X_augmented = X.copy()
        anomaly_scores = anomaly_detector.get_anomaly_score(X)
        X_augmented[feature_name] = anomaly_scores
        
        logger.info(f"Feature '{feature_name}' adicionada. Min={anomaly_scores.min():.4f}, Max={anomaly_scores.max():.4f}")
        
        return X_augmented
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tscv: Optional[TimeSeriesSplit] = None
    ) -> 'AMLTunerPipeline':
        """
        Executa pipeline completo.
        
        Args:
            X_train: Features treino
            y_train: Target treino
            X_val: Features validação
            y_val: Target validação
            tscv: TimeSeriesSplit
        
        Returns:
            Self para method chaining
        """
        
        logger.info("="*80)
        logger.info("ETAPA 3: TUNING COM OPTUNA + ISOLATION FOREST")
        logger.info("="*80)
        
        # Passo 1: Treinar Isolation Forest
        logger.info("\n[1/3] Treinando Isolation Forest...")
        self.anomaly_detector.fit(X_train)
        anomaly_ratio_train = self.anomaly_detector.get_anomaly_ratio(X_train)
        anomaly_ratio_val = self.anomaly_detector.get_anomaly_ratio(X_val)
        logger.info(f"  Anomalias detectadas - Treino: {anomaly_ratio_train:.2%}, Validação: {anomaly_ratio_val:.2%}")
        
        # Passo 2: Adicionar anomaly features
        logger.info("\n[2/3] Adicionando anomaly scores como features...")
        X_train_aug = self.add_anomaly_features(X_train, self.anomaly_detector)
        X_val_aug = self.add_anomaly_features(X_val, self.anomaly_detector)
        
        # Passo 3: Otimizar hiperparâmetros
        logger.info("\n[3/3] Executando Bayesian Optimization...")
        
        if tscv is None:
            logger.warning("tscv não fornecido, usando validation set único")
            tscv = TimeSeriesSplit(n_splits=1)
        
        optim_result = self.optimizer.optimize(
            X_train_aug.values,
            y_train.values,
            X_val_aug.values,
            y_val.values,
            tscv=tscv
        )
        
        # Passo 4: Treinar modelo final com best params
        logger.info("\nTreinando modelo final com best params...")
        self.best_model = self.optimizer.get_best_model()
        self.best_model.fit(X_train_aug, y_train)
        
        # Guardar dados aumentados
        self.X_train_augmented = X_train_aug
        self.X_val_augmented = X_val_aug
        
        logger.success("Pipeline ETAPA 3 completado!")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições binárias (0/1) com modelo final."""
        if self.best_model is None:
            logger.error("Pipeline não foi ajustado!")
            return None
        
        X_aug = self.add_anomaly_features(X, self.anomaly_detector)
        preds = self.best_model.predict(X_aug)
        # Ensure output is always int array (0 or 1)
        return np.asarray(preds, dtype=int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades."""
        if self.best_model is None:
            logger.error("Pipeline não foi ajustado!")
            return None
        
        X_aug = self.add_anomaly_features(X, self.anomaly_detector)
        return self.best_model.predict_proba(X_aug)
