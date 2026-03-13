"""
Modulo de Otimizacao de Threshold e Custos (ETAPA 2)

Implementa cost-sensitive learning com matriz de custo diferenciada
para clientes de alta renda/historico longo, visando melhorar precisao
e recall em deteccao de lavagem de dinheiro.

Autor: TCC - Anti Money Laundering Detection
Data: Marco 2026
"""

from typing import Dict, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from loguru import logger

warnings.filterwarnings('ignore')


class ClientValueClassifier:
    """
    Classifica clientes como 'high-value' ou 'standard' baseado em
    características de renda e histórico.
    
    Um cliente high-value é aquele que:
    1. Tem renda/patrimônio elevado (top 20-30%)
    2. Tem mais de X transações (histórico longo)
    3. Tem conta há mais de Y dias
    
    A lógica é que para clientes high-value, um FP é mais custoso
    (causa mais atrito) que para clientes padrão.
    """
    
    def __init__(
        self,
        income_percentile: float = 75.0,
        min_transaction_count: int = 50,
        min_account_age_days: int = 180
    ):
        """
        Args:
            income_percentile: Percentil para classificar como high-income (padrão: 75%)
            min_transaction_count: Mínimo de transações para histórico longo
            min_account_age_days: Mínimo de dias de conta aberta
        """
        self.income_percentile = income_percentile
        self.min_transaction_count = min_transaction_count
        self.min_account_age_days = min_account_age_days
        
        self.income_threshold = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, income_col: str = 'Amount Received') -> 'ClientValueClassifier':
        """
        Calcula threshold de income baseado no percentil.
        
        Args:
            df: DataFrame com dados de clientes
            income_col: Coluna para calcular income (usa máximo por cliente)
        
        Returns:
            Self para method chaining
        """
        # Calcular renda máxima por cliente (proxy para income)
        if 'Account' in df.columns:
            client_max_income = df.groupby('Account')[income_col].max()
        else:
            client_max_income = df[income_col]
        
        # Calcular threshold
        self.income_threshold = np.percentile(
            client_max_income, 
            self.income_percentile
        )
        
        self.is_fitted = True
        logger.info(f"Client value classifier ajustado. Income threshold: ${self.income_threshold:.2f}")
        
        return self
    
    def classify(self, df: pd.DataFrame, income_col: str = 'Amount Received') -> pd.Series:
        """
        Classifica cada transação como high-value ou standard.
        
        Args:
            df: DataFrame com transações
            income_col: Coluna de valor
        
        Returns:
            Series com classificação (1=high-value, 0=standard)
        """
        if not self.is_fitted and self.income_threshold is None:
            logger.warning("Classifier não foi ajustado. Usando threshold padrão.")
            if 'Account' in df.columns:
                client_max_income = df.groupby('Account')[income_col].max()
                threshold = np.percentile(client_max_income, self.income_percentile)
            else:
                threshold = np.percentile(df[income_col], self.income_percentile)
        else:
            threshold = self.income_threshold
        
        # Classificar: high-value se valor >= threshold
        is_high_value = (df[income_col] >= threshold).astype(int)
        
        return is_high_value
    
    def score(self, df: pd.DataFrame, income_col: str = 'Amount Received') -> float:
        """
        Retorna proporção de clientes high-value.
        """
        is_high = self.classify(df, income_col)
        return is_high.mean()


class CostMatrix:
    """
    Define matriz de custo customizada para False Positives e False Negatives.
    
    Matriz de Custo Padrão:
    
                Predicted Negative    Predicted Positive
    Actual Negative  (TN: 0)         (FP: cost_fp) ← Bloquear cliente legítimo
    Actual Positive  (FN: cost_fn)   (TP: 0)      ← Não detectar fraude
    
    Para clientes high-value:
    - FP cost é MAIOR (cliente de alto valor, mais atrito se bloqueado)
    - FN cost é IGUAL (fraude sempre custosa)
    """
    
    def __init__(
        self,
        cost_fp_standard: float = 50.0,      # Custo de bloquear cliente padrão
        cost_fp_highvalue: float = 500.0,    # Custo de bloquear cliente de alta renda
        cost_fn: float = 10000.0             # Custo de não detectar fraude
    ):
        """
        Args:
            cost_fp_standard: Custo de Falso Positivo para cliente padrão
            cost_fp_highvalue: Custo de Falso Positivo para cliente high-value
            cost_fn: Custo de Falso Negativo (fraude não detectada)
        """
        self.cost_fp_standard = cost_fp_standard
        self.cost_fp_highvalue = cost_fp_highvalue
        self.cost_fn = cost_fn
        
        logger.info(f"Matriz de custo configurada:")
        logger.info(f"  FP (Standard): ${cost_fp_standard:.2f}")
        logger.info(f"  FP (High-Value): ${cost_fp_highvalue:.2f}")
        logger.info(f"  FN (Fraude não detectada): ${cost_fn:.2f}")
    
    def get_cost_per_sample(self, y_pred: np.array, y_true: np.array, 
                           is_high_value: np.array) -> np.array:
        """
        Calcula custo individual para cada amostra.
        
        Args:
            y_pred: Predições binárias (0 ou 1)
            y_true: Valores reais
            is_high_value: Array booleano indicando clientes high-value
        
        Returns:
            Array com custo por amostra
        """
        costs = np.zeros(len(y_pred))
        
        # FN: Prediz 0 mas é 1 (fraude não detectada)
        fn_mask = (y_pred == 0) & (y_true == 1)
        costs[fn_mask] = self.cost_fn
        
        # FP Standard: Prediz 1 mas é 0 (bloqueia cliente padrão legítimo)
        fp_standard_mask = (y_pred == 1) & (y_true == 0) & (is_high_value == 0)
        costs[fp_standard_mask] = self.cost_fp_standard
        
        # FP High-Value: Prediz 1 mas é 0 (bloqueia cliente wealthy legítimo)
        fp_highvalue_mask = (y_pred == 1) & (y_true == 0) & (is_high_value == 1)
        costs[fp_highvalue_mask] = self.cost_fp_highvalue
        
        # TP e TN: custo zero (decisões corretas)
        
        return costs
    
    def get_sample_weights(self, y_true: np.array, is_high_value: np.array) -> np.array:
        """
        Calcula sample_weight para XGBoost/LightGBM baseado na matriz de custo.
        
        Usa a probabilidade de erro para cada classe:
        - Amostras FN têm peso alto (fraude não detectada é caro)
        - Amostras FP high-value têm peso alto (cliente rico bloqueado é caro)
        """
        weights = np.ones(len(y_true))
        
        # FN: Peso = cost_fn
        fn_mask = (y_true == 1)
        weights[fn_mask] = self.cost_fn / self.cost_fn  # Normalizável
        
        # Positive class: Peso baseado no custo FN
        weights[fn_mask] *= self.cost_fn / 100.0
        
        # High-value negative: Peso baseado no custo FP
        hv_negative_mask = (y_true == 0) & (is_high_value == 1)
        weights[hv_negative_mask] = self.cost_fp_highvalue / 50.0
        
        # Normalizar weights para média = 1
        weights = weights / weights.mean()
        
        return weights
    
    def total_cost(self, y_pred: np.array, y_true: np.array, 
                  is_high_value: np.array) -> float:
        """Calcula custo total dadas as predições."""
        costs = self.get_cost_per_sample(y_pred, y_true, is_high_value)
        return costs.sum()


class ThresholdOptimizer:
    """
    Otimiza o threshold de classificação para minimizar custo total.
    
    Em vez de usar threshold=0.5 (padrão), encontra o threshold que minimiza:
    cost_total = (FN * cost_fn) + (FP_standard * cost_fp_standard) + (FP_highvalue * cost_fp_highvalue)
    """
    
    def __init__(self, cost_matrix: CostMatrix):
        """
        Args:
            cost_matrix: Instância de CostMatrix
        """
        self.cost_matrix = cost_matrix
        self.optimal_threshold = None
        self.is_fitted = False
    
    def find_optimal_threshold(
        self,
        y_true: np.array,
        y_pred_proba: np.array,
        is_high_value: np.array,
        thresholds: np.array = np.arange(0.01, 1.0, 0.01)
    ) -> Tuple[float, float]:
        """
        Encontra o threshold que minimiza custo total.
        
        Args:
            y_true: Labels reais
            y_pred_proba: Probabilidades preditas (0-1)
            is_high_value: Indicador de cliente high-value
            thresholds: Array de thresholds para testar
        
        Returns:
            Tuple(optimal_threshold, minimum_cost)
        """
        min_cost = float('inf')
        optimal_threshold = 0.5
        
        costs_per_threshold = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cost = self.cost_matrix.total_cost(y_pred, y_true, is_high_value)
            costs_per_threshold.append((threshold, cost))
            
            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold
        
        self.optimal_threshold = optimal_threshold
        self.is_fitted = True
        
        logger.info(f"Threshold ótimo encontrado: {optimal_threshold:.3f}")
        logger.info(f"Custo total mínimo: ${min_cost:.2f}")
        
        costs_df = pd.DataFrame(costs_per_threshold, columns=['Threshold', 'Cost'])
        
        return optimal_threshold, min_cost
    
    def apply_threshold(self, y_pred_proba: np.array) -> np.array:
        """
        Aplica threshold ótimo para gerar predições binárias.
        
        Args:
            y_pred_proba: Probabilidades preditas
        
        Returns:
            Predições binárias
        """
        if self.optimal_threshold is None:
            logger.warning("Threshold ótimo não foi calculado. Usando 0.5.")
            threshold = 0.5
        else:
            threshold = self.optimal_threshold
        
        return (y_pred_proba >= threshold).astype(int)


class CostSensitiveEvaluator:
    """
    Avalia um modelo considerando a matriz de custo.
    """
    
    def __init__(self, cost_matrix: CostMatrix):
        """
        Args:
            cost_matrix: Instância de CostMatrix
        """
        self.cost_matrix = cost_matrix
    
    def evaluate(
        self,
        y_true: np.array,
        y_pred: np.array,
        y_pred_proba: Optional[np.array] = None,
        is_high_value: Optional[np.array] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Avalia modelo com métricas considerando custo.
        
        Args:
            y_true: Labels reais
            y_pred: Predições binárias (pode ser gerado de y_pred_proba e threshold)
            y_pred_proba: Probabilidades (opcional, para cost curve)
            is_high_value: Indicador high-value (opcional)
            threshold: Threshold usado para predições
        
        Returns:
            Dict com métricas
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        # Ajustar predições se temos probabilidades e threshold
        if y_pred_proba is not None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Se não temos high-value, assumir todos padrão
        if is_high_value is None:
            is_high_value = np.zeros(len(y_true), dtype=int)
        
        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Métricas convencionais
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_proba if y_pred_proba is not None else y_pred)
        else:
            auc = np.nan
        
        # Custo total
        total_cost = self.cost_matrix.total_cost(y_pred, y_true, is_high_value)
        
        # Breakdown de custos
        fn_cost = fn * self.cost_matrix.cost_fn
        fp_standard_cost = (fp * (is_high_value == 0)).sum() * self.cost_matrix.cost_fp_standard
        fp_highvalue_cost = (fp * (is_high_value == 1)).sum() * self.cost_matrix.cost_fp_highvalue
        
        results = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'total_cost': total_cost,
            'fn_cost': fn_cost,
            'fp_standard_cost': fp_standard_cost,
            'fp_highvalue_cost': fp_highvalue_cost,
        }
        
        return results
    
    def report(
        self,
        y_true: np.array,
        y_pred: np.array,
        y_pred_proba: Optional[np.array] = None,
        is_high_value: Optional[np.array] = None,
        threshold: float = 0.5,
        print_output: bool = True
    ) -> Dict[str, float]:
        """
        Gera relatório completo de avaliação com custo.
        """
        results = self.evaluate(y_true, y_pred, y_pred_proba, is_high_value, threshold)
        
        if print_output:
            print("\n" + "="*80)
            print("AVALIACAO COM MATRIZ DE CUSTO")
            print("="*80)
            
            print(f"\nThreshold: {threshold:.3f}")
            print(f"\nMetricas Convencionais:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1']:.4f}")
            print(f"  AUC-ROC: {results['auc_roc']:.4f}")
            
            print(f"\nMatriz de Confusao:")
            print(f"  TN: {results['tn']} | FP: {results['fp']}")
            print(f"  FN: {results['fn']} | TP: {results['tp']}")
            
            print(f"\nCusto Total: ${results['total_cost']:.2f}")
            print(f"  FN Cost (fraude nao detectada): ${results['fn_cost']:.2f}")
            print(f"  FP Standard Cost (cliente padrao bloqueado): ${results['fp_standard_cost']:.2f}")
            print(f"  FP High-Value Cost (cliente rico bloqueado): ${results['fp_highvalue_cost']:.2f}")
            
            print("="*80)
        
        return results


# ==================== FUNCOES AUXILIARES ====================

def create_cost_sensitive_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    cost_matrix: CostMatrix,
    client_value_classifier: ClientValueClassifier
) -> Tuple[pd.DataFrame, pd.Series, np.array]:
    """
    Prepara dados de treino com sample weights baseado em custo.
    
    Args:
        X: Features
        y: Target
        cost_matrix: Matriz de custo
        client_value_classifier: Classificador de valor do cliente
    
    Returns:
        Tuple(X, y, sample_weights)
    """
    # Classificar clientes
    is_high_value = client_value_classifier.classify(X)
    
    # Gerar sample weights
    sample_weights = cost_matrix.get_sample_weights(y.values, is_high_value.values)
    
    return X, y, sample_weights
