"""
ETAPA 5: Métricas de Negócio e Estabilidade

Implementa métricas críticas para AML:
1. Precision@Top-K - Identifica O modelo é preciso ao ranquear top-K scores?
2. PSI (Population Stability Index) - Modelo degrada entre treino e OOT?

Autor: TCC - AML Detection
Data: 2026-03-13
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class BusinessMetrics:
    """
    Métricas de negócio e estabilidade para modelos AML.
    
    Implementa:
    - Precision @ Top-K (K=100, K=500)
    - PSI (Population Stability Index) para monitorar degradação
    """
    
    @staticmethod
    def precision_at_top_k(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        top_k_list: List[int] = None
    ) -> Dict[int, float]:
        """
        Calcula Precision @ Top-K.
        
        Define: Dos top-K scores mais altos, quantos são verdadeiramente positivos?
        
        Interesse de negócio:
        - Top-100: Investigadores investigam os 100 casos de maior suspeita
          → Quantos são realmente suspeitos? (Precision@100=%)
        - Top-500: Análise de padrões em 500 casos mais suspeitos
          → Quantos realmente precisam investigação?
        
        Args:
            y_true : np.ndarray
                Labels verdadeiros (0/1)
            y_scores : np.ndarray
                Scores de probabilidade do modelo [0, 1]
            top_k_list : List[int]
                Valores de K a calcular. Default: [100, 500]
        
        Returns:
            Dict[int, float]
                {K: Precision@K} para cada K
                
        Example:
            >>> y_true = np.array([1, 0, 1, 1, 0])
            >>> y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
            >>> metrics = BusinessMetrics.precision_at_top_k(y_true, y_scores, [1, 2])
            >>> metrics[1]  # Top-1: score 0.9 é label 1
            1.0
            >>> metrics[2]  # Top-2: scores [0.9, 0.8] têm labels [1, 0]
            0.5
        """
        if top_k_list is None:
            top_k_list = [100, 500]
        
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        
        if len(y_true) != len(y_scores):
            raise ValueError("y_true e y_scores devem ter mesmo tamanho")
        
        if len(y_true) == 0:
            return {k: 0.0 for k in top_k_list}
        
        # Ordenar por scores decrescentes (top scores primeiros)
        sorted_indices = np.argsort(-y_scores)
        y_true_sorted = y_true[sorted_indices]
        
        results = {}
        
        for k in top_k_list:
            # Top-K é min(k, total_samples) para evitar IndexError
            k_actual = min(k, len(y_true))
            
            # Top-K labels
            top_k_labels = y_true_sorted[:k_actual]
            
            # Precision@K = TP / K
            precision_at_k = np.sum(top_k_labels) / k_actual if k_actual > 0 else 0.0
            
            results[k] = precision_at_k
        
        return results
    
    @staticmethod
    def calculate_psi(
        train_scores: np.ndarray,
        oot_scores: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-10
    ) -> Dict[str, float]:
        """
        Calcula PSI (Population Stability Index) entre treino e OOT.
        
        Interesse de Negócio:
        - Valida se a distribuição de scores se mantém estável
        - PSI > 0.1: Degradação moderada (refit recomendado)
        - PSI > 0.25: Degradação severa (modelo está obsoleto)
        
        Fórmula PSI:
        PSI = Sum( (OOT_pct - Train_pct) * ln(OOT_pct / Train_pct) )
        
        Args:
            train_scores : np.ndarray
                Scores de probabilidade em dados de TREINO
            oot_scores : np.ndarray
                Scores de probabilidade em dados de OOT
            n_bins : int
                Número de bins para discretizar scores. Default=10
            epsilon : float
                Pequeno valor para evitar log(0). Default=1e-10
        
        Returns:
            Dict[str, float]
                {
                    'psi': valor_psi,
                    'psi_category': 'Estável'/'Moderada'/'Severa',
                    'train_mean': média treino,
                    'oot_mean': média OOT,
                    'train_std': desvio treino,
                    'oot_std': desvio OOT,
                    'divergence': Divergência Kullback-Leibler (alternativa ao PSI)
                }
        
        Example:
            >>> train = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            >>> oot = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
            >>> psi_result = BusinessMetrics.calculate_psi(train, oot, n_bins=5)
            >>> print(f"PSI: {psi_result['psi']:.4f}")
            >>> print(f"Categoria: {psi_result['psi_category']}")
        """
        train_scores = np.asarray(train_scores)
        oot_scores = np.asarray(oot_scores)
        
        if len(train_scores) == 0 or len(oot_scores) == 0:
            logger.warning("Dados vazios fornecidos para PSI")
            return {
                'psi': np.nan,
                'psi_category': 'Undetermined',
                'train_mean': np.nan,
                'oot_mean': np.nan,
                'train_std': np.nan,
                'oot_std': np.nan,
                'divergence': np.nan
            }
        
        # Estatísticas básicas
        train_mean = train_scores.mean()
        oot_mean = oot_scores.mean()
        train_std = train_scores.std()
        oot_std = oot_scores.std()
        
        # Definir bins com base nos dados de TREINO
        # PSI standard: usar percentis de treino como referência
        bin_edges = np.percentile(train_scores, np.linspace(0, 100, n_bins + 1))
        
        # Remover duplicatas em bin_edges (pode acontecer se dados sparsox)
        bin_edges = np.unique(bin_edges)
        
        # Garantir que min e max sejam inclusos
        bin_edges[0] = min(train_scores.min(), oot_scores.min()) - epsilon
        bin_edges[-1] = max(train_scores.max(), oot_scores.max()) + epsilon
        
        # Contar observações em cada bin
        # right=False: [a, b) bins (padrão PSI)
        train_counts, _ = np.histogram(train_scores, bins=bin_edges, range=None)
        oot_counts, _ = np.histogram(oot_scores, bins=bin_edges, range=None)
        
        # Converter para proporções (percentuais)
        train_pct = train_counts / train_counts.sum()
        oot_pct = oot_counts / oot_counts.sum()
        
        # Adicionar epsilon para evitar log(0)
        train_pct = np.clip(train_pct, epsilon, 1.0)
        oot_pct = np.clip(oot_pct, epsilon, 1.0)
        
        # Calcular PSI
        psi = np.sum((oot_pct - train_pct) * np.log(oot_pct / train_pct))
        
        # Categorizar PSI
        if psi < 0.1:
            psi_category = "Estavel"
        elif psi < 0.25:
            psi_category = "Moderada"
        else:
            psi_category = "Severa"
        
        # Calcular divergência KL como alternativa
        # KL(Train || OOT) = Sum( train_pct * ln(train_pct / oot_pct) )
        kl_divergence = np.sum(train_pct * np.log(train_pct / oot_pct))
        
        return {
            'psi': float(psi),
            'psi_category': psi_category,
            'train_mean': float(train_mean),
            'oot_mean': float(oot_mean),
            'train_std': float(train_std),
            'oot_std': float(oot_std),
            'divergence': float(kl_divergence)
        }
    
    @staticmethod
    def calculate_psi_by_feature(
        df_train: pd.DataFrame,
        df_oot: pd.DataFrame,
        score_col: str = 'model_score',
        n_bins: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula PSI para múltiplas features.
        
        Interesse de Negócio:
        - Identifica quais features estão mudando entre treino e OOT
        - Ajuda a diagnosticar degradação do modelo
        
        Args:
            df_train : pd.DataFrame
                DataFrame com scores de treino
            df_oot : pd.DataFrame
                DataFrame com scores de OOT
            score_col : str
                Coluna de scores para calcular PSI
            n_bins : int
                Número de bins
        
        Returns:
            Dict[str, Dict]
                PSI por feature
        
        Example:
            >>> df_train = pd.DataFrame({'score': [0.1, 0.2, ...], 'feature': [...]})
            >>> df_oot = pd.DataFrame({'score': [0.15, 0.25, ...], 'feature': [...]})
            >>> psi_by_feature = BusinessMetrics.calculate_psi_by_feature(
            ...     df_train, df_oot, 'score', n_bins=10
            ... )
        """
        if score_col not in df_train.columns or score_col not in df_oot.columns:
            raise ValueError(f"Coluna '{score_col}' não encontrada")
        
        train_scores = df_train[score_col].values
        oot_scores = df_oot[score_col].values
        
        return BusinessMetrics.calculate_psi(train_scores, oot_scores, n_bins)


class ModelEvaluationReport:
    """
    Classe para gerar relatórios completos de avaliação de modelos.
    
    Integra:
    - Métricas padrão (Precision, Recall, F1, ROC-AUC)
    - Precision@Top-K (Top-100, Top-500)
    - PSI (Population Stability Index)
    """
    
    def __init__(self):
        """Inicializa o relatório."""
        self.metrics_train_ = {}
        self.metrics_oot_ = {}
        self.precision_at_k_ = {}
        self.psi_ = {}
    
    def add_model_metrics(
        self,
        model_name: str,
        y_true_train: np.ndarray,
        y_scores_train: np.ndarray,
        y_true_oot: np.ndarray,
        y_scores_oot: np.ndarray,
        standard_metrics: Dict[str, float],
        top_k_list: List[int] = None
    ) -> None:
        """
        Adiciona métricas de um modelo ao relatório.
        
        Args:
            model_name : str
                Nome do modelo
            y_true_train : np.ndarray
                Labels reais treino
            y_scores_train : np.ndarray
                Scores de probabilidade treino
            y_true_oot : np.ndarray
                Labels reais OOT
            y_scores_oot : np.ndarray
                Scores de probabilidade OOT
            standard_metrics : Dict[str, float]
                Métricas padrão (Precision, Recall, etc)
            top_k_list : List[int]
                Valores de K para Precision@K. Default: [100, 500]
        """
        if top_k_list is None:
            top_k_list = [100, 500]
        
        # Precision @ Top-K
        precision_k_train = BusinessMetrics.precision_at_top_k(
            y_true_train, y_scores_train, top_k_list
        )
        precision_k_oot = BusinessMetrics.precision_at_top_k(
            y_true_oot, y_scores_oot, top_k_list
        )
        
        self.precision_at_k_[model_name] = {
            'train': precision_k_train,
            'oot': precision_k_oot
        }
        
        # PSI
        psi_result = BusinessMetrics.calculate_psi(y_scores_train, y_scores_oot)
        self.psi_[model_name] = psi_result
        
        # Standard metrics
        self.metrics_train_[model_name] = standard_metrics.get('train', {})
        self.metrics_oot_[model_name] = standard_metrics.get('oot', {})
    
    def generate_report(self) -> pd.DataFrame:
        """
        Gera relatório consolidado com todas as métricas.
        
        Returns:
            pd.DataFrame
                Relatório com colunas:
                - model
                - precision_at_100_train / oot
                - precision_at_500_train / oot
                - psi
                - psi_category
                - (outras métricas padrão)
        """
        reports = []
        
        for model_name in self.precision_at_k_.keys():
            report = {'model': model_name}
            
            # Precision @ K
            if model_name in self.precision_at_k_:
                pk = self.precision_at_k_[model_name]
                report['precision_at_100_train'] = pk['train'].get(100, np.nan)
                report['precision_at_100_oot'] = pk['oot'].get(100, np.nan)
                report['precision_at_500_train'] = pk['train'].get(500, np.nan)
                report['precision_at_500_oot'] = pk['oot'].get(500, np.nan)
            
            # PSI
            if model_name in self.psi_:
                psi = self.psi_[model_name]
                report['psi'] = psi['psi']
                report['psi_category'] = psi['psi_category']
                report['oot_mean_score'] = psi['oot_mean']
                report['train_mean_score'] = psi['train_mean']
            
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    def print_summary(self) -> None:
        """Imprime resumo visual das métricas."""
        logger.info("\n" + "="*100)
        logger.info("ETAPA 5: MÉTRICAS DE NEGÓCIO E ESTABILIDADE")
        logger.info("="*100)
        
        for model_name in self.precision_at_k_.keys():
            logger.info(f"\n{model_name}:")
            logger.info("-" * 100)
            
            # Precision @ Top-K
            pk = self.precision_at_k_[model_name]
            logger.info(f"  Precision @ Top-K:")
            logger.info(f"    Top-100 (Treino): {pk['train'].get(100, 0):.4f}")
            logger.info(f"    Top-100 (OOT):    {pk['oot'].get(100, 0):.4f}")
            logger.info(f"    Top-500 (Treino): {pk['train'].get(500, 0):.4f}")
            logger.info(f"    Top-500 (OOT):    {pk['oot'].get(500, 0):.4f}")
            
            # PSI
            psi = self.psi_[model_name]
            logger.info(f"\n  PSI (Population Stability Index):")
            logger.info(f"    PSI Value:      {psi['psi']:.4f}")
            logger.info(f"    Status:         {psi['psi_category']}")
            logger.info(f"    Train Mean:     {psi['train_mean']:.4f}")
            logger.info(f"    OOT Mean:       {psi['oot_mean']:.4f}")
            logger.info(f"    KL Divergence:  {psi['divergence']:.4f}")
            
            # Interpretação
            if psi['psi'] < 0.1:
                logger.success(f"    --> Modelo ESTÁVEL em OOT (PSI < 0.1)")
            elif psi['psi'] < 0.25:
                logger.warning(f"    --> Degradação MODERADA (0.1 <= PSI < 0.25). Refit recomendado em breve")
            else:
                logger.error(f"    --> Degradação SEVERA (PSI >= 0.25). REFIT CRÍTICO!")


def demonstrate_metrics():
    """Função de demonstração com dados sintéticos."""
    logger.info("\n" + "="*100)
    logger.info("DEMONSTRAÇÃO: ETAPA 5 - MÉTRICAS DE NEGÓCIO")
    logger.info("="*100)
    
    # Dados sintéticos
    np.random.seed(42)
    
    # Cenário: Modelo treinado em dados antigos, degradou em novos dados
    y_true_train = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 10)  # 100 samples
    y_scores_train = np.array([
        0.95, 0.05, 0.92, 0.88, 0.10, 0.91, 0.08, 0.12, 0.89, 0.15
    ] * 10)
    
    y_true_oot = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 10)
    y_scores_oot = np.array([
        0.92, 0.85, 0.45, 0.50, 0.82, 0.40, 0.78, 0.75, 0.35, 0.30
    ] * 10)
    
    # Precision @ Top-K
    logger.info("\n[TEST 1] Precision @ Top-K")
    logger.info("-" * 100)
    prec_train = BusinessMetrics.precision_at_top_k(y_true_train, y_scores_train, [10, 20, 50])
    prec_oot = BusinessMetrics.precision_at_top_k(y_true_oot, y_scores_oot, [10, 20, 50])
    
    for k in [10, 20, 50]:
        logger.info(f"  Precision @ Top-{k}:")
        logger.info(f"    Treino: {prec_train[k]:.4f}")
        logger.info(f"    OOT:    {prec_oot[k]:.4f}")
        diff = prec_train[k] - prec_oot[k]
        if abs(diff) > 0.1:
            logger.warning(f"    Diferença: {diff:+.4f} (Degradação)")
    
    # PSI
    logger.info("\n[TEST 2] PSI (Population Stability Index)")
    logger.info("-" * 100)
    psi_result = BusinessMetrics.calculate_psi(y_scores_train, y_scores_oot, n_bins=10)
    
    logger.info(f"  PSI Value: {psi_result['psi']:.4f}")
    logger.info(f"  Status: {psi_result['psi_category']}")
    logger.info(f"  Train Mean Score: {psi_result['train_mean']:.4f}")
    logger.info(f"  OOT Mean Score: {psi_result['oot_mean']:.4f}")
    logger.info(f"  Train Std: {psi_result['train_std']:.4f}")
    logger.info(f"  OOT Std: {psi_result['oot_std']:.4f}")
    logger.info(f"  KL Divergence: {psi_result['divergence']:.4f}")
    
    # Interpretação
    if psi_result['psi'] < 0.1:
        logger.success(f"  --> ESTÁVEL")
    elif psi_result['psi'] < 0.25:
        logger.warning(f"  --> DEGRADAÇÃO MODERADA")
    else:
        logger.error(f"  --> DEGRADAÇÃO SEVERA")


if __name__ == "__main__":
    demonstrate_metrics()
