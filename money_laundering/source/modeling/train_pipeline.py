"""
Script de Treinamento de Modelos para Detecção de Lavagem de Dinheiro

Este script implementa um pipeline completo de treinamento com:
- Carregamento de dados usando caminhos relativos (pathlib)
- Pré-processamento sem data leakage
- Balanceamento via Random Under Sampling (RUS)
- Treinamento de múltiplos algoritmos
- Avaliação em conjunto OOT (Out-of-Time)

Autor: TCC - Anti Money Laundering Detection
Data: Janeiro 2026
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from loguru import logger
import joblib
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score,
    accuracy_score, precision_score, recall_score, average_precision_score,
    fbeta_score
)

# Imbalanced-Learn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost e LightGBM
import xgboost as xgb
import lightgbm as lgb

# Módulos do projeto
import sys
sys.path.append(str(Path(__file__).parent.parent))

from source.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT
from source.preprocessing import AMLPreprocessor


class AMLModelTrainer:
    """
    Classe para treinamento de modelos de detecção de lavagem de dinheiro.
    
    Implementa:
    - Pipeline com pré-processamento + RUS + modelo
    - Treinamento de múltiplos algoritmos
    - Avaliação robusta
    - Persistência de modelos e métricas
    """
    
    def __init__(
        self,
        preprocessor: AMLPreprocessor,
        target_col: str = 'Is Laundering',
        random_state: int = 42,
        rus_neg_pos_ratio: float = 10.0,
        pos_weight_multiplier: float = 0.1,
        beta: float = 0.5
    ):
        """
        Inicializa o treinador.
        
        Args:
            preprocessor: Preprocessador ajustado
            target_col: Nome da coluna target
            random_state: Seed para reprodutibilidade
        """
        self.preprocessor = preprocessor
        self.target_col = target_col
        self.random_state = random_state
        self.rus_neg_pos_ratio = rus_neg_pos_ratio
        self.pos_weight_multiplier = pos_weight_multiplier
        self.beta = beta
        self.models_ = {}
        self.results_ = {}

    @staticmethod
    def _calc_fpr(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute false positive rate from confusion matrix."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    @staticmethod
    def _compute_scale_pos_weight(y: pd.Series, multiplier: float = 0.1) -> float:
        """Compute reduced class weight for positive class."""
        y_series = pd.Series(y)
        positive = int((y_series == 1).sum())
        negative = int((y_series == 0).sum())

        if positive == 0:
            return 1.0

        base_weight = negative / positive
        weighted = base_weight * multiplier
        return float(max(weighted, 1e-6))

    def _get_rus_sampling_strategy(self) -> float:
        """Convert neg:pos ratio (e.g. 10) to imblearn sampling_strategy float."""
        if self.rus_neg_pos_ratio <= 0:
            return 1.0
        return float(1.0 / self.rus_neg_pos_ratio)

    def _set_cost_sensitive_weights(self, model, y_train: pd.Series):
        """Apply scale_pos_weight to XGBoost/LightGBM models."""
        weight = self._compute_scale_pos_weight(y_train, multiplier=self.pos_weight_multiplier)
        model_name = model.__class__.__name__.lower()

        if 'xgb' in model_name or 'xgboost' in model_name:
            model.set_params(scale_pos_weight=weight)
        if 'lgbm' in model_name or 'lightgbm' in model_name:
            model.set_params(scale_pos_weight=weight)

        return model, weight

    def optimize_threshold_fbeta(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        beta: Optional[float] = None,
        threshold_min: float = 0.01,
        threshold_max: float = 0.99,
        threshold_step: float = 0.01,
    ) -> Dict[str, Any]:
        """Find threshold that maximizes F-beta score (default beta=0.5)."""
        beta = self.beta if beta is None else beta
        thresholds = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)

        rows = []
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            fpr = self._calc_fpr(y_true, y_pred)
            rows.append(
                {
                    'threshold': float(thr),
                    'precision': float(precision),
                    'recall': float(recall),
                    'fpr': float(fpr),
                    'f_beta': float(fbeta),
                }
            )

        table = pd.DataFrame(rows)
        best_idx = table['f_beta'].idxmax()
        best_row = table.loc[best_idx].to_dict()

        return {
            'beta': float(beta),
            'best_threshold': float(best_row['threshold']),
            'best_f_beta': float(best_row['f_beta']),
            'table': table,
            'best_row': best_row,
        }

    @staticmethod
    def plot_precision_recall_vs_threshold(
        threshold_table: pd.DataFrame,
        output_path: Path,
        precision_cut: float = 0.90,
    ) -> None:
        """Save chart Precision/Recall/F-beta vs threshold for manual cut selection."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(threshold_table['threshold'], threshold_table['precision'], label='Precision', linewidth=2)
        ax.plot(threshold_table['threshold'], threshold_table['recall'], label='Recall', linewidth=2)
        ax.plot(threshold_table['threshold'], threshold_table['f_beta'], label='F-beta', linewidth=2)
        ax.axhline(precision_cut, color='red', linestyle='--', linewidth=1.5, label=f'Precision Cut {precision_cut:.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision-Recall-Fbeta vs Threshold')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _create_model_pipeline(self, model, use_rus: bool = True):
        """
        Cria pipeline com RUS + modelo.
        
        Args:
            model: Modelo sklearn-compatible
            use_rus: Se True, aplica Random Under Sampling
        
        Returns:
            Pipeline do imblearn
        """
        steps = []
        
        if use_rus:
            # Random Under Sampling
            rus = RandomUnderSampler(
                sampling_strategy=self._get_rus_sampling_strategy(),
                random_state=self.random_state
            )
            steps.append(('rus', rus))
        
        steps.append(('model', model))
        
        return ImbPipeline(steps=steps)
    
    def get_model_configs(self, y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Retorna configurações dos modelos a serem treinados.
        
        Returns:
            Dict com nome -> modelo configurado
        """
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }

        if y_train is not None:
            for model_name in ['XGBoost', 'LightGBM']:
                if model_name in models:
                    weighted_model, weight = self._set_cost_sensitive_weights(models[model_name], y_train)
                    models[model_name] = weighted_model
                    logger.info(
                        f"{model_name} scale_pos_weight ajustado para {weight:.4f} "
                        f"(multiplier={self.pos_weight_multiplier})"
                    )

        return models
    
    def train_model(
        self,
        model_name: str,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_rus: bool = True
    ):
        """
        Treina um modelo individual.
        
        Args:
            model_name: Nome do modelo
            model: Instância do modelo
            X_train: Features de treino
            y_train: Target de treino
            use_rus: Se True, aplica RUS
        
        Returns:
            Pipeline treinado
        """
        logger.info(f"Treinando {model_name}...")
        
        # Criar pipeline
        pipeline = self._create_model_pipeline(model, use_rus=use_rus)
        
        # Treinar
        pipeline.fit(X_train, y_train)
        
        # Armazenar
        self.models_[model_name] = pipeline
        
        logger.success(f"{model_name} treinado com sucesso!")
        
        return pipeline
    
    def evaluate_model(
        self,
        model_name: str,
        pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test',
        optimize_threshold: bool = False
    ) -> Dict[str, float]:
        """
        Avalia um modelo treinado.
        
        Args:
            model_name: Nome do modelo
            pipeline: Pipeline treinado
            X: Features
            y: Target real
            dataset_name: Nome do dataset (train/oot)
        
        Returns:
            Dict com métricas
        """
        # Predições
        y_proba = pipeline.predict_proba(X)[:, 1]
        y_pred = pipeline.predict(X)

        threshold_used = 0.5
        threshold_fbeta = None

        if optimize_threshold:
            thr_result = self.optimize_threshold_fbeta(y, y_proba, beta=self.beta)
            threshold_used = thr_result['best_threshold']
            threshold_fbeta = thr_result['best_f_beta']
            y_pred = (y_proba >= threshold_used).astype(int)
        
        # Métricas
        metrics = {
            'model': model_name,
            'dataset': dataset_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'avg_precision': average_precision_score(y, y_proba),
            'fpr': self._calc_fpr(y, y_pred),
            'threshold': threshold_used,
        }

        if threshold_fbeta is not None:
            metrics[f'f_beta_{self.beta}'] = float(threshold_fbeta)
        
        # Armazenar resultados
        key = f"{model_name}_{dataset_name}"
        self.results_[key] = metrics
        
        # Log
        logger.info(f"\n{model_name} - {dataset_name.upper()}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Prec:  {metrics['avg_precision']:.4f}")
        logger.info(f"  FPR:       {metrics['fpr']:.4f}")
        logger.info(f"  Threshold: {metrics['threshold']:.4f}")
        
        return metrics
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_oot: pd.DataFrame,
        y_oot: pd.Series,
        use_rus: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Treina todos os modelos e avalia em treino e OOT.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_oot: Features OOT
            y_oot: Target OOT
            use_rus: Se True, aplica RUS
        
        Returns:
            Dict com todos os resultados
        """
        model_configs = self.get_model_configs(y_train=y_train)
        all_results = {}
        
        for model_name, model in model_configs.items():
            try:
                # Treinar
                pipeline = self.train_model(
                    model_name, model, X_train, y_train, use_rus=use_rus
                )
                
                # Avaliar em treino
                train_metrics = self.evaluate_model(
                    model_name, pipeline, X_train, y_train, 'train', optimize_threshold=False
                )
                
                # Avaliar em OOT
                oot_metrics = self.evaluate_model(
                    model_name, pipeline, X_oot, y_oot, 'oot', optimize_threshold=True
                )
                
                all_results[model_name] = {
                    'train': train_metrics,
                    'oot': oot_metrics
                }
                
            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {str(e)}")
                continue
        
        return all_results
    
    def save_results(self, output_dir: Path) -> None:
        """
        Salva resultados em CSV e JSON.
        
        Args:
            output_dir: Diretório de saída
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Converter resultados para DataFrame
        results_list = []
        for key, metrics in self.results_.items():
            results_list.append(metrics)
        
        df_results = pd.DataFrame(results_list)
        
        # Salvar CSVs separados
        df_train = df_results[df_results['dataset'] == 'train']
        df_oot = df_results[df_results['dataset'] == 'oot']
        
        train_path = output_dir / 'model_results_train.csv'
        oot_path = output_dir / 'model_results_oot.csv'
        
        df_train.to_csv(train_path, index=False)
        df_oot.to_csv(oot_path, index=False)
        
        logger.success(f"Resultados salvos em {output_dir}")
        logger.info(f"  - {train_path.name}")
        logger.info(f"  - {oot_path.name}")
    
    def save_models(self, output_dir: Path) -> None:
        """
        Salva todos os modelos treinados.
        
        Args:
            output_dir: Diretório de saída
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, pipeline in self.models_.items():
            # Nome de arquivo seguro
            safe_name = model_name.lower().replace(' ', '_')
            filepath = output_dir / f'{safe_name}.pkl'
            
            joblib.dump(pipeline, filepath)
            logger.info(f"  - {model_name} salvo em {filepath.name}")
        
        logger.success(f"Modelos salvos em {output_dir}")
    
    def save_training_info(self, output_dir: Path, **kwargs) -> None:
        """
        Salva informações sobre o treinamento.
        
        Args:
            output_dir: Diretório de saída
            **kwargs: Informações adicionais
        """
        output_dir = Path(output_dir)
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'target_col': self.target_col,
            'random_state': self.random_state,
            'models_trained': list(self.models_.keys()),
            'n_features': len(self.preprocessor.feature_names_),
            'feature_names': self.preprocessor.feature_names_,
            **kwargs
        }
        
        filepath = output_dir / 'training_info.json'
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.success(f"Informações de treinamento salvas em {filepath}")


def suggest_false_alarm_features_from_shap(
    shap_values: np.ndarray,
    X_eval: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Suggest features to review/removal based on SHAP impact concentrated in false positives.

    Strategy:
    - Compare mean |SHAP| in False Positives vs True Negatives.
    - Rank features by excess impact on false positives.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)
    tn_mask = (y_true_arr == 0) & (y_pred_arr == 0)

    if fp_mask.sum() == 0:
        return pd.DataFrame(
            {
                'feature': [],
                'mean_abs_shap_fp': [],
                'mean_abs_shap_tn': [],
                'fp_alarm_excess': [],
            }
        )

    shap_df = pd.DataFrame(np.abs(shap_values), columns=X_eval.columns)

    fp_importance = shap_df.loc[fp_mask].mean(axis=0)
    tn_importance = shap_df.loc[tn_mask].mean(axis=0) if tn_mask.sum() > 0 else pd.Series(0.0, index=shap_df.columns)

    report = pd.DataFrame(
        {
            'feature': shap_df.columns,
            'mean_abs_shap_fp': fp_importance.values,
            'mean_abs_shap_tn': tn_importance.values,
        }
    )
    report['fp_alarm_excess'] = report['mean_abs_shap_fp'] - report['mean_abs_shap_tn']
    report = report.sort_values('fp_alarm_excess', ascending=False).head(top_n).reset_index(drop=True)

    return report


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Carrega dados de treino e OOT.
    
    Args:
        data_dir: Diretório com os dados processados
    
    Returns:
        Tuple (X_train, y_train, X_oot, y_oot)
    """
    data_dir = Path(data_dir)
    
    logger.info("Carregando dados...")
    
    # Verificar se existem dados já processados (X, y separados)
    x_train_path = data_dir / 'X_train.csv'
    y_train_path = data_dir / 'y_train.csv'
    x_oot_path = data_dir / 'X_oot.csv'
    y_oot_path = data_dir / 'y_oot.csv'
    
    if all(p.exists() for p in [x_train_path, y_train_path, x_oot_path, y_oot_path]):
        logger.info("Carregando dados pré-processados (X, y)...")
        
        X_train = pd.read_csv(x_train_path, index_col=0)
        y_train_df = pd.read_csv(y_train_path)
        y_col_train = [col for col in y_train_df.columns if col not in ['Unnamed: 0', 'index']][0]
        y_train = y_train_df[y_col_train]
        
        X_oot = pd.read_csv(x_oot_path, index_col=0)
        y_oot_df = pd.read_csv(y_oot_path)
        y_col_oot = [col for col in y_oot_df.columns if col not in ['Unnamed: 0', 'index']][0]
        y_oot = y_oot_df[y_col_oot]
        
    else:
        # Carregar dados completos
        logger.info("Carregando dados completos (df_treino, df_oot)...")
        
        treino_path = data_dir / 'df_treino.csv'
        oot_path = data_dir / 'df_oot.csv'
        
        if not treino_path.exists() or not oot_path.exists():
            raise FileNotFoundError(
                f"Dados não encontrados em {data_dir}. "
                "Execute o notebook de preparação de dados primeiro."
            )
        
        df_treino = pd.read_csv(treino_path)
        df_oot = pd.read_csv(oot_path)
        
        # Identificar coluna target
        target_col = 'Is Laundering'
        if target_col not in df_treino.columns:
            # Tentar identificar automaticamente
            for col in ['is_laundering', 'flag', 'target', 'label']:
                if col in df_treino.columns:
                    target_col = col
                    break
        
        # Separar X e y
        X_train = df_treino.drop(columns=[target_col])
        y_train = df_treino[target_col]
        X_oot = df_oot.drop(columns=[target_col])
        y_oot = df_oot[target_col]
    
    logger.success("Dados carregados com sucesso!")
    logger.info(f"  Treino: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  OOT:    X={X_oot.shape}, y={y_oot.shape}")
    logger.info(f"  Distribuição Treino: {y_train.value_counts().to_dict()}")
    logger.info(f"  Distribuição OOT:    {y_oot.value_counts().to_dict()}")
    
    return X_train, y_train, X_oot, y_oot


def main(
    data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    use_rus: bool = True,
    random_state: int = 42,
    rus_neg_pos_ratio: float = 10.0,
    pos_weight_multiplier: float = 0.1,
    beta: float = 0.5,
) -> None:
    """
    Função principal de treinamento.
    
    Args:
        data_dir: Diretório com dados processados
        models_dir: Diretório para salvar modelos
        use_rus: Se True, aplica Random Under Sampling
        random_state: Seed para reprodutibilidade
    """
    logger.info("="*80)
    logger.info("TREINAMENTO DE MODELOS - DETECÇÃO DE LAVAGEM DE DINHEIRO")
    logger.info("="*80)
    
    # 1. Carregar dados
    X_train_raw, y_train, X_oot_raw, y_oot = load_data(data_dir)
    
    # 2. Configurar preprocessador
    logger.info("\nConfigurando preprocessador...")
    
    # Configurações específicas baseadas nos notebooks
    preprocessor = AMLPreprocessor(
        target_col='Is Laundering',
        onehot_cols=['Payment Format'],
        target_encoding_cols=['Receiving Currency', 'Payment Currency'],
        frequency_cols=[
            'Timestamp', 'From Account', 'To Account',
            'From Bank Name', 'Account Number', 'From Entity ID', 'From Entity Name',
            'To Bank Name', 'Account Number_To', 'To Entity ID', 'To Entity Name'
        ],
        transform_cols={
            'Amount Paid': 'yeojohnson',
            'Amount Received': 'yeojohnson',
            'Bank ID': 'yeojohnson',
            'Bank ID_To': 'yeojohnson',
            'From Bank': 'yeojohnson',
            'To Bank': 'yeojohnson'
        }
    )
    
    # 3. Fit preprocessador (apenas no treino!)
    logger.info("\nAjustando preprocessador no conjunto de TREINO...")
    preprocessor.fit(X_train_raw, y_train)
    
    # 4. Transform treino e OOT
    logger.info("\nTransformando dados...")
    X_train = preprocessor.transform(X_train_raw)
    X_oot = preprocessor.transform(X_oot_raw)
    
    logger.success(f"Transformação concluída!")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_oot:   {X_oot.shape}")
    
    # 5. Salvar preprocessador
    preprocessor_path = models_dir / 'preprocessor.pkl'
    preprocessor.save(preprocessor_path)
    
    # 6. Treinar modelos
    logger.info("\n" + "="*80)
    logger.info("TREINAMENTO DE MODELOS")
    logger.info("="*80)
    
    trainer = AMLModelTrainer(
        preprocessor=preprocessor,
        target_col='Is Laundering',
        random_state=random_state,
        rus_neg_pos_ratio=rus_neg_pos_ratio,
        pos_weight_multiplier=pos_weight_multiplier,
        beta=beta,
    )
    
    results = trainer.train_all_models(
        X_train, y_train, X_oot, y_oot, use_rus=use_rus
    )
    
    # 7. Salvar resultados
    logger.info("\n" + "="*80)
    logger.info("SALVANDO RESULTADOS")
    logger.info("="*80)
    
    trainer.save_results(data_dir)
    trainer.save_models(models_dir)
    trainer.save_training_info(
        models_dir,
        use_rus=use_rus,
        rus_neg_pos_ratio=rus_neg_pos_ratio,
        pos_weight_multiplier=pos_weight_multiplier,
        beta=beta,
        n_train_samples=len(X_train),
        n_oot_samples=len(X_oot)
    )
    
    # 8. Resumo final
    logger.info("\n" + "="*80)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("="*80)
    
    # Melhor modelo por Average Precision no OOT (métrica principal para classes desbalanceadas)
    df_results_oot = pd.DataFrame([
        trainer.results_[key] for key in trainer.results_ 
        if key.endswith('_oot')
    ])
    
    best_model_idx = df_results_oot['avg_precision'].idxmax()
    best_model_name = df_results_oot.loc[best_model_idx, 'model']
    best_avg_precision = df_results_oot.loc[best_model_idx, 'avg_precision']
    
    logger.success(f"\nMelhor modelo (PR-AUC/Avg Precision OOT): {best_model_name} ({best_avg_precision:.4f})")
    
    logger.info(f"\nArquivos gerados:")
    logger.info(f"  - Preprocessador: {preprocessor_path}")
    logger.info(f"  - Modelos: {models_dir}/*.pkl")
    logger.info(f"  - Resultados: {data_dir}/model_results_*.csv")


if __name__ == '__main__':
    # Usar caminhos relativos à raiz do projeto
    main(
        data_dir=PROCESSED_DATA_DIR,
        models_dir=MODELS_DIR,
        use_rus=True,
        random_state=42,
        rus_neg_pos_ratio=10.0,
        pos_weight_multiplier=0.1,
        beta=0.5,
    )
