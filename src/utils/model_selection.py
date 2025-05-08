"""
Model selection utilities for cross-validation and evaluation.
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Generator, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class CrossValidator:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        results_dir: str = None
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            results_dir: Directory to save fold results
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
        if results_dir is None:
            results_dir = Path('results') / 'cross_validation'
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results for each fold
        self.fold_results = []
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation splits.
        
        Args:
            X: Features or texts
            y: Labels
        
        Yields:
            Tuple of (X_train, X_val, y_train, y_val) for each fold
        """
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            
            # Get train/val splits
            if isinstance(X, np.ndarray):
                X_train, X_val = X[train_idx], X[val_idx]
            elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
            
            if isinstance(y, np.ndarray):
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                y_train, y_val = [y[i] for i in train_idx], [y[i] for i in val_idx]
            
            # Log split info
            if isinstance(y, np.ndarray):
                train_dist = np.bincount(y_train) / len(y_train)
                val_dist = np.bincount(y_val) / len(y_val)
            else:
                train_dist = pd.Series(y_train).value_counts(normalize=True)
                val_dist = pd.Series(y_val).value_counts(normalize=True)
            
            logger.info(f"Training set distribution: {dict(zip(range(len(train_dist)), train_dist))}")
            logger.info(f"Validation set distribution: {dict(zip(range(len(val_dist)), val_dist))}")
            
            yield X_train, X_val, y_train, y_val
    
    def add_fold_result(
        self,
        fold: int,
        metrics: Dict[str, float],
        model_name: str,
        **additional_info: Any
    ) -> None:
        """
        Add results for a fold.
        
        Args:
            fold: Fold number (0-based)
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model
            **additional_info: Additional information to store
        """
        result = {
            'fold': fold + 1,
            'model': model_name,
            'metrics': metrics,
            **additional_info
        }
        self.fold_results.append(result)
        
        # Save results
        results_file = self.results_dir / f'{model_name}_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.fold_results, f, indent=2)
        
        logger.info(f"Saved fold {fold + 1} results to {results_file}")
    
    def get_average_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate average metrics across folds.
        
        Returns:
            Dictionary with mean and std of each metric
        """
        if not self.fold_results:
            return {}
        
        # Get all metric names
        metric_names = set()
        for result in self.fold_results:
            metric_names.update(result['metrics'].keys())
        
        # Calculate statistics
        metrics_stats = {}
        for metric in metric_names:
            values = [r['metrics'][metric] for r in self.fold_results if metric in r['metrics']]
            metrics_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return metrics_stats
    
    def save_summary(self, model_name: str) -> None:
        """
        Save summary of cross-validation results.
        
        Args:
            model_name: Name of the model
        """
        metrics_stats = self.get_average_metrics()
        
        summary = {
            'model': model_name,
            'n_splits': self.n_splits,
            'metrics': metrics_stats,
            'fold_results': self.fold_results
        }
        
        summary_file = self.results_dir / f'{model_name}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved cross-validation summary to {summary_file}")
        
        # Log summary
        logger.info("\nCross-validation Results Summary:")
        for metric, stats in metrics_stats.items():
            logger.info(f"{metric}:")
            logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]") 