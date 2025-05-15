"""
Feature engineering comparison module.

This module provides functionality to compare different feature engineering approaches:
1. Bag of Words (BoW) / TF-IDF
2. Word2Vec (with variations)
3. Transformer (BERT)
"""

import logging
from typing import List, Dict, Any, Callable
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, make_scorer, 
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from features.feature_extraction import (
    BagOfWordsExtractor,
    Word2VecExtractor,
    TransformerExtractor,
    CombinedFeatureExtractor
)

logger = logging.getLogger(__name__)

def macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro', zero_division=0)

def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro', zero_division=0)

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def weighted_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted', zero_division=0)

def weighted_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)

def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

class FeatureComparison:
    """Compare different feature engineering approaches."""
    
    def __init__(self, save_dir: str = None):
        """
        Initialize feature comparison.
        
        Args:
            save_dir: Directory to save results and plots
        """
        if save_dir is None:
            save_dir = Path('results') / 'feature_comparison'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.feature_extractors = {}
        self.features = {}
        self.extraction_times = {}
        
        # Add class names for readability
        self.class_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        
        # Define scoring metrics
        self.scoring = {
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'macro_precision': make_scorer(macro_precision),
            'macro_recall': make_scorer(macro_recall),
            'macro_f1': make_scorer(macro_f1),
            'weighted_precision': make_scorer(weighted_precision),
            'weighted_recall': make_scorer(weighted_recall),
            'weighted_f1': make_scorer(weighted_f1)
        }
        
        # Add per-class metrics
        for class_idx in range(3):
            self.scoring.update({
                f'precision_{self.class_names[class_idx]}': make_scorer(precision_score, labels=[class_idx], average=None),
                f'recall_{self.class_names[class_idx]}': make_scorer(recall_score, labels=[class_idx], average=None),
                f'f1_{self.class_names[class_idx]}': make_scorer(f1_score, labels=[class_idx], average=None)
            })
        
    def add_bow_extractor(
        self,
        name: str = 'bow_tfidf',
        max_features: int = 10000,
        min_df: int = 3,
        max_df: float = 0.90,
        use_tfidf: bool = True,
        ngram_range: tuple = (1, 3),
        norm: str = 'l2',
        sublinear_tf: bool = True,
        stop_words: str = 'english'
    ) -> None:
        """Add BoW/TF-IDF feature extractor with improved settings."""
        self.feature_extractors[name] = BagOfWordsExtractor(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            use_tfidf=use_tfidf,
            ngram_range=ngram_range,
            norm=norm,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words
        )
    
    def add_word2vec_extractors(
        self,
        vector_size: int = 300,
        windows: List[int] = [8, 10],
        min_counts: List[int] = [2, 5],
        architectures: List[int] = [1],
        negative_samples: List[int] = [10, 15],
        epochs: List[int] = [20, 30],
        use_fasttext: bool = True
    ) -> None:
        """Add Word2Vec feature extractors with improved variations."""
        for window in windows:
            for min_count in min_counts:
                for sg in architectures:
                    for negative in negative_samples:
                        for epoch in epochs:
                            name = f'w2v_win{window}_min{min_count}_{"sg" if sg else "cbow"}_neg{negative}_ep{epoch}'
                            self.feature_extractors[name] = Word2VecExtractor(
                                vector_size=vector_size,
                                window=window,
                                min_count=min_count,
                                sg=sg,
                                negative=negative,
                                epochs=epoch,
                                use_fasttext=use_fasttext
                            )
    
    def add_transformer_extractor(
        self,
        name: str = 'finbert',
        model_name: str = 'ProsusAI/finbert',
        max_length: int = 256,
        pooling_strategies: List[str] = ['mean_pooling', 'cls', 'max_pooling']
    ) -> None:
        """Add transformer feature extractors with different pooling strategies."""
        for pooling in pooling_strategies:
            pool_name = f'{name}_{pooling}'
            self.feature_extractors[pool_name] = TransformerExtractor(
                model_name=model_name,
                max_length=max_length,
                pooling_strategy=pooling
            )
    
    def extract_features(self, texts: List[str]) -> None:
        """Extract features using all registered extractors."""
        for name, extractor in self.feature_extractors.items():
            logger.info(f"Extracting features using {name}...")
            start_time = time.time()
            
            # Extract features
            features = extractor.fit_transform(texts)
            
            # Store features and timing
            self.features[name] = features
            self.extraction_times[name] = time.time() - start_time
            
            logger.info(f"Extracted {features.shape[1]} features in {self.extraction_times[name]:.2f} seconds")
    
    def _plot_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray, method_name: str, save_prefix: str) -> None:
        """Plot confusion matrix for a method."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names.values(),
            yticklabels=self.class_names.values()
        )
        plt.title(f'Normalized Confusion Matrix - {method_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_confusion_matrix_{method_name}.png')
        plt.close()
    
    def evaluate_classifiers(
        self,
        labels: List[int],
        cv_folds: int = 5,
        save_prefix: str = None
    ) -> pd.DataFrame:
        """
        Evaluate all feature extractors using cross-validation.
        
        Args:
            labels: List of labels
            cv_folds: Number of cross-validation folds
            save_prefix: Prefix for saved files
        """
        self.results = {}
        self.class_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        
        # Dictionary to store predictions for confusion matrices
        all_predictions = {}
        all_true_labels = {}
        
        for name, features in self.features.items():
            logger.info(f"Evaluating {name}...")
            
            # Initialize classifier with class weights
            clf = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            
            # Record feature dimensionality
            self.results[name] = {
                'feature_dim': features.shape[1]
            }
            
            # Cross-validation with multiple metrics
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Initialize lists to store metrics
            metrics = {
                'balanced_accuracy': [],
                'macro_f1': [],
                'weighted_f1': [],
                'training_time': []
            }
            
            # Per-class metrics
            for class_name in self.class_names.values():
                metrics.update({
                    f'precision_{class_name}': [],
                    f'recall_{class_name}': [],
                    f'f1_{class_name}': []
                })
            
            # Lists to store predictions and true labels
            all_predictions[name] = []
            all_true_labels[name] = []
            
            # Perform cross-validation
            for train_idx, val_idx in cv.split(features, labels):
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # Time the training
                start_time = time.time()
                clf.fit(X_train, y_train)
                train_time = time.time() - start_time
                metrics['training_time'].append(train_time)
                
                # Make predictions
                y_pred = clf.predict(X_val)
                
                # Store predictions and true labels
                all_predictions[name].extend(y_pred)
                all_true_labels[name].extend(y_val)
                
                # Calculate metrics
                metrics['balanced_accuracy'].append(balanced_accuracy_score(y_val, y_pred))
                metrics['macro_f1'].append(f1_score(y_val, y_pred, average='macro'))
                metrics['weighted_f1'].append(f1_score(y_val, y_pred, average='weighted'))
                
                # Calculate per-class metrics
                for class_idx, class_name in self.class_names.items():
                    y_true_binary = (y_val == class_idx)
                    y_pred_binary = (y_pred == class_idx)
                    
                    metrics[f'precision_{class_name}'].append(
                        precision_score(y_true_binary, y_pred_binary)
                    )
                    metrics[f'recall_{class_name}'].append(
                        recall_score(y_true_binary, y_pred_binary)
                    )
                    metrics[f'f1_{class_name}'].append(
                        f1_score(y_true_binary, y_pred_binary)
                    )
            
            # Store mean and std of metrics
            for metric_name, values in metrics.items():
                if metric_name == 'training_time':
                    self.results[name][metric_name] = np.mean(values)
                else:
                    self.results[name][f'{metric_name}_mean'] = np.mean(values)
                    self.results[name][f'{metric_name}_std'] = np.std(values)
            
            # Store extraction time
            self.results[name]['extraction_time'] = self.extraction_times.get(name, 0)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        if save_prefix:
            # Save detailed results
            results_file = self.save_dir / f'{save_prefix}_results.csv'
            results_df.to_csv(results_file)
            logger.info(f"Saved results to {results_file}")
            
            # Get top 4 methods based on weighted F1 score
            top_methods = results_df.nlargest(4, 'weighted_f1_mean').index
            
            # Plot confusion matrices only for top methods
            for method in top_methods:
                self._plot_confusion_matrices(
                    np.array(all_true_labels[method]),
                    np.array(all_predictions[method]),
                    method,
                    save_prefix
                )
            
            # Create and save comparison plots
            self._plot_comparisons(save_prefix)
            self._plot_per_class_metrics(save_prefix)
        
        return results_df
    
    def _plot_comparisons(self, save_prefix: str) -> None:
        """Create comparison plots."""
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Performance plot - now with multiple metrics
        plt.figure(figsize=(15, 8))
        metrics = ['balanced_accuracy', 'macro_f1', 'weighted_f1']
        x = np.arange(len(results_df.index))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            plt.bar(
                x + i*width, 
                results_df[mean_col],
                width,
                yerr=results_df[std_col],
                label=metric.replace('_', ' ').title()
            )
        
        plt.xlabel('Feature Extractor')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x + width, results_df.index, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_performance.png')
        plt.close()
        
        # Time comparison - with error handling
        if 'extraction_time' in results_df.columns and 'training_time' in results_df.columns:
            plt.figure(figsize=(12, 6))
            times_df = results_df[['extraction_time', 'training_time']].copy()
            times_df.plot(kind='bar', stacked=True)
            plt.title('Time Comparison')
            plt.xlabel('Feature Extractor')
            plt.ylabel('Time (seconds)')
            plt.legend(['Feature Extraction', 'Model Training'])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.save_dir / f'{save_prefix}_timing.png')
            plt.close()
        
        # Feature dimensionality
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df.reset_index(),
            x='index',
            y='feature_dim'
        )
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Dimensionality Comparison')
        plt.xlabel('Feature Extractor')
        plt.ylabel('Number of Features')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_dimensions.png')
        plt.close()
    
    def _plot_per_class_metrics(self, save_prefix: str) -> None:
        """Create per-class performance comparison plots."""
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Plot per-class F1 scores
        plt.figure(figsize=(15, 6))
        x = np.arange(len(results_df.index))
        width = 0.25
        
        for i, class_name in enumerate(self.class_names.values()):
            plt.bar(
                x + i*width,
                results_df[f'f1_{class_name}_mean'],
                width,
                label=f'{class_name}'
            )
        
        plt.xlabel('Feature Extractor')
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Scores')
        plt.xticks(x + width, results_df.index, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_per_class_f1.png')
        plt.close()

def run_comparison(
    texts: List[str],
    labels: List[int],
    save_dir: str = None,
    save_prefix: str = 'comparison'
) -> pd.DataFrame:
    """Run complete feature engineering comparison."""
    if save_dir is None:
        save_dir = 'results/feature_comparison'
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    comparison = FeatureComparison(save_dir=save_dir)
    
    # Add improved BoW/TF-IDF extractors
    comparison.add_bow_extractor(
        name='tfidf',
        use_tfidf=True,
        max_features=10000,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 3),
        sublinear_tf=True,
        norm='l2',
        stop_words='english'
    )
    comparison.add_bow_extractor(
        name='bow',
        use_tfidf=False,
        max_features=10000,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 3),
        stop_words='english'
    )
    
    # Add improved Word2Vec variations
    comparison.add_word2vec_extractors(
        vector_size=300,
        windows=[5, 10],
        min_counts=[2, 5],
        architectures=[0, 1],  # Both CBOW and Skip-gram
        negative_samples=[10, 15],
        epochs=[20, 30],
        use_fasttext=True
    )
    
    # Add FinBERT with different configurations
    finbert_configs = [
        {
            'name': 'finbert_mean',
            'model_name': 'ProsusAI/finbert',
            'max_length': 256,
            'pooling_strategy': 'mean_pooling'
        },
        {
            'name': 'finbert_cls',
            'model_name': 'ProsusAI/finbert',
            'max_length': 256,
            'pooling_strategy': 'cls'
        },
        {
            'name': 'finbert_max',
            'model_name': 'ProsusAI/finbert',
            'max_length': 256,
            'pooling_strategy': 'max_pooling'
        }
    ]
    
    for config in finbert_configs:
        comparison.feature_extractors[config['name']] = TransformerExtractor(
            model_name=config['model_name'],
            max_length=config['max_length'],
            pooling_strategy=config['pooling_strategy']
        )
    
    # Add combined feature extractors
    bow_extractor = BagOfWordsExtractor(
        use_tfidf=False,
        max_features=10000,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 3)
    )
    tfidf_extractor = BagOfWordsExtractor(
        use_tfidf=True,
        max_features=10000,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    finbert_mean_extractor = TransformerExtractor(
        model_name='ProsusAI/finbert',
        max_length=256,
        pooling_strategy='mean_pooling'
    )
    finbert_cls_extractor = TransformerExtractor(
        model_name='ProsusAI/finbert',
        max_length=256,
        pooling_strategy='cls'
    )
    
    # Add combined extractors with different weights
    combined_extractors = {
        'bow_tfidf_combined': {
            'extractors': [bow_extractor, tfidf_extractor],
            'weights': [0.5, 0.5]
        },
        'bow_finbert_combined': {
            'extractors': [bow_extractor, finbert_mean_extractor],
            'weights': [0.6, 0.4]  # More weight to BOW as it performs better
        },
        'tfidf_finbert_combined': {
            'extractors': [tfidf_extractor, finbert_mean_extractor],
            'weights': [0.6, 0.4]
        },
        'bow_tfidf_finbert_mean': {
            'extractors': [bow_extractor, tfidf_extractor, finbert_mean_extractor],
            'weights': [0.4, 0.4, 0.2]
        },
        'bow_tfidf_finbert_cls': {
            'extractors': [bow_extractor, tfidf_extractor, finbert_cls_extractor],
            'weights': [0.4, 0.4, 0.2]
        }
    }
    
    for name, config in combined_extractors.items():
        comparison.feature_extractors[name] = CombinedFeatureExtractor(
            extractors=config['extractors'],
            weights=config['weights']
        )
    
    # Extract features and evaluate
    comparison.extract_features(texts)
    results = comparison.evaluate_classifiers(
        labels,
        cv_folds=5,
        save_prefix=save_prefix
    )
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting feature comparison analysis...")
    
    # Load datasets
    from utils.data_loader import load_datasets
    train_df, test_df = load_datasets()
    
    # Load preprocessor
    from preprocessing.text_processor import TextPreprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    logger.info("Preprocessing training texts...")
    texts = [preprocessor.preprocess(text) for text in train_df['text']]
    labels = train_df['label'].values
    
    # Run comparison
    logger.info("Running feature comparison...")
    results = run_comparison(
        texts=texts,
        labels=labels,
        save_dir='results/feature_comparison',
        save_prefix='feature_comparison'
    )
    
    # Updated results printing with per-class metrics
    logger.info("\nFeature Comparison Results:")
    logger.info("-" * 50)
    for method, metrics in results.iterrows():
        logger.info(f"\nMethod: {method}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy_mean']:.3f} ± {metrics['balanced_accuracy_std']:.3f}")
        logger.info(f"Macro F1: {metrics['macro_f1_mean']:.3f} ± {metrics['macro_f1_std']:.3f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1_mean']:.3f} ± {metrics['weighted_f1_std']:.3f}")
        
        # Print per-class metrics
        logger.info("\nPer-class Performance:")
        for class_idx in range(3):
            class_name = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}[class_idx]
            logger.info(f"{class_name}:")
            logger.info(f"  F1: {metrics[f'f1_{class_name}_mean']:.3f}")
            logger.info(f"  Precision: {metrics[f'precision_{class_name}_mean']:.3f}")
            logger.info(f"  Recall: {metrics[f'recall_{class_name}_mean']:.3f}")
        
        logger.info(f"\nFeature Dimension: {metrics['feature_dim']:,}")
        logger.info(f"Extraction Time: {metrics['extraction_time']:.2f}s")
        logger.info(f"Training Time: {metrics['training_time']:.2f}s")
    
    logger.info("\nResults and visualizations have been saved to: results/feature_comparison/") 