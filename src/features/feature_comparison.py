"""
Feature engineering comparison module.

This module provides functionality to compare different feature engineering approaches:
1. Bag of Words (BoW) / TF-IDF
2. Word2Vec (with variations)
3. Transformer (BERT)
"""

import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from .feature_extraction import (
    BagOfWordsExtractor,
    Word2VecExtractor,
    TransformerExtractor
)

logger = logging.getLogger(__name__)

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
        
    def add_bow_extractor(
        self,
        name: str = 'bow_tfidf',
        max_features: int = 5000,
        min_df: int = 5,
        max_df: float = 0.95,
        use_tfidf: bool = True,
        ngram_range: tuple = (1, 2)
    ) -> None:
        """Add BoW/TF-IDF feature extractor."""
        self.feature_extractors[name] = BagOfWordsExtractor(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            use_tfidf=use_tfidf,
            ngram_range=ngram_range
        )
    
    def add_word2vec_extractors(
        self,
        vector_size: int = 100,
        windows: List[int] = [5, 10],
        min_counts: List[int] = [2, 5],
        architectures: List[int] = [0, 1]  # 0: CBOW, 1: Skip-gram
    ) -> None:
        """Add Word2Vec feature extractors with variations."""
        for window in windows:
            for min_count in min_counts:
                for sg in architectures:
                    name = f'w2v_win{window}_min{min_count}_{"sg" if sg else "cbow"}'
                    self.feature_extractors[name] = Word2VecExtractor(
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        sg=sg
                    )
    
    def add_transformer_extractor(
        self,
        name: str = 'bert',
        model_name: str = 'bert-base-uncased',
        max_length: int = 128
    ) -> None:
        """Add transformer feature extractor."""
        self.feature_extractors[name] = TransformerExtractor(
            model_name=model_name,
            max_length=max_length
        )
    
    def extract_features(self, texts: List[str]) -> None:
        """Extract features using all registered extractors."""
        for name, extractor in self.feature_extractors.items():
            logger.info(f"Extracting features using {name}...")
            start_time = time()
            
            # Extract features
            features = extractor.fit_transform(texts)
            
            # Store features and timing
            self.features[name] = features
            self.results[name] = {
                'extraction_time': time() - start_time,
                'feature_dim': features.shape[1]
            }
            
            logger.info(f"Extracted {features.shape[1]} features in {self.results[name]['extraction_time']:.2f} seconds")
    
    def evaluate_classifiers(
        self,
        labels: List[int],
        cv_folds: int = 5,
        save_prefix: str = None
    ) -> pd.DataFrame:
        """
        Evaluate features using logistic regression with cross-validation.
        
        Args:
            labels: Target labels
            cv_folds: Number of cross-validation folds
            save_prefix: Prefix for saving results
        """
        classifier = LogisticRegression(max_iter=1000)
        
        for name, features in self.features.items():
            logger.info(f"Evaluating {name}...")
            start_time = time()
            
            # Perform cross-validation
            scores = cross_val_score(
                classifier,
                features,
                labels,
                cv=cv_folds,
                scoring='f1_weighted'
            )
            
            # Store results
            self.results[name].update({
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'training_time': time() - start_time
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        if save_prefix:
            # Save results
            results_file = self.save_dir / f'{save_prefix}_results.csv'
            results_df.to_csv(results_file)
            logger.info(f"Saved results to {results_file}")
            
            # Create and save comparison plots
            self._plot_comparisons(save_prefix)
        
        return results_df
    
    def _plot_comparisons(self, save_prefix: str) -> None:
        """Create comparison plots."""
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Performance plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df.reset_index(),
            x='index',
            y='cv_mean',
            yerr=results_df['cv_std']
        )
        plt.xticks(rotation=45, ha='right')
        plt.title('Cross-validation Performance Comparison')
        plt.xlabel('Feature Extractor')
        plt.ylabel('F1 Score (weighted)')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_performance.png')
        plt.close()
        
        # Time comparison
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

def run_comparison(
    texts: List[str],
    labels: List[int],
    save_dir: str = None,
    save_prefix: str = 'comparison'
) -> pd.DataFrame:
    """
    Run complete feature engineering comparison.
    
    Args:
        texts: List of preprocessed texts
        labels: List of labels
        save_dir: Directory to save results
        save_prefix: Prefix for saved files
    """
    # Initialize comparison
    comparison = FeatureComparison(save_dir=save_dir)
    
    # Add feature extractors
    comparison.add_bow_extractor(
        name='tfidf',
        use_tfidf=True,
        ngram_range=(1, 2)
    )
    comparison.add_bow_extractor(
        name='bow',
        use_tfidf=False,
        ngram_range=(1, 1)
    )
    
    # Add Word2Vec variations
    comparison.add_word2vec_extractors(
        vector_size=100,
        windows=[5, 10],
        min_counts=[2, 5],
        architectures=[0, 1]  # CBOW and Skip-gram
    )
    
    # Add transformer
    comparison.add_transformer_extractor(
        name='bert',
        model_name='bert-base-uncased'
    )
    
    # Extract features and evaluate
    comparison.extract_features(texts)
    results = comparison.evaluate_classifiers(
        labels,
        cv_folds=5,
        save_prefix=save_prefix
    )
    
    return results 