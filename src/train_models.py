"""
Script to train and evaluate classification models.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import json
from typing import Dict, Any, Tuple, List

from utils.data_loader import load_datasets
from preprocessing.text_processor import TextPreprocessor
from features.feature_extraction import create_feature_extractor
from models.classifiers import LSTMClassifier, TransformerClassifier, TorchClassifierWrapper
from utils.model_selection import CrossValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, output_dir: str = None, n_folds: int = 5):
        if output_dir is None:
            output_dir = Path('results/model_evaluation')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = TextPreprocessor()
        self.feature_extractors = {
            'bow': {'max_features': 5000, 'ngram_range': (1, 2)},
            'word2vec': {'vector_size': 100, 'window': 5},
            'transformer': {'model_name': 'bert-base-uncased', 'max_length': 128}
        }
        self.n_folds = n_folds
        self.cross_validator = CrossValidator(n_splits=n_folds)
    
    def prepare_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Prepare data with different feature extraction methods."""
        # Load data
        train_df, test_df = load_datasets()
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        processed_texts = [
            self.preprocessor.preprocess(text)
            for text in train_df['text']
        ]
        
        # Extract features
        features = {}
        for method, params in self.feature_extractors.items():
            logger.info(f"\nExtracting features using {method}...")
            extractor = create_feature_extractor(method=method, **params)
            features[method] = extractor.fit_transform(processed_texts)
        
        return features, train_df['label'].values
    
    def train_knn(self, X: np.ndarray, y: np.ndarray) -> KNeighborsClassifier:
        """Train KNN classifier."""
        logger.info("\nTraining KNN classifier...")
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        model.fit(X, y)
        return model
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray) -> TorchClassifierWrapper:
        """Train LSTM classifier."""
        logger.info("\nTraining LSTM classifier...")
        model = TorchClassifierWrapper(
            model=LSTMClassifier(
                input_size=X.shape[-1],
                hidden_size=128,
                num_layers=2
            ),
            num_epochs=10,
            batch_size=32
        )
        model.fit(X, y)
        return model
    
    def train_transformer(self, X: np.ndarray, y: np.ndarray) -> TorchClassifierWrapper:
        """Train Transformer classifier."""
        logger.info("\nTraining Transformer classifier...")
        model = TorchClassifierWrapper(
            model=TransformerClassifier(
                input_size=X.shape[-1],
                hidden_size=256,
                dropout=0.1
            ),
            num_epochs=10,
            batch_size=32
        )
        model.fit(X, y)
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        feature_type: str
    ) -> Dict[str, Any]:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Add feature type and model name to report
        report['feature_type'] = feature_type
        report['model_name'] = model_name
        
        return {
            'report': report,
            'confusion_matrix': conf_matrix
        }
    
    def plot_results(self, results: Dict[str, Dict[str, Any]], fold_metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot evaluation results including cross-validation metrics."""
        # Create subplots for each metric
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Bearish', 'Bullish', 'Neutral']
        
        # Calculate number of rows needed
        n_rows = len(metrics) + 1  # +1 for CV metrics
        
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=[metric.capitalize() for metric in metrics] + ['Cross-Validation Metrics'],
            vertical_spacing=0.1
        )
        
        # Plot per-class metrics
        for i, metric in enumerate(metrics, 1):
            for feature_type in self.feature_extractors.keys():
                for model_name in ['knn', 'lstm', 'transformer']:
                    key = f"{feature_type}_{model_name}"
                    if key in results:
                        values = [
                            results[key]['report'][str(j)][metric]
                            for j in range(3)
                        ]
                        fig.add_trace(
                            go.Bar(
                                name=f"{feature_type} - {model_name}",
                                x=classes,
                                y=values,
                                showlegend=(i == 1)
                            ),
                            row=i,
                            col=1
                        )
        
        # Plot cross-validation metrics
        cv_metrics = ['accuracy', 'f1-score']
        for i, metric in enumerate(cv_metrics):
            values = []
            labels = []
            for model_key, metrics in fold_metrics.items():
                if metric in metrics:
                    values.append(metrics[metric]['mean'])
                    labels.append(model_key)
            
            if values:  # Only add trace if we have values
                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=labels,
                        y=values,
                        error_y=dict(
                            type='data',
                            array=[fold_metrics[label][metric]['std'] for label in labels],
                            visible=True
                        ),
                        showlegend=(i == 0)
                    ),
                    row=n_rows,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Model Comparison by Feature Type and Cross-Validation Results',
            template='plotly_white',
            height=300 * n_rows,  # Adjust height based on number of rows
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        for i in range(1, n_rows + 1):
            fig.update_yaxes(title_text="Score", row=i, col=1)
        
        fig.write_html(self.output_dir / 'model_comparison.html')
    
    def save_results(self, results: Dict[str, Dict[str, Any]], fold_metrics: Dict[str, Dict[str, float]]) -> None:
        """Save evaluation results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'per_fold_results': {},
            'cross_validation_metrics': fold_metrics
        }
        
        for key, value in results.items():
            serializable_results['per_fold_results'][key] = {
                'report': value['report'],
                'confusion_matrix': value['confusion_matrix'].tolist()
            }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def train_and_evaluate(self) -> None:
        """Train and evaluate all models using k-fold cross validation."""
        # Prepare data
        features, labels = self.prepare_data()
        
        # Initialize results storage
        results = {}
        fold_metrics = {}
        
        # Perform k-fold cross validation for each feature type and model
        for feature_type, X in features.items():
            for model_name in ['knn', 'lstm', 'transformer']:
                fold_results = []
                
                # Iterate through folds
                for fold, (X_train, X_val, y_train, y_val) in enumerate(self.cross_validator.split(X, labels)):
                    logger.info(f"\nProcessing fold {fold + 1}/{self.n_folds} for {model_name} with {feature_type}")
                    
                    # Train model
                    if model_name == 'knn':
                        model = self.train_knn(X_train, y_train)
                    elif model_name == 'lstm':
                        model = self.train_lstm(X_train, y_train)
                    else:  # transformer
                        model = self.train_transformer(X_train, y_train)
                    
                    # Evaluate model
                    fold_result = self.evaluate_model(model, X_val, y_val, model_name, feature_type)
                    fold_results.append(fold_result)
                    
                    # Add fold result to cross validator
                    self.cross_validator.add_fold_result(
                        fold=fold,
                        metrics=fold_result['report'],
                        model_name=f"{feature_type}_{model_name}"
                    )
                
                # Store results
                key = f"{feature_type}_{model_name}"
                results[key] = fold_results[-1]  # Store last fold's results for detailed analysis
                fold_metrics[key] = self.cross_validator.get_average_metrics()
        
        # Plot and save results
        self.plot_results(results, fold_metrics)
        self.save_results(results, fold_metrics)
        
        logger.info(f"\nResults saved to {self.output_dir}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate() 