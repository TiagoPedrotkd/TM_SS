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
from typing import Dict, Any, Tuple

from utils.data_loader import load_datasets
from preprocessing.text_processor import TextPreprocessor
from features.feature_extraction import create_feature_extractor
from models.classifiers import LSTMClassifier, TransformerClassifier, TorchClassifierWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, output_dir: str = None):
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
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Add feature type and model name to report
        report['feature_type'] = feature_type
        report['model_name'] = model_name
        
        return {
            'report': report,
            'confusion_matrix': conf_matrix
        }
    
    def plot_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Plot evaluation results."""
        # Create subplots for each metric
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Bearish', 'Bullish', 'Neutral']
        
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=[metric.capitalize() for metric in metrics]
        )
        
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
        
        fig.update_layout(
            title='Model Comparison by Feature Type',
            template='plotly_white',
            height=900
        )
        
        fig.write_html(self.output_dir / 'model_comparison.html')
    
    def save_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save evaluation results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = {
                'report': value['report'],
                'confusion_matrix': value['confusion_matrix'].tolist()
            }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def train_and_evaluate(self) -> None:
        """Train and evaluate all models."""
        # Prepare data
        features, labels = self.prepare_data()
        
        # Split data
        results = {}
        for feature_type, X in features.items():
            X_train, X_val, y_train, y_val = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )
            
            # Train and evaluate KNN
            knn_model = self.train_knn(X_train, y_train)
            results[f"{feature_type}_knn"] = self.evaluate_model(
                knn_model, X_val, y_val, 'knn', feature_type
            )
            
            # Train and evaluate LSTM
            lstm_model = self.train_lstm(X_train, y_train)
            results[f"{feature_type}_lstm"] = self.evaluate_model(
                lstm_model, X_val, y_val, 'lstm', feature_type
            )
            
            # Train and evaluate Transformer
            transformer_model = self.train_transformer(X_train, y_train)
            results[f"{feature_type}_transformer"] = self.evaluate_model(
                transformer_model, X_val, y_val, 'transformer', feature_type
            )
        
        # Plot and save results
        self.plot_results(results)
        self.save_results(results)
        
        logger.info(f"\nResults saved to {self.output_dir}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate() 