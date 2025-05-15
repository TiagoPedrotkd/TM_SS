"""
Script to train and evaluate ensemble classification model.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Any, Tuple
import torch
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from train_models import ModelTrainer
from utils.data_loader import load_datasets
from preprocessing.text_processor import TextPreprocessor
from features.feature_extraction import create_feature_extractor
from models.classifiers import LSTMClassifier, TransformerClassifier, TorchClassifierWrapper
from features.feature_extraction import TransformerExtractor, BagOfWordsExtractor, CombinedFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier combining multiple models with weighted voting."""
    
    def __init__(
        self,
        models: List[Tuple[BaseEstimator, float]],
        voting: str = 'soft'
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of (model, weight) tuples
            voting: 'hard' or 'soft' voting strategy
        """
        self.models = models
        self.voting = voting
        
    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> 'EnsembleClassifier':
        """
        Fit each model in the ensemble.
        
        Args:
            X: Dictionary of feature matrices for each model
            y: Target labels
        """
        for (model, _), X_model in zip(self.models, X.values()):
            model.fit(X_model, y)
        return self
    
    def predict_proba(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using weighted voting.
        
        Args:
            X: Dictionary of feature matrices for each model
        """
        predictions = []
        weights = []
        
        for (model, weight), X_model in zip(self.models, X.values()):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_model)
            else:
                # For models without predict_proba, convert hard predictions
                pred = np.eye(3)[model.predict(X_model)]
            predictions.append(pred)
            weights.append(weight)
        
        # Weighted average of probabilities
        weights = np.array(weights) / sum(weights)
        return np.average(predictions, weights=weights, axis=0)
    
    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict classes using weighted voting.
        
        Args:
            X: Dictionary of feature matrices for each model
        """
        if self.voting == 'soft':
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = []
            weights = []
            
            for (model, weight), X_model in zip(self.models, X.values()):
                predictions.append(model.predict(X_model))
                weights.append(weight)
            
            predictions = np.array(predictions)
            weights = np.array(weights) / sum(weights)
            
            # Weighted mode of predictions
            weighted_preds = np.zeros((predictions.shape[1], 3))
            for i, weight in enumerate(weights):
                for j, pred in enumerate(predictions[i]):
                    weighted_preds[j, pred] += weight
            
            return np.argmax(weighted_preds, axis=1)

class EnsembleTrainer:
    def __init__(self, output_dir: str = None, n_folds: int = 5):
        if output_dir is None:
            output_dir = Path('results/ensemble_evaluation')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_trainer = ModelTrainer()
        self.preprocessor = TextPreprocessor()
        self.n_folds = n_folds
        
        # Load datasets once
        logger.info("Loading datasets...")
        self.train_df, self.test_df = load_datasets()
        
        # Updated model configurations using best performing features
        self.models = [
            ('finbert_cls', 0.3),     # FinBERT with CLS token
            ('finbert_mean', 0.25),    # FinBERT with mean pooling
            ('bow_tfidf_combined', 0.25),  # Combined BOW+TF-IDF features
            ('transformer', 0.2),      # Original transformer for diversity
        ]
        
        # Initialize feature extractors
        self.feature_extractors = {
            'finbert_cls': TransformerExtractor(
                model_name='ProsusAI/finbert',
                pooling_strategy='cls'
            ),
            'finbert_mean': TransformerExtractor(
                model_name='ProsusAI/finbert',
                pooling_strategy='mean'
            ),
            'bow_tfidf_combined': CombinedFeatureExtractor([
                BagOfWordsExtractor(use_tfidf=True),
                BagOfWordsExtractor(use_tfidf=False)
            ]),
            'transformer': TransformerExtractor()
        }
        
        # Initialize class weights using loaded data
        class_counts = self.train_df['label'].value_counts()
        total_samples = len(self.train_df)
        self.class_weights = {
            label: (total_samples / (len(class_counts) * count)) * 1.2  # Additional 20% boost
            if count < total_samples / len(class_counts)  # For minority classes
            else total_samples / (len(class_counts) * count)
            for label, count in class_counts.items()
        }
        
        # Model hyperparameters with focus on handling imbalance
        self.model_params = {
            'lstm': {
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.4,
                'batch_size': 64,  # Increased for faster training
                'num_epochs': 15,  # Reduced epochs
                'learning_rate': 0.001,
                'use_focal_loss': True,
                'focal_gamma': 2.5
            },
            'transformer': {
                'hidden_size': 256,  # Reduced size
                'num_heads': 8,
                'num_layers': 2,  # Reduced layers
                'dropout': 0.3,
                'batch_size': 32,
                'num_epochs': 10,  # Reduced epochs
                'learning_rate': 0.0001,
                'use_focal_loss': True,
                'focal_gamma': 2.5
            }
        }
    
    def train_model(self, model_type: str, features: np.ndarray, labels: np.ndarray) -> BaseEstimator:
        """Train a single model with improved handling of class imbalance."""
        logger.info(f"Training {model_type} model...")
        params = self.model_params[model_type]
        
        if model_type == 'lstm':
            model = LSTMClassifier(
                input_size=features.shape[-1],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        else:  # transformer
            model = TransformerClassifier(
                input_size=features.shape[-1],
                hidden_size=params['hidden_size'],
                dropout=params['dropout']
            )
        
        # Wrap model with improved training
        wrapped_model = TorchClassifierWrapper(
            model=model,
            optimizer_kwargs={'lr': params['learning_rate']},
            batch_size=params['batch_size'],
            num_epochs=params['num_epochs'],
            use_focal_loss=params['use_focal_loss'],
            focal_gamma=params['focal_gamma']
        )
        
        # Train model
        logger.info("Starting model training...")
        wrapped_model.fit(features, labels)
        logger.info("Model training completed")
        return wrapped_model
    
    def prepare_features(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """Prepare features for all models."""
        # Process texts using already loaded data
        train_texts = [self.preprocessor.preprocess(text) for text in self.train_df['text']]
        test_texts = [self.preprocessor.preprocess(text) for text in self.test_df['text']]
        
        # Extract features for each model type
        train_features = {}
        test_features = {}
        
        # Use the already initialized feature extractors
        for model_name, _ in self.models:
            logger.info(f"Extracting {model_name} features...")
            extractor = self.feature_extractors[model_name]
            
            # Fit and transform on training data
            train_features[model_name] = extractor.fit_transform(train_texts)
            # Transform test data using fitted extractor
            test_features[model_name] = extractor.transform(test_texts)
        
        return (
            train_features,
            test_features,
            self.train_df['label'].values
        )
    
    def create_confusion_matrix_plot(self, conf_matrix: np.ndarray, title: str) -> go.Figure:
        """Create a plotly confusion matrix visualization."""
        labels = ['Bearish', 'Bullish', 'Neutral']
        
        # Calculate percentages
        conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix_percent,
            x=labels,
            y=labels,
            text=np.array([
                [f"{conf_matrix[i, j]}<br>({conf_matrix_percent[i, j]:.1f}%)"
                 for j in range(len(labels))]
                for i in range(len(labels))
            ]),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorscale='RdYlBu',
            reversescale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=600,
            height=500
        )
        
        return fig
    
    def create_performance_plot(self, cv_results: List[Dict]) -> go.Figure:
        """Create performance comparison plot."""
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Bearish', 'Bullish', 'Neutral']
        
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=[m.capitalize() for m in metrics],
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            means = []
            stds = []
            for c in range(3):  # For each class
                values = [fold[str(c)][metric] for fold in cv_results]
                means.append(np.mean(values))
                stds.append(np.std(values))
            
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=classes,
                    y=means,
                    error_y=dict(
                        type='data',
                        array=stds,
                        visible=True
                    ),
                    showlegend=(i == 1)
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            height=800,
            title_text="Cross-Validation Performance by Class",
            showlegend=True
        )
        
        return fig
    
    def generate_html_report(
        self,
        cv_results: List[Dict],
        final_conf_matrix: np.ndarray,
        test_predictions: np.ndarray
    ) -> None:
        """Generate HTML report with visualizations."""
        # Create visualizations
        conf_matrix_fig = self.create_confusion_matrix_plot(
            final_conf_matrix,
            "Confusion Matrix (Final Model)"
        )
        
        performance_fig = self.create_performance_plot(cv_results)
        
        # Calculate overall metrics
        overall_metrics = {
            'accuracy': np.mean([fold['accuracy'] for fold in cv_results]),
            'macro_f1': np.mean([fold['macro avg']['f1-score'] for fold in cv_results])
        }
        
        # Generate HTML
        html_content = f"""
        <html>
        <head>
            <title>Ensemble Model Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <h1>Ensemble Model Evaluation Report</h1>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Weight</th>
                    </tr>
                    {''.join(f"<tr><td>{name}</td><td>{weight}</td></tr>" for name, weight in self.models)}
                </table>
            </div>
            
            <div class="section">
                <h2>Cross-Validation Results</h2>
                <p>Number of folds: {self.n_folds}</p>
                <p>Overall Accuracy: {overall_metrics['accuracy']:.3f}</p>
                <p>Macro F1-Score: {overall_metrics['macro_f1']:.3f}</p>
            </div>
            
            <div class="section">
                <h2>Performance Visualization</h2>
                {performance_fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <div class="section">
                <h2>Confusion Matrix</h2>
                {conf_matrix_fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <div class="section">
                <h2>Test Set Predictions</h2>
                <p>Number of test samples: {len(test_predictions)}</p>
                <p>Prediction distribution:</p>
                <ul>
                    <li>Bearish: {sum(test_predictions == 0)} ({sum(test_predictions == 0)/len(test_predictions)*100:.1f}%)</li>
                    <li>Bullish: {sum(test_predictions == 1)} ({sum(test_predictions == 1)/len(test_predictions)*100:.1f}%)</li>
                    <li>Neutral: {sum(test_predictions == 2)} ({sum(test_predictions == 2)/len(test_predictions)*100:.1f}%)</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'evaluation_report.html', 'w') as f:
            f.write(html_content)
    
    def train_and_evaluate(self) -> None:
        """Train and evaluate the ensemble model with improved class balance handling."""
        logger.info("Preparing features...")
        train_features, test_features, train_labels = self.prepare_features()
        
        # Initialize cross-validation
        cv_results = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        logger.info(f"\nPerforming {self.n_folds}-fold cross-validation...")
        
        # Use finbert_cls features for splitting - any feature set would work since we maintain indices
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_features['finbert_cls'], train_labels), 1):
            logger.info(f"\nFold {fold}/{self.n_folds}")
            
            # Split features for each model
            fold_train_features = {}
            fold_val_features = {}
            for model_name, _ in self.models:
                logger.info(f"Preparing features for {model_name}...")
                fold_train_features[model_name] = train_features[model_name][train_idx]
                fold_val_features[model_name] = train_features[model_name][val_idx]
            
            # Get labels
            fold_train_labels = train_labels[train_idx]
            fold_val_labels = train_labels[val_idx]
            
            # Train models for this fold
            models = []
            for model_name, weight in self.models:
                logger.info(f"Training model {model_name} with weight {weight}...")
                feat_type = model_name  # Use the full model name as the feature key
                model = self.train_model(
                    model_type='transformer',  # All our models are transformer-based now
                    features=fold_train_features[feat_type],
                    labels=fold_train_labels
                )
                models.append((model, weight))
                logger.info(f"Completed training {model_name}")
            
            # Create and evaluate ensemble
            logger.info("Creating ensemble model...")
            ensemble = EnsembleClassifier(models, voting='soft')
            logger.info("Fitting ensemble...")
            ensemble.fit(fold_train_features, fold_train_labels)
            logger.info("Making predictions...")
            val_predictions = ensemble.predict(fold_val_features)
            
            # Store results
            fold_report = classification_report(fold_val_labels, val_predictions, output_dict=True)
            cv_results.append(fold_report)
            
            logger.info(f"Fold {fold} Results:")
            logger.info(classification_report(fold_val_labels, val_predictions))
            
            # Save intermediate results
            intermediate_results = {
                'fold': fold,
                'cv_results': cv_results
            }
            with open(self.output_dir / f'intermediate_results_fold_{fold}.json', 'w') as f:
                json.dump(intermediate_results, f, indent=2)
        
        # Train final model on all training data
        logger.info("\nTraining final model on all data...")
        final_models = []
        for model_name, weight in self.models:
            logger.info(f"Training final {model_name} model...")
            feat_type = model_name  # Use the full model name as the feature key
            model = self.train_model(
                model_type='transformer',  # All our models are transformer-based now
                features=train_features[feat_type],
                labels=train_labels
            )
            final_models.append((model, weight))
            logger.info(f"Completed training final {model_name} model")
        
        logger.info("Creating final ensemble...")
        final_ensemble = EnsembleClassifier(final_models, voting='soft')
        logger.info("Fitting final ensemble...")
        final_ensemble.fit(train_features, train_labels)
        logger.info("Making final predictions...")
        test_predictions = final_ensemble.predict(test_features)
        train_predictions = final_ensemble.predict(train_features)
        
        # Calculate final confusion matrix
        final_conf_matrix = confusion_matrix(train_labels, train_predictions)
        
        # Save test predictions
        test_df = pd.read_csv(Path('data/test.csv'))
        predictions_df = pd.DataFrame({
            'id': test_df['id'],
            'text': test_df['text'],
            'predicted_label': test_predictions
        })
        predictions_df.to_csv(self.output_dir / 'test_predictions.csv', index=False)
        
        # Generate and save report
        self.generate_html_report(cv_results, final_conf_matrix, test_predictions)
        
        # Save detailed results
        results = {
            'cross_validation_results': cv_results,
            'final_confusion_matrix': final_conf_matrix.tolist(),
            'model_weights': {name: weight for name, weight in self.models}
        }
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {self.output_dir}")
        logger.info("Generated HTML report with visualizations")

    def train(self, texts: List[str], labels: List[int]) -> Dict:
        """Train ensemble model with improved feature extractors"""
        results = {}
        for model_name, weight in self.models:
            # Extract features using corresponding extractor
            features = self.feature_extractors[model_name].fit_transform(texts)
            
            # Train model with class balancing
            model_results = self.base_trainer.train_with_cv(
                features,
                labels,
                n_splits=self.n_folds,
                model_name=model_name,
                class_weight='balanced'
            )
            results[model_name] = model_results
        
        return results

if __name__ == "__main__":
    trainer = EnsembleTrainer()
    trainer.train_and_evaluate() 