"""
Script to train and evaluate ensemble classification model.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Any, Tuple
import torch
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
import joblib

from train_models import ModelTrainer
from utils.data_loader import load_datasets
from preprocessing.text_processor import TextPreprocessor
from features.feature_extraction import create_feature_extractor
from models.classifiers import LSTMClassifier, TransformerClassifier, TorchClassifierWrapper
from features.feature_extraction import TransformerExtractor, BagOfWordsExtractor, CombinedFeatureExtractor, Word2VecExtractor

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
                pred = np.eye(3)[model.predict(X_model)]
            predictions.append(pred)
            weights.append(weight)
        
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
            
            weighted_preds = np.zeros((predictions.shape[1], 3))
            for i, weight in enumerate(weights):
                for j, pred in enumerate(predictions[i]):
                    weighted_preds[j, pred] += weight
            
            return np.argmax(weighted_preds, axis=1)

class EnsembleTrainer:
    def __init__(self, output_dir: str = None, n_folds: int = 5):
        if output_dir is None:
            output_dir = Path('../results/ensemble_evaluation')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_trainer = ModelTrainer()
        self.preprocessor = TextPreprocessor()
        self.n_folds = n_folds
        
        logger.info("Loading datasets...")
        self.train_df, self.test_df = load_datasets()
        
        self.models = [
            ('bow_tfidf_finbert-lstm', 0.30),
            ('bow-knn', 0.20),
            ('word2vec-lstm', 0.25),
            ('finbert-transformer', 0.25),
        ]
        
        self.feature_extractors = {
            'bow_tfidf_finbert-lstm': CombinedFeatureExtractor([
                BagOfWordsExtractor(use_tfidf=True),
                BagOfWordsExtractor(use_tfidf=False),
                TransformerExtractor(model_name='ProsusAI/finbert', pooling_strategy='mean')
            ]),
            'bow-knn': BagOfWordsExtractor(use_tfidf=False),
            'word2vec-lstm': Word2VecExtractor(),
            'finbert-transformer': TransformerExtractor(
                model_name='ProsusAI/finbert',
                pooling_strategy='cls'
            )
        }
        
        self.model_configs = {
            'lstm': {
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.4,
                'batch_size': 64,
                'num_epochs': 15,
                'learning_rate': 0.001,
                'use_focal_loss': True,
                'focal_gamma': 2.5
            },
            'transformer': {
                'hidden_size': 256,
                'num_heads': 8,
                'num_layers': 2,
                'dropout': 0.3,
                'batch_size': 32,
                'num_epochs': 10,
                'learning_rate': 0.0001,
                'use_focal_loss': True,
                'focal_gamma': 2.5
            },
            'knn': {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'cosine'
            }
        }
        
        class_counts = self.train_df['label'].value_counts()
        total_samples = len(self.train_df)
        self.class_weights = {
            label: (total_samples / (len(class_counts) * count)) * 1.2
            if count < total_samples / len(class_counts)
            else total_samples / (len(class_counts) * count)
            for label, count in class_counts.items()
        }
    
    def train_model(self, model_name: str, features: np.ndarray, labels: np.ndarray) -> BaseEstimator:
        """Train a single model with improved logging."""
        logger.info(f"Training {model_name} model...")
        
        if 'knn' in model_name:
            model = KNeighborsClassifier(**self.model_configs['knn'])
            model.fit(features, labels)
            
        elif 'lstm' in model_name:
            params = self.model_configs['lstm']
            model = LSTMClassifier(
                input_size=features.shape[-1],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            model = TorchClassifierWrapper(
                model=model,
                optimizer_kwargs={'lr': params['learning_rate']},
                batch_size=params['batch_size'],
                num_epochs=params['num_epochs'],
                use_focal_loss=params['use_focal_loss'],
                focal_gamma=params['focal_gamma']
            )
            model.fit(features, labels)
            
        else:
            params = self.model_configs['transformer']
            model = TransformerClassifier(
                input_size=features.shape[-1],
                hidden_size=params['hidden_size'],
                dropout=params['dropout']
            )
            model = TorchClassifierWrapper(
                model=model,
                optimizer_kwargs={'lr': params['learning_rate']},
                batch_size=params['batch_size'],
                num_epochs=params['num_epochs'],
                use_focal_loss=params['use_focal_loss'],
                focal_gamma=params['focal_gamma']
            )
            model.fit(features, labels)
        
        predictions = model.predict(features)
        mcc_score = matthews_corrcoef(labels, predictions)
        logger.info(f"{model_name} Training MCC: {mcc_score:.4f}")
        
        return model
    
    def prepare_features(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """Prepare features for all models."""
        train_texts = [self.preprocessor.preprocess(text) for text in self.train_df['text']]
        test_texts = [self.preprocessor.preprocess(text) for text in self.test_df['text']]
        
        train_features = {}
        test_features = {}
        
        for model_name, _ in self.models:
            logger.info(f"Extracting {model_name} features...")
            extractor = self.feature_extractors[model_name]
            
            train_features[model_name] = extractor.fit_transform(train_texts)
            test_features[model_name] = extractor.transform(test_texts)
        
        return (
            train_features,
            test_features,
            self.train_df['label'].values
        )
    
    def create_confusion_matrix_plot(self, conf_matrix: np.ndarray, title: str) -> go.Figure:
        """Create a plotly confusion matrix visualization."""
        labels = ['Bearish', 'Bullish', 'Neutral']
        
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
    
    def create_performance_plot(self, cv_results: List[Dict], mcc_scores: List[float]) -> go.Figure:
        """Create performance comparison plot with MCC."""
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Bearish', 'Bullish', 'Neutral']
        
        fig = make_subplots(
            rows=len(metrics) + 2,
            cols=1,
            subplot_titles=[m.capitalize() for m in metrics] + 
                         ['Individual Model MCC Scores', 'Ensemble MCC'],
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            means = []
            stds = []
            for c in range(3):
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
        
        model_mccs = {}
        for result in cv_results:
            model_name = result.get('model_name', 'Unknown')
            if 'matthews_corrcoef' in result:
                if model_name not in model_mccs:
                    model_mccs[model_name] = []
                model_mccs[model_name].append(result['matthews_corrcoef'])
        
        model_names = []
        mcc_means = []
        mcc_stds = []
        for model_name, scores in model_mccs.items():
            model_names.append(model_name)
            mcc_means.append(np.mean(scores))
            mcc_stds.append(np.std(scores))
        
        fig.add_trace(
            go.Bar(
                name='Individual MCCs',
                x=model_names,
                y=mcc_means,
                error_y=dict(
                    type='data',
                    array=mcc_stds,
                    visible=True
                ),
                showlegend=True
            ),
            row=len(metrics) + 1,
            col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Ensemble MCC',
                x=['Cross-validation'],
                y=[np.mean(mcc_scores)],
                error_y=dict(
                    type='data',
                    array=[np.std(mcc_scores)],
                    visible=True
                ),
                showlegend=True
            ),
            row=len(metrics) + 2,
            col=1
        )
        
        fig.update_layout(
            height=1200,
            title_text="Model Performance Metrics Including Individual and Ensemble MCC",
            showlegend=True
        )
        
        return fig
    
    def generate_html_report(
        self,
        cv_results: List[Dict],
        final_conf_matrix: np.ndarray,
        test_predictions: np.ndarray,
        mcc_scores: List[float],
        final_mcc: float
    ) -> None:
        """Generate HTML report with visualizations."""
        performance_fig = self.create_performance_plot(cv_results, mcc_scores)
        
        test_pred_dist = pd.Series(test_predictions).value_counts()
        
        overall_metrics = {
            'accuracy': np.mean([fold['accuracy'] for fold in cv_results]),
            'macro_f1': np.mean([fold['macro avg']['f1-score'] for fold in cv_results]),
            'mcc_cv_mean': np.mean(mcc_scores),
            'mcc_cv_std': np.std(mcc_scores)
        }
        
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
                .metric-box {{
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .metric-title {{
                    font-weight: bold;
                    color: #495057;
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
                <h2>Overall Performance Metrics</h2>
                <div class="metric-box">
                    <div class="metric-title">Cross-Validation Accuracy</div>
                    <div>{overall_metrics['accuracy']:.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Cross-Validation Macro F1-Score</div>
                    <div>{overall_metrics['macro_f1']:.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Cross-Validation MCC</div>
                    <div>{overall_metrics['mcc_cv_mean']:.3f} ± {overall_metrics['mcc_cv_std']:.3f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Visualization</h2>
                {performance_fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <div class="section">
                <h2>Test Set Predictions</h2>
                <div class="metric-box">
                    <div class="metric-title">Number of test samples</div>
                    <div>{len(test_predictions)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Prediction distribution</div>
                    <ul>
                        <li>Bearish: {test_pred_dist.get(0, 0)} ({test_pred_dist.get(0, 0)/len(test_predictions)*100:.1f}%)</li>
                        <li>Bullish: {test_pred_dist.get(1, 0)} ({test_pred_dist.get(1, 0)/len(test_predictions)*100:.1f}%)</li>
                        <li>Neutral: {test_pred_dist.get(2, 0)} ({test_pred_dist.get(2, 0)/len(test_predictions)*100:.1f}%)</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'evaluation_report.html', 'w') as f:
            f.write(html_content)
    
    def train_and_evaluate(self) -> None:
        """Train and evaluate the ensemble model."""
        logger.info("Preparing features...")
        train_features, test_features, train_labels = self.prepare_features()
        
        cv_results = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        mcc_scores = []
        
        logger.info(f"\nPerforming {self.n_folds}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_features['finbert-transformer'], train_labels), 1):
            logger.info(f"\nFold {fold}/{self.n_folds}")
            
            fold_train_features = {}
            fold_val_features = {}
            for model_name, _ in self.models:
                logger.info(f"Preparing features for {model_name}...")
                fold_train_features[model_name] = train_features[model_name][train_idx]
                fold_val_features[model_name] = train_features[model_name][val_idx]
            
            fold_train_labels = train_labels[train_idx]
            fold_val_labels = train_labels[val_idx]
            
            models = []
            for model_name, weight in self.models:
                logger.info(f"Training model {model_name} with weight {weight}...")
                feat_type = model_name
                model = self.train_model(
                    model_name=feat_type,
                    features=fold_train_features[feat_type],
                    labels=fold_train_labels
                )
                models.append((model, weight))
                logger.info(f"Completed training {model_name}")
            
            logger.info("Creating ensemble model...")
            ensemble = EnsembleClassifier(models, voting='soft')
            logger.info("Fitting ensemble...")
            ensemble.fit(fold_train_features, fold_train_labels)
            logger.info("Making predictions...")
            val_predictions = ensemble.predict(fold_val_features)
            
            fold_report = classification_report(fold_val_labels, val_predictions, output_dict=True)
            mcc_score = matthews_corrcoef(fold_val_labels, val_predictions)
            fold_report['matthews_corrcoef'] = mcc_score
            mcc_scores.append(mcc_score)
            cv_results.append(fold_report)
            
            logger.info(f"Fold {fold} Results:")
            logger.info(classification_report(fold_val_labels, val_predictions))
            logger.info(f"Matthews Correlation Coefficient: {mcc_score:.4f}")
            
            intermediate_results = {
                'fold': fold,
                'cv_results': cv_results,
                'mcc_scores': mcc_scores
            }
            with open(self.output_dir / f'intermediate_results_fold_{fold}.json', 'w') as f:
                json.dump(intermediate_results, f, indent=2)
        
        logger.info("\nTraining final model on all data...")
        final_models = []
        for model_name, weight in self.models:
            logger.info(f"Training final {model_name} model...")
            feat_type = model_name
            model = self.train_model(
                model_name=feat_type,
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
        
        models_dir = Path('../models')
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_ensemble, models_dir / 'final_ensemble_model.joblib')
        logger.info(f"Final ensemble model exported to {models_dir / 'final_ensemble_model.joblib'}")
        
        self.generate_html_report(
            cv_results=cv_results,
            final_conf_matrix=None,
            test_predictions=test_predictions,
            mcc_scores=mcc_scores,
            final_mcc=None
        )
        
        results = {
            'cross_validation_results': cv_results,
            'test_predictions': test_predictions.tolist(),
            'model_weights': {name: weight for name, weight in self.models},
            'mcc_scores': {
                'cv_mean': np.mean(mcc_scores),
                'cv_std': np.std(mcc_scores)
            }
        }
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {self.output_dir}")
        logger.info("Generated HTML report with visualizations")

    def train(self, texts: List[str], labels: List[int]) -> Dict:
        """Train ensemble model with improved feature extractors"""
        results = {}
        for model_name, weight in self.models:
            features = self.feature_extractors[model_name].fit_transform(texts)
            
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