import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

class BalancedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier with improved handling of class imbalance.
    
    Features:
    - Uses SMOTE for minority class oversampling
    - Applies class weights
    - Combines multiple base models with balanced voting
    """
    
    def __init__(self, base_models, sampling_strategy='auto', random_state=42):
        """
        Initialize the balanced ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            sampling_strategy: Strategy for SMOTE ('auto' or dict)
            random_state: Random seed for reproducibility
        """
        self.base_models = base_models
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        
        # Initialize samplers
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        
        # Optional: Add undersampling for extreme imbalance
        self.undersampler = RandomUnderSampler(
            sampling_strategy=0.8,  # Keep 80% of majority class
            random_state=random_state
        )
        
    def fit(self, X, y):
        """
        Fit the ensemble using balanced data.
        
        Args:
            X: Features
            y: Labels
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create resampling pipeline
        logger.info("Applying SMOTE oversampling...")
        resampling = ImbPipeline([
            ('smote', self.smote),
            ('undersampler', self.undersampler)
        ])
        
        # Resample data
        X_resampled, y_resampled = resampling.fit_resample(X, y_encoded)
        
        # Calculate class weights
        class_counts = np.bincount(y_encoded)
        class_weights = {
            i: len(y_encoded) / (len(np.unique(y_encoded)) * count)
            for i, count in enumerate(class_counts)
        }
        
        logger.info("Class distribution after resampling:")
        for label, count in zip(
            self.label_encoder.classes_,
            np.bincount(y_resampled)
        ):
            logger.info(f"{label}: {count} samples")
        
        # Fit each base model on balanced data
        self.fitted_models = []
        for name, model in self.base_models:
            logger.info(f"Fitting {name}...")
            
            # Set class weights if model supports it
            if hasattr(model, 'class_weight'):
                model.class_weight = class_weights
            
            # Fit model
            model.fit(X_resampled, y_resampled)
            self.fitted_models.append((name, model))
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted voting.
        
        Args:
            X: Features
        
        Returns:
            Array of class probabilities
        """
        # Collect predictions from all models
        predictions = []
        for name, model in self.fitted_models:
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        # Average predictions (can be weighted based on model performance)
        return np.mean(predictions, axis=0)
    
    def predict(self, X):
        """
        Predict classes using weighted voting.
        
        Args:
            X: Features
        
        Returns:
            Array of predicted labels
        """
        # Get probability predictions
        proba = self.predict_proba(X)
        
        # Convert to class labels
        predictions = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(predictions)

def evaluate_balanced_ensemble(X, y, base_models, cv=5):
    """
    Evaluate the balanced ensemble using cross-validation.
    
    Args:
        X: Features
        y: Labels
        base_models: List of (name, model) tuples
        cv: Number of cross-validation folds
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Create and evaluate ensemble
    ensemble = BalancedEnsembleClassifier(base_models)
    
    # Perform cross-validation
    scores = cross_val_score(
        ensemble, X, y,
        cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    )
    
    # Return metrics
    return {
        'accuracy': scores['test_accuracy'].mean(),
        'f1_macro': scores['test_f1_macro'].mean(),
        'precision_macro': scores['test_precision_macro'].mean(),
        'recall_macro': scores['test_recall_macro'].mean()
    }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load your data and models here
    # Example:
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.linear_model import LogisticRegression
    # 
    # base_models = [
    #     ('rf', RandomForestClassifier(class_weight='balanced')),
    #     ('lr', LogisticRegression(class_weight='balanced'))
    # ]
    # 
    # results = evaluate_balanced_ensemble(X, y, base_models)
    # print("Evaluation results:", results) 