import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from models.balanced_ensemble import BalancedEnsembleClassifier
from utils.data_loader import load_datasets
from preprocessing.text_processor import TextPreprocessor
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data():
    """Load and prepare data for training"""
    # Load datasets
    train_df, test_df = load_datasets()
    
    # Preprocess texts
    preprocessor = TextPreprocessor()
    X_train = [preprocessor.preprocess(text) for text in train_df['text']]
    y_train = train_df['label'].values
    
    X_test = [preprocessor.preprocess(text) for text in test_df['text']]
    y_test = test_df['label'].values
    
    return X_train, y_train, X_test, y_test

def create_base_models():
    """Create base models with balanced settings"""
    return [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        )),
        ('lr', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]

def save_results(y_true, y_pred, output_dir):
    """Save evaluation results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save results
    results = {
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, 'balanced_ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to %s", output_dir)

def main():
    # Prepare data
    logger.info("Loading and preparing data...")
    X_train, y_train, X_test, y_test = prepare_data()
    
    # Create base models
    base_models = create_base_models()
    
    # Create and train balanced ensemble
    logger.info("Training balanced ensemble...")
    ensemble = BalancedEnsembleClassifier(
        base_models,
        sampling_strategy='auto',
        random_state=42
    )
    
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = ensemble.predict(X_test)
    
    # Save results
    output_dir = Path('results/balanced_ensemble')
    save_results(y_test, y_pred, output_dir)
    
    # Print summary
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main() 