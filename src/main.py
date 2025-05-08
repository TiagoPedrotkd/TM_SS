import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from pathlib import Path

from preprocessing.text_processor import TextProcessor
from features.feature_engineering import FeatureExtractor
from models.classifiers import LSTMClassifier, TransformerClassifier, TorchClassifierWrapper
from utils.visualization import (
    plot_class_distribution,
    plot_word_frequencies,
    plot_evaluation_metrics,
    save_plot
)

def load_data():
    """Load training and test data."""
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    return train_df, test_df

def explore_data(train_df):
    """Perform data exploration and save visualizations."""
    # Plot class distribution
    class_dist_fig = plot_class_distribution(train_df)
    save_plot(class_dist_fig, 'class_distribution')
    
    # Plot word frequencies
    word_freq_fig = plot_word_frequencies(train_df['text'])
    save_plot(word_freq_fig, 'word_frequencies')

def preprocess_data(train_df, test_df):
    """Preprocess text data."""
    processor = TextProcessor()
    
    # Apply preprocessing to train and test data
    train_processed = train_df['text'].apply(
        lambda x: processor.preprocess(x, steps=['urls', 'special_chars', 'stopwords', 'lemmatize'])
    )
    test_processed = test_df['text'].apply(
        lambda x: processor.preprocess(x, steps=['urls', 'special_chars', 'stopwords', 'lemmatize'])
    )
    
    return train_processed, test_processed

def extract_features(train_texts, test_texts, method='transformer'):
    """Extract features from preprocessed texts."""
    feature_extractor = FeatureExtractor(
        method=method,
        model_name='bert-base-uncased' if method == 'transformer' else None
    )
    
    # Fit and transform training data
    train_features = feature_extractor.fit_transform(train_texts)
    # Transform test data
    test_features = feature_extractor.transform(test_texts)
    
    return train_features, test_features

def train_and_evaluate_models(X_train, y_train, X_test=None):
    """Train and evaluate multiple models."""
    # Split training data into train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    models = {
        'lstm': TorchClassifierWrapper(
            model=LSTMClassifier(
                input_size=X_train.shape[-1],
                hidden_size=128,
                num_layers=2
            ),
            num_epochs=10,
            batch_size=32
        ),
        'transformer': TorchClassifierWrapper(
            model=TransformerClassifier(
                input_size=X_train.shape[-1],
                nhead=8,
                num_layers=2
            ),
            num_epochs=10,
            batch_size=32
        )
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train_split, y_train_split)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        results[name] = classification_report(y_val, y_pred, output_dict=True)
    
    # Plot evaluation metrics
    metrics_fig = plot_evaluation_metrics(results)
    save_plot(metrics_fig, 'model_comparison')
    
    return models, results

def main():
    # Create necessary directories
    Path('results').mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Explore data
    print("Exploring data...")
    explore_data(train_df)
    
    # Preprocess data
    print("Preprocessing data...")
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Extract features
    print("Extracting features...")
    train_features, test_features = extract_features(train_processed, test_processed)
    
    # Train and evaluate models
    print("Training and evaluating models...")
    models, results = train_and_evaluate_models(
        train_features,
        train_df['label'].values,
        test_features
    )
    
    # Save results
    print("Results saved in the 'results' directory")

if __name__ == "__main__":
    main() 