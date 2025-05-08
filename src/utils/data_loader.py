"""
Data loading utilities for the text mining project.
"""
import pandas as pd
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_dataset(df: pd.DataFrame, is_train: bool = True) -> bool:
    """
    Validate the dataset structure and content.
    
    Args:
        df: DataFrame to validate
        is_train: Whether this is training data (with labels) or test data
    
    Returns:
        bool: True if validation passes
    """
    # Check required columns
    required_cols = ['text']
    if is_train:
        required_cols.append('label')
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for null values
    null_counts = df['text'].isnull().sum()
    if null_counts > 0:
        logger.error(f"Found {null_counts} null values in 'text' column")
        return False
    
    if is_train:
        # Validate labels
        valid_labels = {0, 1, 2}  # Bearish, Bullish, Neutral
        invalid_labels = set(df['label'].unique()) - valid_labels
        if invalid_labels:
            logger.error(f"Found invalid labels: {invalid_labels}")
            return False
        
        # Check label distribution
        label_dist = df['label'].value_counts()
        logger.info("Label distribution:")
        for label, count in label_dist.items():
            logger.info(f"Label {label}: {count} samples ({count/len(df)*100:.2f}%)")
    
    logger.info(f"Dataset validation {'passed' if is_train else 'passed (test set)'}")
    return True

def load_datasets():
    """
    Load training and test datasets.
    
    Returns:
        tuple: (train_df, test_df) pandas DataFrames
    """
    # Get the data directory path
    current_dir = Path(__file__).parent.parent.parent
    data_dir = current_dir / 'data'
    
    # Load datasets
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Test the data loader
    try:
        train_df, test_df = load_datasets()
        print("\nDataset Statistics:")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
    except Exception as e:
        print(f"Error loading datasets: {str(e)}") 