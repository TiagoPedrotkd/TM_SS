"""
Script to compare different feature engineering approaches.
"""

import logging
import pandas as pd
from pathlib import Path
from src.features.feature_comparison import run_comparison
from src.preprocessing.text_processor import TextPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parents[2]  # TM_SS directory
    data_file = base_dir / "data" / "train.csv"
    results_dir = base_dir / "results" / "feature_comparison"
    
    # Load data
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Preprocess texts
    logger.info("Preprocessing texts...")
    preprocessor = TextPreprocessor()
    texts = [preprocessor.preprocess(text) for text in df['text']]
    labels = df['label'].values  # Already binary (0: Bearish/Negative, 1: Bullish/Positive)
    
    # Run comparison
    logger.info("Running feature comparison...")
    results = run_comparison(
        texts=texts,
        labels=labels,
        save_dir=str(results_dir),
        save_prefix='financial_sentiment'
    )
    
    # Print results
    logger.info("\nFeature Engineering Comparison Results:")
    logger.info("-" * 50)
    logger.info("\nPerformance (F1 Score):")
    for idx, row in results.iterrows():
        logger.info(f"{idx:20s}: {row['cv_mean']:.4f} ± {row['cv_std']:.4f}")
    
    logger.info("\nFeature Dimensions:")
    for idx, row in results.iterrows():
        logger.info(f"{idx:20s}: {row['feature_dim']}")
    
    logger.info("\nProcessing Times:")
    for idx, row in results.iterrows():
        total_time = row['extraction_time'] + row['training_time']
        logger.info(f"{idx:20s}: {total_time:.2f}s (Extract: {row['extraction_time']:.2f}s, Train: {row['training_time']:.2f}s)")

if __name__ == "__main__":
    main() 