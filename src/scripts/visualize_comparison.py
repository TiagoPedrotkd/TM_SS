"""
Script to create detailed visualizations of feature engineering comparison results.
"""

import logging
from pathlib import Path
from src.visualization.feature_comparison_viz import create_visualizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parents[2]  # TM_SS directory
    results_file = base_dir / "results" / "feature_comparison" / "financial_sentiment_results.csv"
    viz_dir = base_dir / "results" / "feature_comparison" / "visualizations"
    
    # Create visualizations
    logger.info(f"Creating visualizations from {results_file}...")
    create_visualizations(
        results_file=str(results_file),
        save_dir=str(viz_dir)
    )
    logger.info(f"Saved visualizations to {viz_dir}")

if __name__ == "__main__":
    main() 