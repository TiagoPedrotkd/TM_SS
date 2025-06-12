"""
Visualization module for feature engineering comparison results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def plot_detailed_comparison(results_df: pd.DataFrame, save_dir: Path) -> None:
    """Create detailed comparison visualizations."""
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 1. Performance Comparison with Error Bars
    plt.figure(figsize=(15, 8))
    performance_data = results_df.sort_values('cv_mean', ascending=False)
    
    # Create bar plot
    ax = sns.barplot(
        data=performance_data.reset_index(),
        x='index',
        y='cv_mean',
        yerr=performance_data['cv_std'],
        capsize=5
    )
    
    # Add value labels
    for i, v in enumerate(performance_data['cv_mean']):
        ax.text(
            i, v + 0.01,
            f'{v:.3f}±{performance_data["cv_std"].iloc[i]:.3f}',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    plt.title('Model Performance Comparison (F1 Score)', pad=20, size=14)
    plt.xlabel('Feature Engineering Method', size=12)
    plt.ylabel('F1 Score (weighted)', size=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'detailed_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time Analysis
    plt.figure(figsize=(15, 8))
    
    # Prepare time data
    time_data = results_df[['extraction_time', 'training_time']].copy()
    time_data['total_time'] = time_data['extraction_time'] + time_data['training_time']
    time_data = time_data.sort_values('total_time')
    
    # Create stacked bars
    ax = time_data[['extraction_time', 'training_time']].plot(
        kind='bar',
        stacked=True,
        figsize=(15, 8)
    )
    
    # Add total time labels
    for i, total in enumerate(time_data['total_time']):
        ax.text(
            i, total + 0.5,
            f'{total:.1f}s',
            ha='center',
            va='bottom'
        )
    
    plt.title('Processing Time Analysis', pad=20, size=14)
    plt.xlabel('Feature Engineering Method', size=12)
    plt.ylabel('Time (seconds)', size=12)
    plt.legend(['Feature Extraction', 'Model Training'], loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'detailed_timing.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency Analysis
    plt.figure(figsize=(15, 8))
    
    # Calculate efficiency metrics
    efficiency_data = pd.DataFrame({
        'Performance': results_df['cv_mean'],
        'Time': results_df['extraction_time'] + results_df['training_time'],
        'Features': results_df['feature_dim']
    })
    
    # Create scatter plot
    scatter = plt.scatter(
        efficiency_data['Time'],
        efficiency_data['Performance'],
        s=efficiency_data['Features'] / 10,  # Size proportional to feature count
        alpha=0.6
    )
    
    # Add labels
    for i, row in efficiency_data.iterrows():
        plt.annotate(
            i,
            (row['Time'], row['Performance']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title('Efficiency Analysis (Performance vs. Time vs. Feature Count)', pad=20, size=14)
    plt.xlabel('Total Processing Time (seconds)', size=12)
    plt.ylabel('F1 Score', size=12)
    
    # Add legend for bubble sizes
    legend_elements = [
        plt.scatter([], [], s=s/10, label=f'{s} features', alpha=0.6)
        for s in [100, 1000, 3000]
    ]
    plt.legend(handles=legend_elements, title='Feature Dimensions', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Dimensionality Analysis
    plt.figure(figsize=(15, 8))
    
    # Sort by dimension
    dim_data = results_df.sort_values('feature_dim', ascending=True)
    
    # Create bar plot with log scale
    ax = plt.subplot(111)
    bars = plt.bar(range(len(dim_data)), dim_data['feature_dim'])
    
    # Add value labels
    for i, v in enumerate(dim_data['feature_dim']):
        plt.text(
            i,
            v * 1.1,
            f'{int(v)}',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    plt.yscale('log')
    plt.title('Feature Dimensionality Comparison', pad=20, size=14)
    plt.xlabel('Feature Engineering Method', size=12)
    plt.ylabel('Number of Features (log scale)', size=12)
    plt.xticks(range(len(dim_data)), dim_data.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'dimensionality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Summary Table
    summary_df = pd.DataFrame({
        'F1 Score': [f"{row['cv_mean']:.3f} ± {row['cv_std']:.3f}" for _, row in results_df.iterrows()],
        'Features': results_df['feature_dim'].astype(int),
        'Extract Time (s)': results_df['extraction_time'].round(2),
        'Train Time (s)': results_df['training_time'].round(2),
        'Total Time (s)': (results_df['extraction_time'] + results_df['training_time']).round(2)
    })
    
    # Sort by F1 Score
    summary_df = summary_df.sort_values('F1 Score', ascending=False)
    
    # Save to CSV
    summary_df.to_csv(save_dir / 'detailed_summary.csv')

def create_visualizations(results_file: str, save_dir: str = None) -> None:
    """
    Create detailed visualizations from results file.
    
    Args:
        results_file: Path to the results CSV file
        save_dir: Directory to save visualizations
    """
    # Load results
    results_df = pd.read_csv(results_file, index_col=0)
    
    # Set up save directory
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'results' / 'visualizations'
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    plot_detailed_comparison(results_df, save_dir)