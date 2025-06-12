"""
Corpus splitting analysis module that generates an HTML report with visualizations.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import sys
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from collections import Counter

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from utils.data_loader import load_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusSplitter:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Use Path for more reliable path handling
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]  # Go up 2 levels from src/analysis
            output_dir = project_root / 'results' / 'corpus_splitting'
        
        self.output_dir = Path(output_dir)
        logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_random_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[go.Figure, Dict]:
        """Analyze random train/validation split."""
        # Perform split
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state
        )
        
        # Analyze label distribution
        train_dist = df.iloc[train_idx]['label'].value_counts().sort_index()
        val_dist = df.iloc[val_idx]['label'].value_counts().sort_index()
        
        # Create visualization
        label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        
        fig = go.Figure(data=[
            go.Bar(
                name='Training Set',
                x=[label_names[i] for i in train_dist.index],
                y=train_dist.values,
                text=[f"{v/len(train_idx)*100:.1f}%" for v in train_dist.values],
                textposition='auto',
                marker_color='rgb(55, 83, 109)'
            ),
            go.Bar(
                name='Validation Set',
                x=[label_names[i] for i in val_dist.index],
                y=val_dist.values,
                text=[f"{v/len(val_idx)*100:.1f}%" for v in val_dist.values],
                textposition='auto',
                marker_color='rgb(26, 118, 255)'
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Label Distribution in Random Split',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Sentiment',
            yaxis_title='Count',
            template='plotly_white',
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        stats = {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_dist': {label_names[i]: v/len(train_idx)*100 for i, v in train_dist.items()},
            'val_dist': {label_names[i]: v/len(val_idx)*100 for i, v in val_dist.items()}
        }
        
        return fig, stats
    
    def analyze_stratified_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[go.Figure, Dict]:
        """Analyze stratified train/validation split."""
        # Perform split
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            stratify=df['label'],
            random_state=random_state
        )
        
        # Analyze label distribution
        train_dist = df.iloc[train_idx]['label'].value_counts().sort_index()
        val_dist = df.iloc[val_idx]['label'].value_counts().sort_index()
        
        # Create visualization
        label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        
        fig = go.Figure(data=[
            go.Bar(
                name='Training Set',
                x=[label_names[i] for i in train_dist.index],
                y=train_dist.values,
                text=[f"{v/len(train_idx)*100:.1f}%" for v in train_dist.values],
                textposition='auto'
            ),
            go.Bar(
                name='Validation Set',
                x=[label_names[i] for i in val_dist.index],
                y=val_dist.values,
                text=[f"{v/len(val_idx)*100:.1f}%" for v in val_dist.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Label Distribution in Stratified Split',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            template='plotly_white',
            barmode='group'
        )
        
        stats = {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_dist': {label_names[i]: v/len(train_idx)*100 for i, v in train_dist.items()},
            'val_dist': {label_names[i]: v/len(val_idx)*100 for i, v in val_dist.items()}
        }
        
        return fig, stats
    
    def analyze_kfold(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        stratified: bool = True,
        random_state: int = 42
    ) -> Tuple[go.Figure, Dict]:
        """Analyze K-Fold cross-validation splits."""
        # Create K-Fold splitter
        if stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(df, df['label']))
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(df))
        
        # Analyze label distribution for each fold
        label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        fold_stats = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_dist = df.iloc[train_idx]['label'].value_counts().sort_index()
            val_dist = df.iloc[val_idx]['label'].value_counts().sort_index()
            
            fold_stats.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_dist': {label_names[i]: v/len(train_idx)*100 for i, v in train_dist.items()},
                'val_dist': {label_names[i]: v/len(val_idx)*100 for i, v in val_dist.items()}
            })
        
        # Create visualization
        fig = go.Figure()
        
        for label in [0, 1, 2]:
            # Training set distribution
            fig.add_trace(go.Box(
                name=f'{label_names[label]} (Train)',
                y=[stats['train_dist'][label_names[label]] for stats in fold_stats],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            # Validation set distribution
            fig.add_trace(go.Box(
                name=f'{label_names[label]} (Val)',
                y=[stats['val_dist'][label_names[label]] for stats in fold_stats],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title=f'Label Distribution Across {n_splits}-Fold {"Stratified " if stratified else ""}Cross-Validation',
            yaxis_title='Percentage (%)',
            template='plotly_white',
            showlegend=True
        )
        
        return fig, fold_stats
    
    def generate_report(self) -> None:
        """Generate HTML report with corpus splitting analysis."""
        # Load data
        train_df, _ = load_datasets()
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # 1. Random split analysis
        random_fig, random_stats = self.analyze_random_split(train_df)
        random_fig.write_html(self.output_dir / 'random_split.html', include_plotlyjs='cdn')
        
        # 2. Stratified split analysis
        stratified_fig, stratified_stats = self.analyze_stratified_split(train_df)
        stratified_fig.write_html(self.output_dir / 'stratified_split.html', include_plotlyjs='cdn')
        
        # 3. K-Fold analysis
        kfold_fig, kfold_stats = self.analyze_kfold(train_df)
        kfold_fig.write_html(self.output_dir / 'kfold_split.html', include_plotlyjs='cdn')
        
        # Generate main report
        report_html = f"""
        <html>
        <head>
            <title>Corpus Splitting Analysis Report</title>
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
                .plot-container {{
                    width: 100%;
                    margin: 20px 0;
                }}
                iframe {{
                    border: none;
                    width: 100%;
                    height: 600px;
                }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Corpus Splitting Analysis Report</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <p>Total training samples: {len(train_df)}</p>
                <p>Label distribution:</p>
                <ul>
                    <li>Bearish (0): {sum(train_df['label'] == 0)} samples ({sum(train_df['label'] == 0)/len(train_df)*100:.1f}%)</li>
                    <li>Bullish (1): {sum(train_df['label'] == 1)} samples ({sum(train_df['label'] == 1)/len(train_df)*100:.1f}%)</li>
                    <li>Neutral (2): {sum(train_df['label'] == 2)} samples ({sum(train_df['label'] == 2)/len(train_df)*100:.1f}%)</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Random Split Analysis</h2>
                <p>Analysis of simple random train/validation split:</p>
                <ul>
                    <li>Training set: {random_stats['train_size']} samples</li>
                    <li>Validation set: {random_stats['val_size']} samples</li>
                </ul>
                <div class="plot-container">
                    <iframe src="random_split.html"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>Stratified Split Analysis</h2>
                <p>Analysis of stratified train/validation split (maintaining label proportions):</p>
                <ul>
                    <li>Training set: {stratified_stats['train_size']} samples</li>
                    <li>Validation set: {stratified_stats['val_size']} samples</li>
                </ul>
                <div class="plot-container">
                    <iframe src="stratified_split.html"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>K-Fold Cross-Validation Analysis</h2>
                <p>Analysis of 5-fold stratified cross-validation:</p>
                <table>
                    <tr>
                        <th>Fold</th>
                        <th>Training Size</th>
                        <th>Validation Size</th>
                        <th>Training Distribution</th>
                        <th>Validation Distribution</th>
                    </tr>
                    {''.join(f'''
                    <tr>
                        <td>Fold {stats['fold']}</td>
                        <td>{stats['train_size']}</td>
                        <td>{stats['val_size']}</td>
                        <td>
                            Bearish: {stats['train_dist']['Bearish']:.1f}%<br>
                            Bullish: {stats['train_dist']['Bullish']:.1f}%<br>
                            Neutral: {stats['train_dist']['Neutral']:.1f}%
                        </td>
                        <td>
                            Bearish: {stats['val_dist']['Bearish']:.1f}%<br>
                            Bullish: {stats['val_dist']['Bullish']:.1f}%<br>
                            Neutral: {stats['val_dist']['Neutral']:.1f}%
                        </td>
                    </tr>
                    ''' for stats in kfold_stats)}
                </table>
                <div class="plot-container">
                    <iframe src="kfold_split.html"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Given the class imbalance, stratified splitting is recommended to maintain label proportions</li>
                    <li>K-Fold cross-validation provides more robust evaluation, especially for the minority classes</li>
                    <li>Consider using weighted loss or sampling techniques to handle class imbalance during training</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / 'corpus_splitting_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Report generated at {report_path.absolute()}")

if __name__ == "__main__":
    # Generate corpus splitting report
    splitter = CorpusSplitter()
    splitter.generate_report() 