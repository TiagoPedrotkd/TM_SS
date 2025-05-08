"""
Feature analysis module that generates an HTML report comparing different feature extraction methods.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
import sys
import os
from typing import List, Dict, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from tqdm import tqdm

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from utils.data_loader import load_datasets
from preprocessing.text_processor import TextPreprocessor
from features.feature_extraction import create_feature_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Use absolute path relative to project root
            project_root = os.path.dirname(os.path.dirname(src_dir))
            output_dir = os.path.join(project_root, 'results', 'feature_analysis')
        
        self.output_dir = Path(output_dir)
        logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = TextPreprocessor()
    
    def visualize_feature_space(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        method: str,
        reduction: str = 'pca'
    ) -> go.Figure:
        """
        Visualize feature space using dimensionality reduction.
        
        Args:
            features: Feature vectors
            labels: Class labels
            texts: Original preprocessed texts
            method: Feature extraction method name
            reduction: Dimensionality reduction method ('pca' or 'tsne')
        """
        # Apply dimensionality reduction
        if reduction == 'pca':
            reducer = PCA(n_components=3)  # Changed to 3 components
        else:  # t-SNE
            reducer = TSNE(n_components=3, random_state=42)  # Changed to 3 components
        
        reduced_features = reducer.fit_transform(features)
        
        # Create 3D scatter plot
        label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        fig = go.Figure()
        
        for label in sorted(np.unique(labels)):
            mask = labels == label
            
            # Create hover text with original tweets
            hover_texts = [
                f"Text: {text}<br>"
                f"Label: {label_names[label]}"
                for text in np.array(texts)[mask]
            ]
            
            fig.add_trace(go.Scatter3d(
                x=reduced_features[mask, 0],
                y=reduced_features[mask, 1],
                z=reduced_features[mask, 2],
                mode='markers',
                name=label_names[label],
                marker=dict(
                    size=4,
                    opacity=0.7
                ),
                text=hover_texts,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'Feature Space Visualization ({method} + {reduction.upper()})',
            scene=dict(
                xaxis_title=f'{reduction.upper()} 1',
                yaxis_title=f'{reduction.upper()} 2',
                zaxis_title=f'{reduction.upper()} 3'
            ),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            # Add camera settings for better initial view
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            # Increase size for better visualization
            height=800
        )
        
        return fig
    
    def analyze_feature_statistics(
        self,
        features: np.ndarray,
        method: str
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        """Analyze feature statistics."""
        # Calculate statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_sparsity = np.mean(features == 0, axis=0)
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Feature Means', 'Feature Standard Deviations', 'Feature Sparsity')
        )
        
        fig.add_trace(
            go.Histogram(x=feature_means, nbinsx=50),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=feature_stds, nbinsx=50),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(x=feature_sparsity, nbinsx=50),
            row=1, col=3
        )
        
        fig.update_layout(
            title=f'Feature Statistics ({method})',
            template='plotly_white',
            showlegend=False
        )
        
        stats = {
            'mean_range': (float(np.min(feature_means)), float(np.max(feature_means))),
            'std_range': (float(np.min(feature_stds)), float(np.max(feature_stds))),
            'sparsity': float(np.mean(feature_sparsity)),
            'dimension': features.shape[1]
        }
        
        return fig, stats
    
    def generate_report(self) -> None:
        """Generate HTML report comparing feature extraction methods."""
        # Load and preprocess data
        train_df, _ = load_datasets()
        
        logger.info("Preprocessing texts...")
        processed_texts = [
            self.preprocessor.preprocess(text)
            for text in tqdm(train_df['text'], desc="Preprocessing")
        ]
        
        # Extract features using different methods
        methods = {
            'bow': {'max_features': 5000, 'ngram_range': (1, 2)},
            'word2vec': {'vector_size': 100, 'window': 5},
            'transformer': {'model_name': 'bert-base-uncased', 'max_length': 128}
        }
        
        features = {}
        method_stats = {}
        
        for method, params in methods.items():
            logger.info(f"\nExtracting features using {method}...")
            extractor = create_feature_extractor(method=method, **params)
            features[method] = extractor.fit_transform(processed_texts)
            
            # Save feature space visualization
            pca_fig = self.visualize_feature_space(
                features[method],
                train_df['label'],
                processed_texts,  # Added processed texts for hover info
                method,
                reduction='pca'
            )
            pca_fig.write_html(self.output_dir / f'{method}_pca.html')
            
            tsne_fig = self.visualize_feature_space(
                features[method],
                train_df['label'],
                processed_texts,  # Added processed texts for hover info
                method,
                reduction='tsne'
            )
            tsne_fig.write_html(self.output_dir / f'{method}_tsne.html')
            
            # Analyze feature statistics
            stats_fig, stats = self.analyze_feature_statistics(features[method], method)
            stats_fig.write_html(self.output_dir / f'{method}_stats.html')
            method_stats[method] = stats
            
            # Save extractor configuration
            extractor.save(method)
        
        # Generate main report
        report_html = f"""
        <html>
        <head>
            <title>Feature Analysis Report</title>
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
            <h1>Feature Analysis Report</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <p>Number of samples: {len(train_df)}</p>
                <p>Label distribution:</p>
                <ul>
                    <li>Bearish (0): {sum(train_df['label'] == 0)} samples ({sum(train_df['label'] == 0)/len(train_df)*100:.1f}%)</li>
                    <li>Bullish (1): {sum(train_df['label'] == 1)} samples ({sum(train_df['label'] == 1)/len(train_df)*100:.1f}%)</li>
                    <li>Neutral (2): {sum(train_df['label'] == 2)} samples ({sum(train_df['label'] == 2)/len(train_df)*100:.1f}%)</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Feature Extraction Methods</h2>
                
                <h3>Bag of Words (TF-IDF)</h3>
                <p>Statistics:</p>
                <ul>
                    <li>Feature dimension: {method_stats['bow']['dimension']}</li>
                    <li>Mean range: [{method_stats['bow']['mean_range'][0]:.4f}, {method_stats['bow']['mean_range'][1]:.4f}]</li>
                    <li>Standard deviation range: [{method_stats['bow']['std_range'][0]:.4f}, {method_stats['bow']['std_range'][1]:.4f}]</li>
                    <li>Sparsity: {method_stats['bow']['sparsity']*100:.1f}%</li>
                </ul>
                <iframe src="./bow_stats.html" width="100%" height="400" frameborder="0"></iframe>
                <h4>Feature Space Visualization</h4>
                <iframe src="./bow_pca.html" width="100%" height="400" frameborder="0"></iframe>
                <iframe src="./bow_tsne.html" width="100%" height="400" frameborder="0"></iframe>
                
                <h3>Word2Vec</h3>
                <p>Statistics:</p>
                <ul>
                    <li>Feature dimension: {method_stats['word2vec']['dimension']}</li>
                    <li>Mean range: [{method_stats['word2vec']['mean_range'][0]:.4f}, {method_stats['word2vec']['mean_range'][1]:.4f}]</li>
                    <li>Standard deviation range: [{method_stats['word2vec']['std_range'][0]:.4f}, {method_stats['word2vec']['std_range'][1]:.4f}]</li>
                    <li>Sparsity: {method_stats['word2vec']['sparsity']*100:.1f}%</li>
                </ul>
                <iframe src="./word2vec_stats.html" width="100%" height="400" frameborder="0"></iframe>
                <h4>Feature Space Visualization</h4>
                <iframe src="./word2vec_pca.html" width="100%" height="400" frameborder="0"></iframe>
                <iframe src="./word2vec_tsne.html" width="100%" height="400" frameborder="0"></iframe>
                
                <h3>Transformer (BERT)</h3>
                <p>Statistics:</p>
                <ul>
                    <li>Feature dimension: {method_stats['transformer']['dimension']}</li>
                    <li>Mean range: [{method_stats['transformer']['mean_range'][0]:.4f}, {method_stats['transformer']['mean_range'][1]:.4f}]</li>
                    <li>Standard deviation range: [{method_stats['transformer']['std_range'][0]:.4f}, {method_stats['transformer']['std_range'][1]:.4f}]</li>
                    <li>Sparsity: {method_stats['transformer']['sparsity']*100:.1f}%</li>
                </ul>
                <iframe src="./transformer_stats.html" width="100%" height="400" frameborder="0"></iframe>
                <h4>Feature Space Visualization</h4>
                <iframe src="./transformer_pca.html" width="100%" height="400" frameborder="0"></iframe>
                <iframe src="./transformer_tsne.html" width="100%" height="400" frameborder="0"></iframe>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>BoW/TF-IDF provides interpretable features but high sparsity</li>
                    <li>Word2Vec captures semantic relationships with dense representations</li>
                    <li>BERT provides rich contextual embeddings but higher computational cost</li>
                    <li>Consider using BERT for best performance, or Word2Vec for a good balance</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / 'feature_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Report generated at {report_path.absolute()}")

if __name__ == "__main__":
    # Generate feature analysis report
    analyzer = FeatureAnalyzer()
    analyzer.generate_report() 