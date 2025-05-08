"""
Preprocessing analysis module that generates an HTML report with visualizations.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
from pathlib import Path
import re
from typing import List, Dict, Tuple
import logging
import sys
import os

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from preprocessing.text_processor import TextPreprocessor
from utils.data_loader import load_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingAnalyzer:
    def __init__(self, output_dir: str = None):
        self.preprocessor = TextPreprocessor()
        
        if output_dir is None:
            # Use absolute path relative to project root
            project_root = os.path.dirname(os.path.dirname(src_dir))
            output_dir = os.path.join(project_root, 'results', 'preprocessing')
        
        self.output_dir = Path(output_dir)
        logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_text_lengths(self, texts: List[str]) -> go.Figure:
        """Create distribution plot of text lengths."""
        lengths = [len(text.split()) for text in texts]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=lengths,
            nbinsx=50,
            name='Text Length Distribution'
        ))
        
        fig.update_layout(
            title='Distribution of Text Lengths (words)',
            xaxis_title='Number of Words',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        return fig
    
    def analyze_common_terms(
        self,
        texts: List[str],
        top_n: int = 20,
        by_label: bool = False,
        labels: List[int] = None
    ) -> go.Figure:
        """Create bar chart of most common terms."""
        if by_label and labels is not None:
            # Separate analysis by sentiment label
            label_texts = {
                0: [], # Bearish
                1: [], # Bullish
                2: []  # Neutral
            }
            for text, label in zip(texts, labels):
                label_texts[label].append(text)
            
            # Process each category
            label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
            fig = go.Figure()
            
            for label, texts in label_texts.items():
                words = ' '.join(texts).split()
                word_freq = Counter(words).most_common(top_n)
                
                fig.add_trace(go.Bar(
                    name=label_names[label],
                    x=[word for word, _ in word_freq],
                    y=[freq for _, freq in word_freq],
                    text=[freq for _, freq in word_freq],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title=f'Top {top_n} Most Common Terms by Sentiment',
                xaxis_title='Term',
                yaxis_title='Frequency',
                template='plotly_white',
                barmode='group'
            )
        else:
            # Overall analysis
            words = ' '.join(texts).split()
            word_freq = Counter(words).most_common(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[word for word, _ in word_freq],
                    y=[freq for _, freq in word_freq],
                    text=[freq for _, freq in word_freq],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f'Top {top_n} Most Common Terms',
                xaxis_title='Term',
                yaxis_title='Frequency',
                template='plotly_white'
            )
        
        return fig
    
    def analyze_preprocessing_impact(self, texts: List[str]) -> Tuple[go.Figure, Dict]:
        """Analyze impact of each preprocessing step."""
        steps = [
            'urls', 'mentions', 'hashtags', 'numbers',
            'punctuation', 'stopwords', 'lemmatize'
        ]
        
        # Analyze each step's impact
        impacts = {}
        processed_texts = texts.copy()
        
        for step in steps:
            # Process texts with only this step
            step_texts = [
                self.preprocessor.preprocess(text, steps=[step])
                for text in texts
            ]
            
            # Calculate average reduction in length
            original_lengths = [len(text.split()) for text in texts]
            processed_lengths = [len(text.split()) for text in step_texts]
            
            reduction = np.mean([
                (o - p) / o * 100 if o > 0 else 0
                for o, p in zip(original_lengths, processed_lengths)
            ])
            
            impacts[step] = {
                'reduction_percent': reduction,
                'avg_tokens_removed': np.mean([
                    o - p for o, p in zip(original_lengths, processed_lengths)
                ])
            }
        
        # Create impact visualization
        fig = go.Figure(data=[
            go.Bar(
                x=list(impacts.keys()),
                y=[impact['reduction_percent'] for impact in impacts.values()],
                text=[f"{impact['reduction_percent']:.1f}%" for impact in impacts.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Impact of Preprocessing Steps',
            xaxis_title='Preprocessing Step',
            yaxis_title='Average Text Length Reduction (%)',
            template='plotly_white'
        )
        
        return fig, impacts
    
    def generate_report(self) -> None:
        """Generate HTML report with preprocessing analysis."""
        # Load data
        train_df, test_df = load_datasets()
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        train_processed = [
            self.preprocessor.preprocess(text)
            for text in train_df['text']
        ]
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # 1. Text length distribution
        length_fig = self.analyze_text_lengths(train_processed)
        length_fig.write_html(self.output_dir / 'text_lengths.html')
        
        # 2. Common terms overall
        terms_fig = self.analyze_common_terms(train_processed)
        terms_fig.write_html(self.output_dir / 'common_terms.html')
        
        # 3. Common terms by sentiment
        terms_by_sentiment_fig = self.analyze_common_terms(
            train_processed,
            by_label=True,
            labels=train_df['label']
        )
        terms_by_sentiment_fig.write_html(self.output_dir / 'terms_by_sentiment.html')
        
        # 4. Preprocessing impact
        impact_fig, impacts = self.analyze_preprocessing_impact(train_df['text'][:1000])
        impact_fig.write_html(self.output_dir / 'preprocessing_impact.html')
        
        # Generate main report
        report_html = f"""
        <html>
        <head>
            <title>Text Preprocessing Analysis Report</title>
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
            <h1>Text Preprocessing Analysis Report</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <p>Training samples: {len(train_df)}</p>
                <p>Test samples: {len(test_df)}</p>
                <p>Label distribution:</p>
                <ul>
                    <li>Bearish (0): {sum(train_df['label'] == 0)} samples ({sum(train_df['label'] == 0)/len(train_df)*100:.1f}%)</li>
                    <li>Bullish (1): {sum(train_df['label'] == 1)} samples ({sum(train_df['label'] == 1)/len(train_df)*100:.1f}%)</li>
                    <li>Neutral (2): {sum(train_df['label'] == 2)} samples ({sum(train_df['label'] == 2)/len(train_df)*100:.1f}%)</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Text Length Analysis</h2>
                <iframe src="./text_lengths.html" width="100%" height="600" frameborder="0"></iframe>
            </div>
            
            <div class="section">
                <h2>Common Terms Analysis</h2>
                <h3>Overall Most Common Terms</h3>
                <iframe src="./common_terms.html" width="100%" height="600" frameborder="0"></iframe>
                
                <h3>Terms by Sentiment</h3>
                <iframe src="./terms_by_sentiment.html" width="100%" height="600" frameborder="0"></iframe>
            </div>
            
            <div class="section">
                <h2>Preprocessing Impact Analysis</h2>
                <iframe src="./preprocessing_impact.html" width="100%" height="600" frameborder="0"></iframe>
                
                <h3>Detailed Impact Statistics</h3>
                <table>
                    <tr>
                        <th>Step</th>
                        <th>Average Tokens Removed</th>
                        <th>Text Reduction (%)</th>
                    </tr>
                    {''.join(f'''
                    <tr>
                        <td>{step}</td>
                        <td>{stats['avg_tokens_removed']:.2f}</td>
                        <td>{stats['reduction_percent']:.1f}%</td>
                    </tr>
                    ''' for step, stats in impacts.items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Preprocessing Steps Applied</h2>
                <ul>
                    <li>URL removal</li>
                    <li>Twitter mentions removal</li>
                    <li>Hashtag processing (keeping text)</li>
                    <li>Number handling (preserving prices)</li>
                    <li>Punctuation removal</li>
                    <li>Stopword removal (with finance-specific adjustments)</li>
                    <li>Lemmatization</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / 'preprocessing_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Report generated at {report_path.absolute()}")

if __name__ == "__main__":
    # Generate preprocessing analysis report
    analyzer = PreprocessingAnalyzer()
    analyzer.generate_report() 