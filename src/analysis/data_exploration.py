"""
Data exploration module that generates an HTML report with visualizations.
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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from utils.data_loader import load_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExplorer:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Use absolute path relative to project root
            project_root = os.path.dirname(os.path.dirname(src_dir))
            output_dir = os.path.join(project_root, 'results', 'data_exploration')
        
        self.output_dir = Path(output_dir)
        logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_tweet_lengths(self, df: pd.DataFrame) -> go.Figure:
        """Analyze tweet length distribution."""
        # Character lengths
        char_lengths = df['text'].str.len()
        # Word lengths
        word_lengths = df['text'].str.split().str.len()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Character Length Distribution', 'Word Length Distribution')
        )
        
        fig.add_trace(
            go.Histogram(x=char_lengths, name='Characters', nbinsx=50),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=word_lengths, name='Words', nbinsx=30),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Tweet Length Distributions',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def analyze_sentiment_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Analyze sentiment label distribution."""
        label_counts = df['label'].value_counts().sort_index()
        label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=[label_names[i] for i in label_counts.index],
                values=label_counts.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Sentiment Distribution',
            template='plotly_white'
        )
        
        return fig
    
    def generate_wordcloud(self, texts: List[str], by_sentiment: bool = False, labels: List[int] = None) -> str:
        """Generate wordcloud and return as base64 image."""
        if by_sentiment and labels is not None:
            # Create wordcloud for each sentiment
            sentiment_texts = {
                0: [], # Bearish
                1: [], # Bullish
                2: []  # Neutral
            }
            for text, label in zip(texts, labels):
                sentiment_texts[label].append(text)
            
            # Create subplot with three wordclouds
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            sentiment_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
            
            for i, (sentiment, texts) in enumerate(sentiment_texts.items()):
                text = ' '.join(texts)
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white'
                ).generate(text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].axis('off')
                axes[i].set_title(sentiment_names[sentiment])
            
            plt.tight_layout()
        else:
            # Create single wordcloud
            text = ' '.join(texts)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white'
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
        
        # Convert plot to base64 image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode()
        plt.close()
        
        return img_data
    
    def analyze_special_patterns(self, df: pd.DataFrame) -> Tuple[go.Figure, Dict]:
        """Analyze special patterns in tweets (hashtags, mentions, URLs, etc.)."""
        patterns = {
            'hashtags': r'#\w+',
            'mentions': r'@\w+',
            'urls': r'https?://\S+|www\.\S+',
            'cashtags': r'\$[A-Za-z]+',
            'numbers': r'\d+\.?\d*'
        }
        
        # Count occurrences
        pattern_counts = {}
        for name, pattern in patterns.items():
            counts = df['text'].str.count(pattern)
            pattern_counts[name] = {
                'total': counts.sum(),
                'tweets_with': (counts > 0).sum(),
                'avg_per_tweet': counts.mean()
            }
        
        # Create visualization
        fig = go.Figure()
        
        # Add bars for total counts
        fig.add_trace(go.Bar(
            name='Total Occurrences',
            x=list(pattern_counts.keys()),
            y=[stats['total'] for stats in pattern_counts.values()],
            text=[stats['total'] for stats in pattern_counts.values()],
            textposition='auto'
        ))
        
        # Add bars for tweets containing pattern
        fig.add_trace(go.Bar(
            name='Tweets Containing Pattern',
            x=list(pattern_counts.keys()),
            y=[stats['tweets_with'] for stats in pattern_counts.values()],
            text=[stats['tweets_with'] for stats in pattern_counts.values()],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Special Pattern Analysis',
            xaxis_title='Pattern Type',
            yaxis_title='Count',
            template='plotly_white',
            barmode='group'
        )
        
        return fig, pattern_counts
    
    def generate_report(self) -> None:
        """Generate HTML report with data exploration analysis."""
        # Load data
        train_df, test_df = load_datasets()
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # 1. Tweet length analysis
        length_fig = self.analyze_tweet_lengths(train_df)
        length_fig.write_html(self.output_dir / 'tweet_lengths.html')
        
        # 2. Sentiment distribution
        sentiment_fig = self.analyze_sentiment_distribution(train_df)
        sentiment_fig.write_html(self.output_dir / 'sentiment_distribution.html')
        
        # 3. Word clouds
        wordcloud_all = self.generate_wordcloud(train_df['text'])
        wordcloud_by_sentiment = self.generate_wordcloud(
            train_df['text'],
            by_sentiment=True,
            labels=train_df['label']
        )
        
        # 4. Special patterns analysis
        patterns_fig, pattern_stats = self.analyze_special_patterns(train_df)
        patterns_fig.write_html(self.output_dir / 'special_patterns.html')
        
        # Generate main report
        report_html = f"""
        <html>
        <head>
            <title>Data Exploration Report</title>
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
                .wordcloud {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .wordcloud img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>Data Exploration Report</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <p>Training samples: {len(train_df)}</p>
                <p>Test samples: {len(test_df)}</p>
                
                <h3>Basic Statistics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Average tweet length (characters)</td>
                        <td>{train_df['text'].str.len().mean():.1f}</td>
                    </tr>
                    <tr>
                        <td>Average tweet length (words)</td>
                        <td>{train_df['text'].str.split().str.len().mean():.1f}</td>
                    </tr>
                    <tr>
                        <td>Vocabulary size</td>
                        <td>{len(set(' '.join(train_df['text']).split()))}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Tweet Length Analysis</h2>
                <iframe src="./tweet_lengths.html" width="100%" height="500" frameborder="0"></iframe>
            </div>
            
            <div class="section">
                <h2>Sentiment Distribution</h2>
                <iframe src="./sentiment_distribution.html" width="100%" height="500" frameborder="0"></iframe>
            </div>
            
            <div class="section">
                <h2>Word Clouds</h2>
                <h3>Overall Word Cloud</h3>
                <div class="wordcloud">
                    <img src="data:image/png;base64,{wordcloud_all}" alt="Word Cloud">
                </div>
                
                <h3>Word Clouds by Sentiment</h3>
                <div class="wordcloud">
                    <img src="data:image/png;base64,{wordcloud_by_sentiment}" alt="Sentiment Word Clouds">
                </div>
            </div>
            
            <div class="section">
                <h2>Special Pattern Analysis</h2>
                <iframe src="./special_patterns.html" width="100%" height="500" frameborder="0"></iframe>
                
                <h3>Pattern Statistics</h3>
                <table>
                    <tr>
                        <th>Pattern</th>
                        <th>Total Occurrences</th>
                        <th>Tweets Containing</th>
                        <th>Average per Tweet</th>
                    </tr>
                    {''.join(f'''
                    <tr>
                        <td>{pattern}</td>
                        <td>{stats['total']}</td>
                        <td>{stats['tweets_with']} ({stats['tweets_with']/len(train_df)*100:.1f}%)</td>
                        <td>{stats['avg_per_tweet']:.2f}</td>
                    </tr>
                    ''' for pattern, stats in pattern_stats.items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <ul>
                    <li>Dataset is imbalanced with {sum(train_df['label'] == 2)/len(train_df)*100:.1f}% neutral tweets</li>
                    <li>Average tweet length is {train_df['text'].str.split().str.len().mean():.1f} words</li>
                    <li>{pattern_stats['hashtags']['tweets_with']/len(train_df)*100:.1f}% of tweets contain hashtags</li>
                    <li>{pattern_stats['cashtags']['tweets_with']/len(train_df)*100:.1f}% of tweets contain cashtags ($)</li>
                    <li>URLs appear in {pattern_stats['urls']['tweets_with']/len(train_df)*100:.1f}% of tweets</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / 'data_exploration_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Report generated at {report_path.absolute()}")

if __name__ == "__main__":
    # Generate data exploration report
    explorer = DataExplorer()
    explorer.generate_report() 