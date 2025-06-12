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
        try:
            if output_dir is None:
                # Use the workspace root directory
                workspace_root = Path(__file__).resolve().parents[2]
                output_dir = workspace_root / 'results' / 'data_exploration'
            else:
                output_dir = Path(output_dir)
            
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            raise
    
    def analyze_tweet_lengths(self, df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
        """
        Analyze tweet length distribution.
        Returns two figures: one for distributions and one for box plots.
        """
        # Character lengths
        char_lengths = df['text'].str.len()
        # Word lengths
        word_lengths = df['text'].str.split().str.len()
        
        # Calculate statistics
        char_stats = {
            'mean': char_lengths.mean(),
            'median': char_lengths.median(),
            'std': char_lengths.std(),
            'min': char_lengths.min(),
            'max': char_lengths.max()
        }
        
        word_stats = {
            'mean': word_lengths.mean(),
            'median': word_lengths.median(),
            'std': word_lengths.std(),
            'min': word_lengths.min(),
            'max': word_lengths.max()
        }

        # Figure 1: Distributions
        dist_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Character Length Distribution',
                'Word Length Distribution'
            ),
            horizontal_spacing=0.15
        )
        
        # Histogram for character lengths
        dist_fig.add_trace(
            go.Histogram(
                x=char_lengths,
                name='Characters',
                nbinsx=50,
                showlegend=False,
                hovertemplate='Character Length: %{x}<br>Count: %{y}'
            ),
            row=1, col=1
        )
        
        # Histogram for word lengths
        dist_fig.add_trace(
            go.Histogram(
                x=word_lengths,
                name='Words',
                nbinsx=30,
                showlegend=False,
                hovertemplate='Word Count: %{x}<br>Count: %{y}'
            ),
            row=1, col=2
        )
        
        # Add statistics as annotations to distribution figure
        stats_text_char = (
            f"Character Length Statistics:<br>"
            f"Mean: {char_stats['mean']:.1f}<br>"
            f"Median: {char_stats['median']:.1f}<br>"
            f"Std Dev: {char_stats['std']:.1f}<br>"
            f"Range: {char_stats['min']:.0f}-{char_stats['max']:.0f}"
        )
        
        stats_text_word = (
            f"Word Count Statistics:<br>"
            f"Mean: {word_stats['mean']:.1f}<br>"
            f"Median: {word_stats['median']:.1f}<br>"
            f"Std Dev: {word_stats['std']:.1f}<br>"
            f"Range: {word_stats['min']:.0f}-{word_stats['max']:.0f}"
        )
        
        dist_fig.add_annotation(
            text=stats_text_char,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        )
        
        dist_fig.add_annotation(
            text=stats_text_word,
            xref="paper", yref="paper",
            x=0.52, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        )
        
        # Update distribution figure layout
        dist_fig.update_layout(
            title='Tweet Length Distributions',
            template='plotly_white',
            height=500,
            showlegend=False,
            margin=dict(t=120, b=40, l=40, r=40)
        )
        
        # Update axes labels for distribution figure
        dist_fig.update_xaxes(title_text="Number of Characters", row=1, col=1)
        dist_fig.update_xaxes(title_text="Number of Words", row=1, col=2)
        dist_fig.update_yaxes(title_text="Count", row=1, col=1)
        dist_fig.update_yaxes(title_text="Count", row=1, col=2)

        # Figure 2: Box Plots
        box_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Character Length Box Plot',
                'Word Length Box Plot'
            ),
            horizontal_spacing=0.15
        )
        
        # Box plot for character lengths
        box_fig.add_trace(
            go.Box(
                y=char_lengths,
                name='Characters',
                boxpoints='outliers',
                hovertemplate='Character Length<br>Median: %{median}<br>Q1: %{q1}<br>Q3: %{q3}<br>Value: %{y}',
                boxmean=True,  # Add mean marker
                jitter=0.3,
                pointpos=-1.5,
                line_color='rgb(93, 164, 214)'
            ),
            row=1, col=1
        )
        
        # Box plot for word lengths
        box_fig.add_trace(
            go.Box(
                y=word_lengths,
                name='Words',
                boxpoints='outliers',
                hovertemplate='Word Count<br>Median: %{median}<br>Q1: %{q1}<br>Q3: %{q3}<br>Value: %{y}',
                boxmean=True,  # Add mean marker
                jitter=0.3,
                pointpos=-1.5,
                line_color='rgb(255, 144, 14)'
            ),
            row=1, col=2
        )
        
        # Update box plot figure layout
        box_fig.update_layout(
            title='Tweet Length Box Plots',
            template='plotly_white',
            height=600,
            showlegend=False,
            margin=dict(t=100, b=40, l=40, r=40)
        )
        
        # Update axes labels for box plot figure
        box_fig.update_yaxes(title_text="Character Length", row=1, col=1)
        box_fig.update_yaxes(title_text="Word Count", row=1, col=2)
        
        return dist_fig, box_fig
    
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
        try:
            # Load data
            train_df, test_df = load_datasets()
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            
            # 1. Tweet length analysis (now returns two figures)
            dist_fig, box_fig = self.analyze_tweet_lengths(train_df)
            dist_html = dist_fig.to_html(full_html=False, include_plotlyjs='cdn')
            box_html = box_fig.to_html(full_html=False, include_plotlyjs=False)
            
            # 2. Sentiment distribution
            sentiment_fig = self.analyze_sentiment_distribution(train_df)
            sentiment_html = sentiment_fig.to_html(full_html=False, include_plotlyjs=False)
            
            # 3. Word clouds
            wordcloud_all = self.generate_wordcloud(train_df['text'])
            wordcloud_by_sentiment = self.generate_wordcloud(
                train_df['text'],
                by_sentiment=True,
                labels=train_df['label']
            )
            
            # 4. Special patterns analysis
            patterns_fig, pattern_stats = self.analyze_special_patterns(train_df)
            patterns_html = patterns_fig.to_html(full_html=False, include_plotlyjs=False)
            
            # Generate main report
            report_html = f"""
            <html>
            <head>
                <title>Data Exploration Report</title>
                <meta charset="utf-8">
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
                    .plot-container {{
                        width: 100%;
                        margin: 20px 0;
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
                    <div class="plot-container">
                        {dist_html}
                    </div>
                    <div class="plot-container">
                        {box_html}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Sentiment Distribution</h2>
                    <div class="plot-container">
                        {sentiment_html}
                    </div>
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
                    <div class="plot-container">
                        {patterns_html}
                    </div>
                    
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
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

if __name__ == "__main__":
    # Generate data exploration report
    explorer = DataExplorer()
    explorer.generate_report() 