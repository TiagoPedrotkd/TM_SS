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
import emoji
from bs4 import BeautifulSoup
import contractions
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from preprocessing.text_processor import TextPreprocessor
from utils.data_loader import load_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingAnalyzer:
    def __init__(self, output_dir: str = None):
        self.preprocessor = TextPreprocessor()
        
        if output_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]
            output_dir = project_root / 'results' / 'preprocessing'
        
        self.output_dir = Path(output_dir)
        logger.info(f"Output directory set to: {self.output_dir.absolute()}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for images
        self.img_dir = self.output_dir / 'images'
        self.img_dir.mkdir(exist_ok=True)
    
    def get_plotly_figure_html(self, fig: go.Figure) -> str:
        """Helper method to get HTML representation of a plotly figure."""
        fig.update_layout(
            template='plotly_white',
            height=600,
            width=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def analyze_text_lengths(self, texts: List[str], original_texts: List[str] = None) -> go.Figure:
        """Create distribution plot of text lengths."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Original Text Lengths', 'Processed Text Lengths'))
        
        # Plot original lengths if provided
        if original_texts:
            original_lengths = [len(text.split()) for text in original_texts]
            fig.add_trace(
                go.Histogram(x=original_lengths, nbinsx=50, name='Original'),
                row=1, col=1
            )
        
        # Plot processed lengths
        processed_lengths = [len(text.split()) for text in texts]
        fig.add_trace(
            go.Histogram(x=processed_lengths, nbinsx=50, name='Processed'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Text Length Distribution Comparison',
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Number of Words', row=1, col=1)
        fig.update_xaxes(title_text='Number of Words', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        
        return fig
    
    def analyze_vocabulary(self, texts: List[str], original_texts: List[str] = None) -> Tuple[go.Figure, Dict]:
        """Analyze vocabulary before and after preprocessing."""
        stats = {}
        
        # Analyze original texts
        if original_texts:
            original_words = ' '.join(original_texts).lower().split()
            original_vocab = set(original_words)
            stats['original'] = {
                'total_words': len(original_words),
                'unique_words': len(original_vocab),
                'avg_word_length': np.mean([len(word) for word in original_words])
            }
        
        # Analyze processed texts
        processed_words = ' '.join(texts).lower().split()
        processed_vocab = set(processed_words)
        stats['processed'] = {
            'total_words': len(processed_words),
            'unique_words': len(processed_vocab),
            'avg_word_length': np.mean([len(word) for word in processed_words])
        }
        
        # Create comparison visualization
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=('Total Words', 'Unique Words', 'Average Word Length'))
        
        if original_texts:
            fig.add_trace(
                go.Bar(name='Original', x=['Total Words'], y=[stats['original']['total_words']]),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Original', x=['Unique Words'], y=[stats['original']['unique_words']]),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name='Original', x=['Avg Length'], y=[stats['original']['avg_word_length']]),
                row=1, col=3
            )
        
        fig.add_trace(
            go.Bar(name='Processed', x=['Total Words'], y=[stats['processed']['total_words']]),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Processed', x=['Unique Words'], y=[stats['processed']['unique_words']]),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Processed', x=['Avg Length'], y=[stats['processed']['avg_word_length']]),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Vocabulary Analysis Comparison',
            template='plotly_white',
            showlegend=True,
            barmode='group'
        )
        
        return fig, stats
    
    def generate_wordclouds(self, texts: List[str], original_texts: List[str] = None) -> None:
        """Generate word clouds for original and processed texts."""
        try:
            if original_texts:
                # Generate word cloud for original texts
                original_text = ' '.join(original_texts)
                wordcloud_original = WordCloud(width=800, height=400, background_color='white').generate(original_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud_original, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - Original Texts')
                plt.savefig(self.img_dir / 'wordcloud_original.png')
                plt.close()
            
            # Generate word cloud for processed texts
            processed_text = ' '.join(texts)
            wordcloud_processed = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_processed, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Processed Texts')
            plt.savefig(self.img_dir / 'wordcloud_processed.png')
            plt.close()
        except Exception as e:
            logger.warning(f"Error generating word clouds: {str(e)}")
    
    def analyze_word_frequency_distribution(self, texts: List[str], original_texts: List[str] = None) -> go.Figure:
        """Analyze word frequency distribution before and after preprocessing."""
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Original Text', 'Processed Text'))
        
        if original_texts:
            # Analyze original texts
            original_words = ' '.join(original_texts).lower().split()
            original_fdist = FreqDist(original_words)
            original_freq = pd.Series(original_fdist).sort_values(ascending=False)
            
            fig.add_trace(
                go.Scatter(x=list(range(len(original_freq))), y=original_freq.values.tolist(),
                          mode='lines', name='Original'),
                row=1, col=1
            )
        
        # Analyze processed texts
        processed_words = ' '.join(texts).lower().split()
        processed_fdist = FreqDist(processed_words)
        processed_freq = pd.Series(processed_fdist).sort_values(ascending=False)
        
        fig.add_trace(
            go.Scatter(x=list(range(len(processed_freq))), y=processed_freq.values.tolist(),
                      mode='lines', name='Processed'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Word Frequency Distribution',
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Word Rank', row=1, col=1)
        fig.update_xaxes(title_text='Word Rank', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=1, col=2)
        
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
    
    def analyze_special_characters(self, texts: List[str]) -> Tuple[go.Figure, Dict]:
        """Analyze special characters in texts."""
        stats = {
            'emojis': 0,
            'html_tags': 0,
            'urls': 0,
            'mentions': 0,
            'hashtags': 0,
            'contractions': 0
        }
        
        for text in texts:
            # Count emojis
            stats['emojis'] += len(emoji.emoji_list(text))
            
            # Count HTML tags
            soup = BeautifulSoup(text, "html.parser")
            stats['html_tags'] += len(soup.find_all())
            
            # Count URLs
            stats['urls'] += len(re.findall(r'https?://\S+|www\.\S+', text))
            
            # Count mentions
            stats['mentions'] += len(re.findall(r'@\w+', text))
            
            # Count hashtags
            stats['hashtags'] += len(re.findall(r'#\w+', text))
            
            # Count contractions
            stats['contractions'] += len([word for word in text.split() if "'" in word])
        
        # Create visualization
        fig = go.Figure(data=[
            go.Bar(
                x=list(stats.keys()),
                y=list(stats.values()),
                text=list(stats.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Special Characters Analysis',
            xaxis_title='Character Type',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        return fig, stats
    
    def analyze_preprocessing_impact(self, texts: List[str]) -> Tuple[go.Figure, Dict]:
        """Analyze impact of each preprocessing step."""
        steps = [
            'html', 'urls', 'mentions', 'hashtags', 'numbers',
            'punctuation', 'emojis', 'contractions', 'unicode',
            'stopwords', 'lemmatize'
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
    
    def compare_common_terms_by_label(self, original_texts: list, processed_texts: list, labels: list, top_n: int = 20):
        """Generate side-by-side bar charts of top N terms per label before and after preprocessing."""
        label_names = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        colors = {'Original': '#1f77b4', 'Processed': '#ff7f0e'}  # Consistent colors
        
        # Create subplots with 3 rows (one per sentiment) and 2 columns (Original vs Processed)
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f"Original Text - {label_names[0]}", f"Processed Text - {label_names[0]}",  # Row 1: Bearish
                f"Original Text - {label_names[1]}", f"Processed Text - {label_names[1]}",  # Row 2: Bullish
                f"Original Text - {label_names[2]}", f"Processed Text - {label_names[2]}"   # Row 3: Neutral
            ],
            vertical_spacing=0.15,  # Increased spacing
            horizontal_spacing=0.08,  # Increased spacing
        )
        
        # Update subplot titles font size
        for annotation in fig.layout.annotations:
            annotation.font.size = 16
        
        for i, label in enumerate([0, 1, 2]):  # Process labels in order: Bearish, Bullish, Neutral
            # Original text analysis (col 1)
            orig_texts = [t for t, l in zip(original_texts, labels) if l == label]
            orig_words = ' '.join(orig_texts).lower().split()
            orig_freq = Counter(orig_words).most_common(top_n)
            fig.add_trace(
                go.Bar(
                    x=[w for w, _ in orig_freq],
                    y=[c for _, c in orig_freq],
                    name=f'Original - {label_names[label]}',
                    marker_color=colors['Original'],
                    showlegend=(i == 0)  # Show legend only for first row
                ),
                row=i+1, col=1
            )
            
            # Processed text analysis (col 2)
            proc_texts = [t for t, l in zip(processed_texts, labels) if l == label]
            proc_words = ' '.join(proc_texts).lower().split()
            proc_freq = Counter(proc_words).most_common(top_n)
            fig.add_trace(
                go.Bar(
                    x=[w for w, _ in proc_freq],
                    y=[c for _, c in proc_freq],
                    name=f'Processed - {label_names[label]}',
                    marker_color=colors['Processed'],
                    showlegend=(i == 0)  # Show legend only for first row
                ),
                row=i+1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1500,  # Increased height
            width=1400,   # Increased width
            title={
                'text': 'Most Common Terms by Sentiment: Before vs After Preprocessing',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}  # Larger title
            },
            showlegend=True,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 14}  # Larger legend text
            },
            template='plotly_white'
        )
        
        # Update axes for each subplot
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(
                    title_text='Terms',
                    title_font={'size': 14},  # Larger axis title
                    tickangle=45,
                    row=i,
                    col=j,
                    tickfont={'size': 12}  # Larger tick labels
                )
                fig.update_yaxes(
                    title_text='Frequency',
                    title_font={'size': 14},  # Larger axis title
                    row=i,
                    col=j,
                    tickfont={'size': 12}  # Larger tick labels
                )
        
        return fig
    
    def generate_report(self) -> None:
        """Generate HTML report with preprocessing analysis."""
        try:
            # Load data
            train_df, test_df = load_datasets()
            
            # Store original texts
            original_texts = train_df['text'].tolist()
            
            # Preprocess texts
            logger.info("Preprocessing texts...")
            train_processed = [
                self.preprocessor.preprocess(text)
                for text in original_texts
            ]
            labels = train_df['label'].tolist()
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            
            # Generate all figures
            length_fig = self.analyze_text_lengths(train_processed, original_texts)
            vocab_fig, vocab_stats = self.analyze_vocabulary(train_processed, original_texts)
            self.generate_wordclouds(train_processed, original_texts)
            freq_fig = self.analyze_word_frequency_distribution(train_processed, original_texts)
            terms_fig = self.analyze_common_terms(train_processed)
            terms_by_sentiment_fig = self.analyze_common_terms(
                train_processed,
                by_label=True,
                labels=train_df['label']
            )
            compare_terms_fig = self.compare_common_terms_by_label(
                original_texts, train_processed, labels, top_n=20
            )
            special_chars_fig, special_chars_stats = self.analyze_special_characters(original_texts)
            impact_fig, impacts = self.analyze_preprocessing_impact(original_texts[:1000])
            
            # Generate main report
            report_html = f"""
            <html>
            <head>
                <title>Text Preprocessing Analysis Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .section {{
                        margin-bottom: 40px;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f5f5f5;
                    }}
                    .comparison {{
                        display: flex;
                        justify-content: space-between;
                        margin: 20px 0;
                        flex-wrap: wrap;
                    }}
                    .comparison img {{
                        width: 48%;
                        min-width: 300px;
                        margin: 10px 0;
                        border-radius: 4px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .plot-container {{
                        width: 100%;
                        margin: 20px 0;
                        background: white;
                        padding: 10px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    h1, h2, h3 {{
                        color: #333;
                    }}
                    .stats-highlight {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 4px;
                        margin: 10px 0;
                    }}
                    .metric-card {{
                        background: white;
                        padding: 15px;
                        border-radius: 4px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        margin: 10px 0;
                    }}
                    .metric-value {{
                        font-size: 1.2em;
                        font-weight: bold;
                        color: #2c5282;
                    }}
                </style>
            </head>
            <body>
                <h1>Text Preprocessing Analysis Report</h1>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="stats-highlight">
                        <div class="metric-card">
                            <p>Training samples: <span class="metric-value">{len(train_df)}</span></p>
                            <p>Test samples: <span class="metric-value">{len(test_df)}</span></p>
                        </div>
                        <h3>Label Distribution</h3>
                        <ul>
                            <li>Bearish (0): <span class="metric-value">{sum(train_df['label'] == 0)} samples ({sum(train_df['label'] == 0)/len(train_df)*100:.1f}%)</span></li>
                            <li>Bullish (1): <span class="metric-value">{sum(train_df['label'] == 1)} samples ({sum(train_df['label'] == 1)/len(train_df)*100:.1f}%)</span></li>
                            <li>Neutral (2): <span class="metric-value">{sum(train_df['label'] == 2)} samples ({sum(train_df['label'] == 2)/len(train_df)*100:.1f}%)</span></li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Text Length Analysis</h2>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(length_fig)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Vocabulary Analysis</h2>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(vocab_fig)}
                    </div>
                    
                    <h3>Vocabulary Statistics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Original</th>
                            <th>Processed</th>
                            <th>Change (%)</th>
                        </tr>
                        <tr>
                            <td>Total Words</td>
                            <td>{vocab_stats['original']['total_words']:,.0f}</td>
                            <td>{vocab_stats['processed']['total_words']:,.0f}</td>
                            <td>{((vocab_stats['processed']['total_words'] - vocab_stats['original']['total_words']) / vocab_stats['original']['total_words'] * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>Unique Words</td>
                            <td>{vocab_stats['original']['unique_words']:,.0f}</td>
                            <td>{vocab_stats['processed']['unique_words']:,.0f}</td>
                            <td>{((vocab_stats['processed']['unique_words'] - vocab_stats['original']['unique_words']) / vocab_stats['original']['unique_words'] * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>Average Word Length</td>
                            <td>{vocab_stats['original']['avg_word_length']:.2f}</td>
                            <td>{vocab_stats['processed']['avg_word_length']:.2f}</td>
                            <td>{((vocab_stats['processed']['avg_word_length'] - vocab_stats['original']['avg_word_length']) / vocab_stats['original']['avg_word_length'] * 100):.1f}%</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Word Clouds</h2>
                    <div class="comparison">
                        <img src="data:image/png;base64,{self.get_base64_image('wordcloud_original.png')}" alt="Original Word Cloud">
                        <img src="data:image/png;base64,{self.get_base64_image('wordcloud_processed.png')}" alt="Processed Word Cloud">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Word Frequency Distribution</h2>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(freq_fig)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Special Characters Analysis</h2>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(special_chars_fig)}
                    </div>
                    
                    <h3>Special Characters Statistics</h3>
                    <table>
                        <tr>
                            <th>Character Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                        <tr>
                            <td>Emojis</td>
                            <td>{special_chars_stats['emojis']:,}</td>
                            <td>{(special_chars_stats['emojis'] / sum(special_chars_stats.values()) * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>HTML Tags</td>
                            <td>{special_chars_stats['html_tags']:,}</td>
                            <td>{(special_chars_stats['html_tags'] / sum(special_chars_stats.values()) * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>URLs</td>
                            <td>{special_chars_stats['urls']:,}</td>
                            <td>{(special_chars_stats['urls'] / sum(special_chars_stats.values()) * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>Mentions (@)</td>
                            <td>{special_chars_stats['mentions']:,}</td>
                            <td>{(special_chars_stats['mentions'] / sum(special_chars_stats.values()) * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>Hashtags (#)</td>
                            <td>{special_chars_stats['hashtags']:,}</td>
                            <td>{(special_chars_stats['hashtags'] / sum(special_chars_stats.values()) * 100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>Contractions</td>
                            <td>{special_chars_stats['contractions']:,}</td>
                            <td>{(special_chars_stats['contractions'] / sum(special_chars_stats.values()) * 100):.1f}%</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Common Terms Analysis</h2>
                    <h3>Overall Most Common Terms</h3>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(terms_fig)}
                    </div>
                    
                    <h3>Terms by Sentiment</h3>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(terms_by_sentiment_fig)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Common Terms by Sentiment: Before vs After Preprocessing</h2>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(compare_terms_fig)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Preprocessing Impact Analysis</h2>
                    <div class="plot-container">
                        {self.get_plotly_figure_html(impact_fig)}
                    </div>
                    
                    <h3>Detailed Impact Statistics</h3>
                    <table>
                        <tr>
                            <th>Step</th>
                            <th>Average Tokens Removed</th>
                            <th>Text Reduction (%)</th>
                        </tr>
                        {''.join(
                            f'<tr><td>{step}</td><td>{stats["avg_tokens_removed"]:.2f}</td><td>{stats["reduction_percent"]:.1f}%</td></tr>'
                            for step, stats in impacts.items()
                        )}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Preprocessing Steps Applied</h2>
                    <ul>
                        <li>HTML tag removal</li>
                        <li>URL removal</li>
                        <li>Twitter mentions removal</li>
                        <li>Hashtag processing (keeping text)</li>
                        <li>Number handling (preserving prices)</li>
                        <li>Punctuation removal</li>
                        <li>Emoji removal</li>
                        <li>Contraction expansion</li>
                        <li>Unicode normalization</li>
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
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def get_base64_image(self, filename: str) -> str:
        """Convert image to base64 string."""
        import base64
        image_path = self.img_dir / filename
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    # Generate preprocessing analysis report
    analyzer = PreprocessingAnalyzer()
    analyzer.generate_report() 