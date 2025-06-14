import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os

def save_plot(fig, filename, output_dir='../results/'):
    """Save plotly figure as HTML."""
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, f"{filename}.html"))

def save_fig(fig, filename, output_dir='../results/'):
    """Save matplotlib figure as PNG."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight')

def plot_class_distribution(df, label_col='label'):
    """Plot distribution of classes."""
    label_map = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
    value_counts = df[label_col].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=[label_map[i] for i in value_counts.index],
            y=value_counts.values,
            text=value_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Distribution of Market Sentiment Classes',
        xaxis_title='Sentiment',
        yaxis_title='Count',
        template='plotly_white'
    )
    
    return fig

def plot_word_frequencies(texts, top_n=20):
    """Plot top N most frequent words."""
    from collections import Counter
    import itertools
    
    words = itertools.chain.from_iterable(text.split() for text in texts)
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
        title=f'Top {top_n} Most Frequent Words',
        xaxis_title='Word',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    
    return fig

def plot_evaluation_metrics(metrics_dict):
    """Plot evaluation metrics for each model."""
    fig = go.Figure(data=[
        go.Bar(
            name=metric,
            x=list(metrics_dict.keys()),
            y=[model_metrics[metric] for model_metrics in metrics_dict.values()],
            text=[f"{model_metrics[metric]:.3f}" for model_metrics in metrics_dict.values()],
            textposition='auto',
        )
        for metric in ['precision', 'recall', 'f1_score', 'accuracy']
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        template='plotly_white',
        barmode='group'
    )
    
    return fig

def plot_wordcloud(texts, title="Word Cloud", width=800, height=400, background_color='white', colormap='viridis'):
    """
    Gera e retorna uma word cloud para uma lista de textos.
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    text_all = " ".join([str(t) for t in texts])

    wordcloud = WordCloud(width=width, height=height, background_color=background_color, colormap=colormap).generate(text_all)

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()

    return fig