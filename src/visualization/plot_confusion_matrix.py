import plotly.graph_objects as go
import numpy as np
import json
import os

# Diretório raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_evaluation_results():
    """Load evaluation results from JSON file"""
    eval_path = os.path.join(PROJECT_ROOT, 'results', 'model_evaluation', 'evaluation_results.json')
    with open(eval_path, 'r') as f:
        return json.load(f)

def create_confusion_matrix_plot(cm, labels):
    """
    Create an improved confusion matrix visualization
    
    Args:
        cm: Confusion matrix as a list of lists
        labels: List of class labels
    """
    # Convert to numpy array
    cm = np.array(cm)
    
    # Calculate percentages (normalize by row to show classification accuracy for each class)
    row_sums = cm.sum(axis=1)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(len(cm)):
        if row_sums[i] > 0:
            cm_percent[i] = (cm[i] / row_sums[i]) * 100
    
    # Create text annotations showing both count and percentage
    text = []
    for i in range(len(cm)):
        text.append([])
        for j in range(len(cm)):
            text[-1].append(
                f'{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)'
            )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_percent,  # Use percentages for colors
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        colorscale='RdBu',  # Red-Blue diverging colorscale
        reversescale=True,  # Reverse so darker blue means higher values
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis={'side': 'bottom'},
        width=800,
        height=600,
        font=dict(size=14)
    )
    
    # Add color bar title
    fig.update_traces(
        colorbar_title="Percentage"
    )
    
    return fig

def main():
    # Load evaluation results
    results = load_evaluation_results()
    
    # Get confusion matrix from the ensemble results
    # Using bow_transformer as an example (you can change this to any model)
    model_results = results['per_fold_results']['bow_transformer']
    confusion_matrix = model_results['confusion_matrix']
    
    # Define labels
    labels = ["Bearish", "Bullish", "Neutral"]

    # Diretório de saída absoluto
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'ensemble_evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # Create confusion matrix plot
    fig = create_confusion_matrix_plot(confusion_matrix, labels)

    # Caminhos absolutos para salvar
    html_path = os.path.join(output_dir, 'confusion_matrix.html')
    metrics_path = os.path.join(output_dir, 'confusion_matrix_metrics.json')

    fig.write_html(html_path)

    # Also save metrics
    metrics = {
        'confusion_matrix': confusion_matrix,
        'labels': labels,
        'accuracy': model_results['report']['accuracy'],
        'macro_avg': model_results['report']['macro avg']
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()