# Text Mining Project 2025: Financial Sentiment Analysis

## Overview
This project focuses on financial sentiment analysis using an ensemble of different machine learning and deep learning models. Our goal is to classify financial texts into three sentiment categories: Bearish, Bullish, and Neutral, using a variety of text representation techniques and classification algorithms.

## Data Exploration
- **Dataset**: Financial news articles and social media content with sentiment labels
- **Classes**: 3 sentiment classes (Bearish (0), Bullish (1), Neutral (2))
- **Distribution**: The dataset contains approximately 12,000 samples (9,543 for training, 2,388 for testing)
- **Text Characteristics**: 
  - Variable length texts ranging from short tweets to longer financial news excerpts
  - Contains financial jargon, stock tickers ($AAPL, $GOOG), and domain-specific terminology
  - Many references to market performance, earnings reports, and analyst ratings

## Corpus Splitting
- **Training/Test Split**: 80/20 split, resulting in 9,543 training samples and 2,388 test samples
- **Cross-Validation**: 5-fold stratified cross-validation implemented for model evaluation
- **Stratified Sampling**: Maintained the class distribution across splits to handle imbalance
- **Data Handling**: Implemented in `utils/data_loader.py` which loads CSV files from the data directory

## Preprocessing Techniques
Our `TextPreprocessor` class implements comprehensive text cleaning:

- **Text Cleaning**:
  - Case normalization (lowercase)
  - Special character removal with domain-specific considerations
  - URL and hashtag standardization
  - Emoji removal
  - HTML tag removal
- **Tokenization**: Using NLTK's word_tokenize with custom handling for financial terms
- **Stop Word Removal**: 
  - Used NLTK stopwords with custom additions
  - Preserved important financial terms like 'bull', 'bear', 'bullish', 'bearish'
- **Lemmatization**: WordNet lemmatizer for word normalization
- **Contraction Expansion**: Expanded contractions (e.g., "it's" → "it is")
- **Financial Specific Handling**: 
  - Preserved price patterns (e.g., "$123.45")
  - Kept stock tickers and financial symbols
  - Maintained numeric values important for sentiment
- **Unicode Normalization**: Standardized special characters

## Feature Extractors
We implemented three different feature extraction approaches, each optimized for financial text:

### 1. Bag of Words (BOW) Extractor
- **TF-IDF Vectorization** with enhanced parameters:
  - Maximum features: 10,000 for better vocabulary coverage
  - N-gram range: (1,3) to capture phrases and expressions
  - Minimum document frequency: 3 to focus on meaningful terms
  - L2 normalization and sublinear term frequency scaling
  - English stopwords removal 

### 2. Word2Vec Extractor
- **Custom-trained Word2Vec** on our financial corpus:
  - Vector size: 300 dimensions for richer representation
  - Window size: 8 to capture broader context
  - Skip-gram model (sg=1) for better quality
  - Negative sampling: 10 examples
  - 20 training epochs
  - IDF-like weighting for aggregating word vectors
- **FastText Backup**:
  - Handles out-of-vocabulary words
  - Same parameters as Word2Vec
  - Enables robust handling of unseen financial terms

### 3. Transformer-Based Extractor
- **FinBERT Model**:
  - Specialized financial BERT model (ProsusAI/finbert)
  - Pre-trained on financial corpus
  - Maximum sequence length: 256 tokens
  - Multiple pooling strategies: 
    - Mean pooling
    - CLS token representation
    - Max pooling
  - Batch processing for efficiency

## Classifiers
We implemented both traditional ML and deep learning models:

### 1. K-Nearest Neighbors (KNN)
- Distance-weighted
- Cosine similarity metric (better for text)
- K=5 neighbors
- Fast and interpretable baseline

### 2. LSTM Classifier
- Bidirectional architecture
- 2 layers with 256 hidden units
- Dropout (0.3) for regularization
- Multiple input dimensions based on feature extractor
- Advanced concatenation of forward and backward outputs

### 3. Transformer Classifier
- Custom encoder with 4 layers
- 8 attention heads
- 512 hidden dimensions
- Dropout (0.2) for regularization
- Input projection to match transformer dimensions
- Mean pooling over sequence dimension

### Training Features
- **Early Stopping**: Patience of 5 epochs to prevent overfitting
- **Class Balancing**: Automatic class weight computation based on frequencies
- **Learning Rate**: 0.001 for LSTM, 0.0001 for Transformer
- **Batch Processing**: 64 for LSTM, 32 for Transformer
- **Optimizer**: Adam with custom parameters

## Ensemble Model
Our `EnsembleTrainer` implements a sophisticated model combination approach:

### Architecture
- **Weighted Voting Ensemble**: Combines multiple models with learned weights
- **Model Configurations**:
  - bow_lstm: BOW features with LSTM classifier
  - bow_transformer: BOW features with Transformer classifier
  - word2vec_lstm: Word2Vec features with LSTM classifier
  - transformer_knn: FinBERT features with KNN classifier
  - word2vec_transformer: Word2Vec features with Transformer classifier

### Weight Optimization
- **Dynamic Weight Determination**:
  - Initialized with equal weights
  - Optimized based on validation performance
  - Used Dirichlet distribution for weight exploration
  - Selected weights that maximize macro F1-score
  - Automatically adapts to model strengths

### Inference Process
- **Soft Voting**:
  - Weighted probability averaging
  - Final prediction based on highest weighted probability
  - Handles models without probability output by converting to one-hot

### Evaluation and Reporting
- **Cross-Validation**: 5-fold evaluation of ensemble
- **Performance Metrics**:
  - Accuracy
  - Precision, Recall, F1-score per class
  - Macro average F1-score
  - Confusion matrix
- **Visualization**: 
  - Interactive plotly visualizations
  - Performance comparison charts
  - Confusion matrix heatmaps

## Results and Benefits
- **Performance Improvement**: The ensemble approach outperforms individual models
- **Robustness**: Better handling of diverse text styles and financial terminology
- **Adaptability**: Custom weighting allows adaptation to different financial contexts
- **Explainability**: Visualization tools for understanding model performance
- **Production-Ready**: Implemented with efficiency and deployment considerations

## Next Steps
- Additional feature combinations
- Hyperparameter optimization
- External financial knowledge integration
- Real-time sentiment analysis capabilities
- Integration with trading signals and market indicators
