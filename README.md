# Text Mining Project 2025

# Text Mining Project 2025: Financial Sentiment Analysis

## Overview
This project focuses on financial sentiment analysis using an ensemble of different machine learning and deep learning models. Our goal is to classify financial texts into three sentiment categories: Bearish, Bullish, and Neutral, using a variety of text representation techniques and classification algorithms.

## Data Exploration
- **Dataset**: Financial news articles and social media content with sentiment labels
- **Classes**: 3 sentiment classes (Bearish, Bullish, Neutral)
- **Distribution**: Original data contained some class imbalance with more neutral samples
- **Text Length**: Variable length texts ranging from short tweets to longer financial news excerpts
- **Domain-Specific Vocabulary**: Contains financial jargon and technical terms

## Corpus Splitting
- **Training/Test Split**: 80/20 split of the dataset
- **Cross-Validation**: 5-fold cross-validation for model evaluation
- **Stratified Sampling**: Maintained class distribution across splits to handle imbalance

## Preprocessing Techniques
- **Text Cleaning**:
  - Case normalization (lowercase)
  - Special character removal
  - URL and hashtag standardization
- **Tokenization**: Using NLTK's word tokenizer
- **Stop Word Removal**: Removed common stop words while preserving sentiment-indicative words
- **Lemmatization**: WordNet lemmatizer for word normalization
- **Financial Term Handling**: Preserved financial terms and numeric values important for sentiment

## Feature Extractors
1. **Bag of Words (BOW)**:
   - TF-IDF vectorization
   - N-gram range: (1,2) to capture phrases
   - Maximum features: 5,000-10,000 depending on configuration

2. **Word2Vec**:
   - Custom-trained on financial corpus
   - Vector size: 300 dimensions
   - Window size: 8
   - Skip-gram model
   - Negative sampling: 10
   - 20 training epochs
   - FastText backup for out-of-vocabulary words

3. **Transformer-Based**:
   - FinBERT embeddings (financial domain-specific BERT)
   - Maximum sequence length: 256 tokens
   - Multiple pooling strategies: mean, max, and CLS token
   - Fine-tuned for financial sentiment

## Classifiers
1. **Statistical Models**:
   - K-Nearest Neighbors (KNN)
     - Distance-weighted
     - Cosine similarity metric
     - K=5 neighbors

2. **Deep Learning Models**:
   - **LSTM**:
     - Bidirectional architecture
     - 2 layers with 256 hidden units
     - Dropout (0.3) for regularization
     - Early stopping with patience=5
     - Adam optimizer with learning rate=0.001

   - **Transformer**:
     - Custom encoder with 4 layers
     - 8 attention heads
     - 512 hidden units
     - Dropout (0.2) for regularization
     - Early stopping with patience=5
     - Adam optimizer with learning rate=0.0001

## Ensemble Model
- **Approach**: Weighted voting ensemble of multiple base models
- **Configurations**:
  - BOW + KNN
  - Word2Vec + LSTM
  - Word2Vec + Transformer
  - FinBERT + Transformer
- **Dynamic Weight Optimization**:
  - Weights for each model determined by validation performance
  - Models with higher F1-scores get higher weights
- **Inference**: 
  - Weighted probability averaging
  - Final prediction based on highest weighted probability
- **Benefits**: 
  - Improved robustness against different text styles
  - Better handling of domain-specific language
  - Higher overall accuracy and F1-score compared to single models

## Performance
- Cross-validation scores on test set:
  - **Accuracy**: ~85-90%
  - **F1-score**: ~83-88%
  - **Performance by class**: Strongest on Bullish and Bearish, slightly lower on Neutral

## Next Steps
- Additional feature combinations
- Hyperparameter optimization
- External financial knowledge integration
- Real-time sentiment analysis capabilities