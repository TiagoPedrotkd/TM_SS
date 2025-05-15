"""
Feature extraction module implementing different approaches:
1. Bag of Words (BoW) / TF-IDF
2. Word2Vec
3. Transformer (BERT)
4. Combined Features
"""

import numpy as np
from typing import List, Union, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BagOfWordsExtractor:
    """Extract features using Bag of Words or TF-IDF."""
    
    def __init__(
        self,
        max_features: int = 10000,  # Increased for better coverage
        min_df: int = 3,  # Reduced to capture more rare but potentially important terms
        max_df: float = 0.90,  # Slightly reduced to remove more common terms
        use_tfidf: bool = True,
        ngram_range: tuple = (1, 3),  # Extended to trigrams
        norm: str = 'l2',  # Add normalization (TF-IDF only)
        sublinear_tf: bool = True,  # Better scaling for term frequencies (TF-IDF only)
        stop_words: str = 'english'  # Add stop words removal
    ):
        """
        Initialize BoW/TF-IDF extractor with improved parameters.
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_tfidf: Whether to use TF-IDF instead of raw counts
            ngram_range: Range of n-grams to use
            norm: Normalization method ('l1', 'l2', or None) - TF-IDF only
            sublinear_tf: Apply sublinear scaling to term frequencies - TF-IDF only
            stop_words: Stop words to remove ('english' or None)
        """
        # Common parameters for both vectorizers
        common_params = {
            'max_features': max_features,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': ngram_range,
            'stop_words': stop_words
        }
        
        if use_tfidf:
            # TF-IDF specific parameters
            tfidf_params = {
                'norm': norm,
                'sublinear_tf': sublinear_tf
            }
            self.vectorizer = TfidfVectorizer(**common_params, **tfidf_params)
        else:
            # CountVectorizer doesn't use TF-IDF specific parameters
            self.vectorizer = CountVectorizer(**common_params)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Feature matrix
        """
        return self.vectorizer.fit_transform(texts).toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Feature matrix
        """
        return self.vectorizer.transform(texts).toarray()
    
    def save(self, method: str) -> None:
        """
        Save the vectorizer configuration.
        
        Args:
            method: Name of the feature extraction method
        """
        config = {
            'max_features': self.vectorizer.max_features,
            'min_df': self.vectorizer.min_df,
            'max_df': self.vectorizer.max_df,
            'ngram_range': self.vectorizer.ngram_range,
            'vocabulary_size': len(self.vectorizer.vocabulary_)
        }
        return config

class Word2VecExtractor:
    """Extract features using Word2Vec with FastText backup."""
    
    def __init__(
        self,
        vector_size: int = 300,  # Increased dimension for better representation
        window: int = 8,  # Increased context window
        min_count: int = 2,
        sg: int = 1,  # Changed to skip-gram for better quality
        negative: int = 10,  # Increased negative samples
        epochs: int = 20,  # More training epochs
        use_fasttext: bool = True  # Add FastText option
    ):
        """
        Initialize Word2Vec extractor with improved parameters.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word frequency
            sg: Training algorithm (0: CBOW, 1: Skip-gram)
            negative: Number of negative samples
            epochs: Number of training epochs
            use_fasttext: Whether to use FastText as backup for OOV words
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.negative = negative
        self.epochs = epochs
        self.use_fasttext = use_fasttext
        self.model = None
        self.fasttext_model = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Train Word2Vec model and transform texts."""
        # Tokenize texts
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Train Word2Vec model
        logger.info("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            negative=self.negative,
            epochs=self.epochs,
            workers=4
        )
        
        # Train FastText model if enabled
        if self.use_fasttext:
            logger.info("Training FastText model for OOV words...")
            self.fasttext_model = FastText(
                sentences=tokenized_texts,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                negative=self.negative,
                epochs=self.epochs,
                workers=4
            )
        
        return self._get_doc_vectors(tokenized_texts)
    
    def _get_doc_vectors(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Get document vectors with improved aggregation."""
        doc_vectors = []
        for tokens in tokenized_texts:
            # Get vectors for tokens that are in vocabulary
            token_vectors = []
            token_weights = []  # Add IDF-like weighting
            
            for token in tokens:
                vector = None
                weight = 1.0  # Default weight
                
                if token in self.model.wv:
                    vector = self.model.wv[token]
                    # Get word frequency from vocabulary using vocab object
                    try:
                        # Try the new gensim API first
                        word_count = self.model.wv.get_vecattr(token, "count")
                    except (AttributeError, KeyError):
                        # Fallback for older gensim versions or if count not available
                        word_count = self.model.wv.vocab[token].count if hasattr(self.model.wv, 'vocab') else 1
                    # Calculate IDF-like weight
                    weight = 1.0 / (1.0 + np.log1p(word_count))  # Using log1p for smoother weights
                elif self.use_fasttext and token in self.fasttext_model.wv:
                    vector = self.fasttext_model.wv[token]
                    # For FastText words, use a default weight
                    weight = 0.8  # Slightly lower weight for FastText words
                
                if vector is not None:
                    token_vectors.append(vector)
                    token_weights.append(weight)
            
            if token_vectors:
                # Weighted average of vectors
                token_vectors = np.array(token_vectors)
                token_weights = np.array(token_weights)
                token_weights = token_weights / token_weights.sum()  # Normalize weights
                doc_vector = np.average(token_vectors, weights=token_weights, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)
            
            doc_vectors.append(doc_vector)
        
        # Normalize document vectors
        return normalize(np.array(doc_vectors))
    
    def save(self, method: str) -> None:
        """
        Save the Word2Vec configuration.
        
        Args:
            method: Name of the feature extraction method
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit_transform first.")
            
        config = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'sg': self.sg,
            'vocabulary_size': len(self.model.wv.key_to_index)
        }
        return config

class TransformerExtractor:
    """Extract features using transformer models with improved FinBERT support."""
    
    def __init__(
        self,
        model_name: str = 'ProsusAI/finbert',
        max_length: int = 256,
        batch_size: int = 32,
        pooling_strategy: str = 'mean_pooling'
    ):
        """
        Initialize transformer extractor.
        
        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            pooling_strategy: Pooling strategy ('mean_pooling', 'cls', or 'max_pooling')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling_strategy = pooling_strategy
        
        # Load tokenizer and model
        logger.info(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _max_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform max pooling on token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def _process_batch(self, batch_texts: List[str]) -> np.ndarray:
        """Process a batch of texts with improved pooling."""
        # Tokenize
        inputs = self.tokenizer(
            batch_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if self.pooling_strategy == 'cls':
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooling_strategy == 'mean_pooling':
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            else:  # max_pooling
                embeddings = self._max_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using the transformer model.
        No fitting needed as we use a pretrained model.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Document embeddings
        """
        return self.transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using the transformer model.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Document embeddings
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def save(self, method: str) -> None:
        """
        Save the transformer configuration.
        
        Args:
            method: Name of the feature extraction method
        """
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'hidden_size': self.model.config.hidden_size
        }
        return config

class CombinedFeatureExtractor:
    """Extract and combine multiple types of features."""
    
    def __init__(
        self,
        extractors: List[Union[BagOfWordsExtractor, Word2VecExtractor, TransformerExtractor]],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize combined feature extractor.
        
        Args:
            extractors: List of feature extractors to combine
            weights: Optional weights for each extractor (default: equal weights)
        """
        self.extractors = extractors
        if weights is None:
            self.weights = [1.0 / len(extractors)] * len(extractors)
        else:
            if len(weights) != len(extractors):
                raise ValueError("Number of weights must match number of extractors")
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit all extractors and combine their features.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Combined feature matrix
        """
        # Get features from each extractor
        all_features = []
        for extractor in self.extractors:
            features = extractor.fit_transform(texts)
            # Normalize features
            features = normalize(features)
            all_features.append(features)
        
        # Combine features with weights
        combined = np.hstack([
            feat * weight for feat, weight in zip(all_features, self.weights)
        ])
        
        return combined
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using all fitted extractors.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Combined feature matrix
        """
        # Get features from each extractor
        all_features = []
        for extractor in self.extractors:
            features = extractor.transform(texts)
            # Normalize features
            features = normalize(features)
            all_features.append(features)
        
        # Combine features with weights
        combined = np.hstack([
            feat * weight for feat, weight in zip(all_features, self.weights)
        ])
        
        return combined

def create_feature_extractor(method: str, **kwargs) -> Union[BagOfWordsExtractor, Word2VecExtractor, TransformerExtractor, CombinedFeatureExtractor]:
    """Create feature extractor based on method name."""
    if method == 'bow':
        return BagOfWordsExtractor(**kwargs)
    elif method == 'word2vec':
        return Word2VecExtractor(**kwargs)
    elif method == 'transformer':
        return TransformerExtractor(**kwargs)
    elif method == 'combined':
        # Extract extractors and weights from kwargs
        extractors = kwargs.pop('extractors', None)
        weights = kwargs.pop('weights', None)
        if extractors is None:
            raise ValueError("Must provide 'extractors' for combined method")
        return CombinedFeatureExtractor(extractors=extractors, weights=weights)
    else:
        raise ValueError(f"Unknown feature extraction method: {method}") 