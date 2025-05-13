"""
Feature extraction module implementing different approaches:
1. Bag of Words (BoW) / TF-IDF
2. Word2Vec
3. Transformer (BERT)
"""

import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

class BagOfWordsExtractor:
    """Extract features using Bag of Words or TF-IDF."""
    
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 5,
        max_df: float = 0.95,
        use_tfidf: bool = True,
        ngram_range: tuple = (1, 2)
    ):
        """
        Initialize BoW/TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_tfidf: Whether to use TF-IDF instead of raw counts
            ngram_range: Range of n-grams to use
        """
        self.vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
        self.vectorizer = self.vectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
    
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

class Word2VecExtractor:
    """Extract features using Word2Vec."""
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 0  # 0: CBOW, 1: Skip-gram
    ):
        """
        Initialize Word2Vec extractor.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word frequency
            sg: Training algorithm (0: CBOW, 1: Skip-gram)
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Train Word2Vec model and transform texts.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Document vectors (averaged word vectors)
        """
        # Tokenize texts
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=4
        )
        
        return self._get_doc_vectors(tokenized_texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using trained model.
        
        Args:
            texts: List of preprocessed texts
        
        Returns:
            Document vectors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit_transform first.")
        
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        return self._get_doc_vectors(tokenized_texts)
    
    def _get_doc_vectors(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Get document vectors by averaging word vectors."""
        doc_vectors = []
        for tokens in tokenized_texts:
            # Get vectors for tokens that are in vocabulary
            token_vectors = [
                self.model.wv[token]
                for token in tokens
                if token in self.model.wv
            ]
            
            if token_vectors:
                # Average vectors if we have any
                doc_vector = np.mean(token_vectors, axis=0)
            else:
                # Use zero vector if no tokens found
                doc_vector = np.zeros(self.vector_size)
            
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)

class TransformerExtractor:
    """Extract features using a transformer model."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        batch_size: int = 32
    ):
        """
        Initialize transformer extractor.
        
        Args:
            model_name: Name of the pretrained model to use
            max_length: Maximum sequence length
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def _process_batch(self, batch_texts: List[str]) -> np.ndarray:
        """Process a batch of texts."""
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
            
            # Use [CLS] token embedding as document representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
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

def create_feature_extractor(method: str, **kwargs) -> Union[BagOfWordsExtractor, Word2VecExtractor, TransformerExtractor]:
    """
    Factory function to create a feature extractor based on the specified method.
    
    Args:
        method: Feature extraction method ('bow', 'word2vec', or 'transformer')
        **kwargs: Additional arguments to pass to the extractor
    
    Returns:
        Feature extractor instance
    
    Raises:
        ValueError: If method is not supported
    """
    if method == 'bow':
        return BagOfWordsExtractor(**kwargs)
    elif method == 'word2vec':
        return Word2VecExtractor(**kwargs)
    elif method == 'transformer':
        return TransformerExtractor(**kwargs)
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}") 