"""
Feature extraction module implementing various text representation techniques.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union, Optional, Dict
import logging
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self, save_dir: Optional[str] = None):
        if save_dir is None:
            save_dir = Path('results') / 'features'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, texts: List[str]) -> None:
        """Fit the feature extractor on training data."""
        raise NotImplementedError
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        raise NotImplementedError
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, name: str) -> None:
        """Save feature extractor configuration."""
        raise NotImplementedError
    
    def load(self, name: str) -> None:
        """Load feature extractor configuration."""
        raise NotImplementedError

class BagOfWordsExtractor(FeatureExtractor):
    """Bag of Words feature extractor with TF-IDF option."""
    
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 5,
        max_df: float = 0.95,
        use_tfidf: bool = True,
        ngram_range: tuple = (1, 2),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_tfidf = use_tfidf
        self.ngram_range = ngram_range
        
        # Initialize vectorizer
        vectorizer_class = TfidfVectorizer if use_tfidf else CountVectorizer
        self.vectorizer = vectorizer_class(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
        
        self.vocabulary_size = None
        self.top_features = None
    
    def fit(self, texts: List[str]) -> None:
        """Fit the vectorizer on training data."""
        logger.info(f"Fitting {'TF-IDF' if self.use_tfidf else 'BoW'} vectorizer...")
        self.vectorizer.fit(texts)
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        
        # Get top features
        if self.use_tfidf:
            feature_scores = np.mean(self.vectorizer.transform(texts).toarray(), axis=0)
        else:
            feature_scores = np.sum(self.vectorizer.transform(texts).toarray(), axis=0)
        
        feature_names = self.vectorizer.get_feature_names_out()
        self.top_features = sorted(
            zip(feature_names, feature_scores),
            key=lambda x: x[1],
            reverse=True
        )[:50]  # Store top 50 features
        
        logger.info(f"Vocabulary size: {self.vocabulary_size}")
        logger.info("Top 10 features:")
        for feature, score in self.top_features[:10]:
            logger.info(f"  {feature}: {score:.4f}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        return self.vectorizer.transform(texts).toarray()
    
    def save(self, name: str) -> None:
        """Save vectorizer configuration and vocabulary."""
        config = {
            'type': 'tfidf' if self.use_tfidf else 'bow',
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'vocabulary_size': self.vocabulary_size,
            'top_features': self.top_features
        }
        
        config_file = self.save_dir / f'{name}_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_file}")

class Word2VecExtractor(FeatureExtractor):
    """Word2Vec feature extractor."""
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 4,
        sg: int = 1,  # Skip-gram (1) vs CBOW (0)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
        self.vocabulary_size = None
        self.similar_words = {}
    
    def fit(self, texts: List[str]) -> None:
        """Train Word2Vec model on texts."""
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        logger.info("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        
        self.vocabulary_size = len(self.model.wv.key_to_index)
        logger.info(f"Vocabulary size: {self.vocabulary_size}")
        
        # Find similar words for some financial terms
        financial_terms = ['bullish', 'bearish', 'buy', 'sell', 'market']
        for term in financial_terms:
            if term in self.model.wv:
                similar = self.model.wv.most_similar(term, topn=5)
                self.similar_words[term] = similar
                logger.info(f"\nMost similar to '{term}':")
                for word, score in similar:
                    logger.info(f"  {word}: {score:.4f}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors using average word embeddings."""
        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = []
            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
            if word_vectors:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)
    
    def save(self, name: str) -> None:
        """Save Word2Vec model and configuration."""
        if self.model is not None:
            model_file = self.save_dir / f'{name}_word2vec.model'
            self.model.save(str(model_file))
            
            config = {
                'type': 'word2vec',
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'sg': self.sg,
                'vocabulary_size': self.vocabulary_size,
                'similar_words': self.similar_words
            }
            
            config_file = self.save_dir / f'{name}_config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved model to {model_file}")
            logger.info(f"Saved configuration to {config_file}")

class TransformerExtractor(FeatureExtractor):
    """Transformer-based feature extractor."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        batch_size: int = 32,
        device: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors using transformer embeddings."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting features"):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def fit(self, texts: List[str]) -> None:
        """No fitting required for pre-trained transformers."""
        pass
    
    def save(self, name: str) -> None:
        """Save transformer configuration."""
        config = {
            'type': 'transformer',
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }
        
        config_file = self.save_dir / f'{name}_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_file}")

def create_feature_extractor(
    method: str = 'transformer',
    **kwargs
) -> FeatureExtractor:
    """
    Factory function to create feature extractors.
    
    Args:
        method: One of 'bow', 'word2vec', 'transformer'
        **kwargs: Additional arguments for the specific extractor
    """
    extractors = {
        'bow': BagOfWordsExtractor,
        'word2vec': Word2VecExtractor,
        'transformer': TransformerExtractor
    }
    
    if method not in extractors:
        raise ValueError(f"Unknown method: {method}. Choose from {list(extractors.keys())}")
    
    return extractors[method](**kwargs) 