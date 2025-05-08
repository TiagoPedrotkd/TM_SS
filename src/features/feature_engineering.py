from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union
import numpy.typing as npt

class FeatureExtractor:
    def __init__(self, method='bow', **kwargs):
        """
        Initialize feature extractor.
        
        Args:
            method: One of 'bow', 'tfidf', 'word2vec', 'transformer'
            **kwargs: Additional arguments for the specific method
        """
        self.method = method
        self.model = None
        self.tokenizer = None
        
        if method == 'bow':
            self.model = CountVectorizer(**kwargs)
        elif method == 'tfidf':
            self.model = TfidfVectorizer(**kwargs)
        elif method == 'word2vec':
            self.vector_size = kwargs.get('vector_size', 100)
            self.window = kwargs.get('window', 5)
            self.min_count = kwargs.get('min_count', 1)
        elif method == 'transformer':
            model_name = kwargs.get('model_name', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
    
    def fit(self, texts: List[str]) -> None:
        """Fit the feature extractor on the training data."""
        if self.method in ['bow', 'tfidf']:
            self.model.fit(texts)
        elif self.method == 'word2vec':
            # Tokenize texts for word2vec
            tokenized_texts = [text.split() for text in texts]
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count
            )
    
    def transform(self, texts: List[str]) -> Union[npt.NDArray, torch.Tensor]:
        """Transform texts to feature vectors."""
        if self.method in ['bow', 'tfidf']:
            return self.model.transform(texts).toarray()
        
        elif self.method == 'word2vec':
            # Average word vectors for each text
            vectors = []
            for text in texts:
                words = text.split()
                word_vectors = []
                for word in words:
                    try:
                        word_vectors.append(self.model.wv[word])
                    except KeyError:
                        continue
                if word_vectors:
                    vectors.append(np.mean(word_vectors, axis=0))
                else:
                    vectors.append(np.zeros(self.vector_size))
            return np.array(vectors)
        
        elif self.method == 'transformer':
            # Get BERT embeddings
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            return embeddings
    
    def fit_transform(self, texts: List[str]) -> Union[npt.NDArray, torch.Tensor]:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts) 