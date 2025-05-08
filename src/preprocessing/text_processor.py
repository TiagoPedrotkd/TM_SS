"""
Text preprocessing module with multiple preprocessing techniques.
"""
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Optional
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing class with multiple cleaning techniques."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Add custom stopwords relevant to financial tweets
        self.custom_stopwords = {
            'rt', 'via', 'new', 'time', 'today', 'join', 'check', 'get', 'see',
            'want', 'make', 'amp', 'u', 'say', 'going', 'would', 'could'
        }
        self.stop_words.update(self.custom_stopwords)
        
        # Common financial terms to keep
        self.keep_terms = {
            'bull', 'bear', 'buy', 'sell', 'hold', 'long', 'short', 'call',
            'put', 'market', 'trade', 'trading', 'price', 'up', 'down'
        }
        self.stop_words = self.stop_words - self.keep_terms
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove Twitter mentions (@user)."""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text: str, keep_text: bool = True) -> str:
        """
        Remove hashtags from text.
        
        Args:
            text: Input text
            keep_text: If True, keep the text part of the hashtag
        """
        if keep_text:
            # Replace #word with word
            return re.sub(r'#(\w+)', r'\1', text)
        # Remove entire hashtag
        return re.sub(r'#\w+', '', text)
    
    def remove_numbers(self, text: str, keep_prices: bool = True) -> str:
        """
        Remove numbers from text.
        
        Args:
            text: Input text
            keep_prices: If True, keep price-like patterns (e.g., $123.45)
        """
        if keep_prices:
            # Temporarily replace price patterns
            text = re.sub(r'\$\d+\.?\d*', 'PRICE_TOKEN', text)
            # Remove other numbers
            text = re.sub(r'\d+', '', text)
            # Restore price patterns
            text = text.replace('PRICE_TOKEN', 'price')
        else:
            text = re.sub(r'\d+', '', text)
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        return ' '.join(text.split())
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = word_tokenize(text.lower())
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text."""
        words = word_tokenize(text.lower())
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def stem_text(self, text: str) -> str:
        """Apply Porter Stemming."""
        words = word_tokenize(text.lower())
        return ' '.join([self.stemmer.stem(word) for word in words])
    
    def preprocess(
        self,
        text: str,
        steps: Optional[List[str]] = None,
        keep_hashtag_text: bool = True,
        keep_prices: bool = True
    ) -> str:
        """
        Apply preprocessing steps in sequence.
        
        Args:
            text: Input text
            steps: List of preprocessing steps to apply. If None, applies all steps.
                  Available steps: ['urls', 'mentions', 'hashtags', 'numbers',
                                  'punctuation', 'stopwords', 'lemmatize', 'stem']
            keep_hashtag_text: If True, keeps the text part of hashtags
            keep_prices: If True, keeps price-like patterns
        """
        if steps is None:
            steps = [
                'urls', 'mentions', 'hashtags', 'numbers',
                'punctuation', 'stopwords', 'lemmatize'
            ]
        
        text = text.lower()
        
        for step in steps:
            if step == 'urls':
                text = self.remove_urls(text)
            elif step == 'mentions':
                text = self.remove_mentions(text)
            elif step == 'hashtags':
                text = self.remove_hashtags(text, keep_hashtag_text)
            elif step == 'numbers':
                text = self.remove_numbers(text, keep_prices)
            elif step == 'punctuation':
                text = self.remove_punctuation(text)
            elif step == 'stopwords':
                text = self.remove_stopwords(text)
            elif step == 'lemmatize':
                text = self.lemmatize_text(text)
            elif step == 'stem':
                text = self.stem_text(text)
        
        return self.remove_extra_whitespace(text)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "RT @trader: Check out $AAPL's price at $150.50! #bullish #stocks",
        "The market is down 5% today... bearish outlook for tech stocks. https://example.com",
        "#Bitcoin going to the moon!!! 🚀 $BTC $ETH",
    ]
    
    print("Testing text preprocessor:")
    print("-" * 50)
    
    for text in test_texts:
        print(f"\nOriginal text:\n{text}")
        processed = preprocessor.preprocess(text)
        print(f"\nProcessed text:\n{processed}")
        print("-" * 50) 