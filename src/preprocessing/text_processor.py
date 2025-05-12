"""
Text preprocessing module with multiple preprocessing techniques.

This module implements several text preprocessing techniques shown in class:
1. Stop Words Removal - Using NLTK's stopwords with custom financial additions
2. Regular Expressions - For pattern matching (URLs, numbers, punctuation, etc.)
3. Lemmatization - Using NLTK's WordNetLemmatizer (preferred over stemming for financial text)
4. Tokenization - Using NLTK's word_tokenize
5. Case Normalization - Converting text to lowercase
6. Punctuation Removal - With special handling for financial symbols
7. Contraction Expansion - Using the contractions library
8. Stemming - Available but not used by default (see note below)

Note on Lemmatization vs Stemming:
    We use lemmatization by default instead of stemming because:
    - Lemmatization produces actual words, maintaining readability
    - Financial terms need precise meaning (e.g., 'trading' vs 'trade')
    - Stemming can be too aggressive for financial terminology
    However, stemming is still available through the stem_text() method if needed.
"""
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Optional, Dict, Set
import logging
import emoji
import contractions
from bs4 import BeautifulSoup
import unicodedata

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
        
        # Keep essential financial terms and their semantic relationships
        self.keep_terms = {
            # Market sentiment
            'bull', 'bear', 'bullish', 'bearish',
            # Actions
            'buy', 'sell', 'hold', 'long', 'short',
            # Directions
            'up', 'down', 'higher', 'lower',
            # Financial terms
            'q', 'quarter',  # Keep quarter references
            'a', 'b', 'c',   # Share classes
            'market', 'stock', 'price', 'share',
            'bid', 'ask', 'trading', 'volume'
        }
        
        # Add custom stopwords relevant to financial tweets
        self.custom_stopwords = {
            'rt', 'via', 'new', 'time', 'today', 'join', 'check', 'get', 'see',
            'want', 'make', 'amp', 'u', 'say', 'going', 'would', 'could',
            # Add basic stopwords
            "'", "'s", "''", '""'
        }
        
        # Update stopwords while preserving important terms
        self.stop_words.update(self.custom_stopwords)
        self.stop_words = self.stop_words - self.keep_terms
        
        # Initialize emoji patterns
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return BeautifulSoup(text, "html.parser").get_text()
    
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
    
    def remove_punctuation(self, text: str, keep_hyphens: bool = False) -> str:
        """Remove punctuation from text, optionally keeping hyphens."""
        if keep_hyphens:
            punct = string.punctuation.replace('-', '')
        else:
            punct = string.punctuation
        
        # First handle possessives
        text = re.sub(r"'s\b", "", text)  # Remove 's
        text = re.sub(r"s'\b", "s", text)  # Change s' to s
        
        # Tokenize and remove tokens that are only punctuation
        tokens = word_tokenize(text)
        filtered = [t for t in tokens if not (
            all(c in punct for c in t) and 
            t not in ('-',) if keep_hyphens else 
            all(c in punct for c in t)
        )]
        
        # Additional filtering for single characters and remaining apostrophes
        filtered = [t for t in filtered if len(t) > 1 or t in self.keep_terms]
        filtered = [t.strip("'") for t in filtered]  # Remove surrounding apostrophes
        
        return ' '.join(filtered)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        return ' '.join(text.split())
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return self.emoji_pattern.sub('', text)
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text and clean up artifacts."""
        # Expand contractions
        text = contractions.fix(text)
        
        # Clean up any remaining contraction artifacts
        text = re.sub(r"'\s*s\b", "", text)  # Remove 's with possible space
        text = re.sub(r"'\s*t\b", "", text)  # Remove 't with possible space
        text = re.sub(r"'\s*m\b", "", text)  # Remove 'm with possible space
        text = re.sub(r"'\s*re\b", "", text)  # Remove 're with possible space
        text = re.sub(r"'\s*ve\b", "", text)  # Remove 've with possible space
        text = re.sub(r"'\s*ll\b", "", text)  # Remove 'll with possible space
        text = re.sub(r"'\s*d\b", "", text)  # Remove 'd with possible space
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
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
        keep_prices: bool = True,
        keep_hyphens: bool = True
    ) -> str:
        """
        Apply preprocessing steps in sequence.
        
        Args:
            text: Input text
            steps: List of preprocessing steps to apply. If None, applies core steps.
                  Available steps: [
                      'html', 'urls', 'mentions', 'hashtags', 'numbers',
                      'punctuation', 'emojis', 'stopwords', 'lemmatize'
                  ]
            keep_hashtag_text: If True, keeps the text part of hashtags
            keep_prices: If True, keeps price-like patterns
            keep_hyphens: If True, keeps hyphens in text
        """
        if steps is None:
            # Simplified core steps based on lab examples
            steps = [
                'urls',          # Remove URLs first
                'mentions',      # Remove @mentions
                'hashtags',      # Process hashtags
                'contractions',  # Expand contractions before other processing
                'unicode',       # Normalize unicode
                'numbers',       # Handle numbers/prices
                'punctuation',   # Remove punctuation
                'stopwords',     # Remove stopwords
                'lemmatize'      # Lemmatization last
            ]
        
        # Always start with lowercase
        text = text.lower()
        
        for step in steps:
            if step == 'urls':
                text = self.remove_urls(text)
            elif step == 'mentions':
                text = self.remove_mentions(text)
            elif step == 'hashtags':
                text = self.remove_hashtags(text, keep_hashtag_text)
            elif step == 'contractions':
                text = self.expand_contractions(text)
            elif step == 'unicode':
                text = self.normalize_unicode(text)
            elif step == 'numbers':
                text = self.remove_numbers(text, keep_prices)
            elif step == 'punctuation':
                text = self.remove_punctuation(text, keep_hyphens=keep_hyphens)
            elif step == 'stopwords':
                text = self.remove_stopwords(text)
            elif step == 'lemmatize':
                text = self.lemmatize_text(text)
        
        return self.remove_extra_whitespace(text)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "RT @trader: Check out $AAPL's price at $150.50! #bullish #stocks",
        "The market is down 5% today... bearish outlook for tech stocks. https://example.com",
        "#Bitcoin going to the moon!!! 🚀 $BTC $ETH",
        "I'm gonna buy some stocks tomorrow! Don't wanna miss out!",
        "HTML <b>tags</b> and &amp; entities",
        "Unicode characters: é, ñ, ü"
    ]
    
    print("Testing text preprocessor:")
    print("-" * 50)
    
    for text in test_texts:
        print(f"\nOriginal text:\n{text}")
        processed = preprocessor.preprocess(text)
        print(f"\nProcessed text:\n{processed}")
        print("-" * 50) 