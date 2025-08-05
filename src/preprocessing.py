"""
Text preprocessing module for sentiment analysis.
Includes tokenization, stemming, stop word removal, and other NLP techniques.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from typing import List, Optional
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing class for sentiment analysis."""
    
    def __init__(self, remove_stopwords: bool = True, 
                 use_stemming: bool = True, 
                 use_lemmatization: bool = False,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 lowercase: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stop words
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            lowercase: Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stop words for sentiment analysis
        self.custom_stop_words = {
            'rt', 'via', 'new', 'time', 'today', 'amp', 'get', 'got', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can'
        }
        self.stop_words.update(self.custom_stop_words)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stop words
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of stemmed tokens
        """
        if not self.use_stemming:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.use_lemmatization:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        
        # Apply stemming or lemmatization
        if self.use_stemming and not self.use_lemmatization:
            tokens = self.stem_tokens(tokens)
        elif self.use_lemmatization and not self.use_stemming:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def preprocess_series(self, text_series: pd.Series) -> pd.Series:
        """
        Preprocess a pandas Series of texts.
        
        Args:
            text_series: Series of texts
            
        Returns:
            Series of preprocessed texts
        """
        return text_series.apply(self.preprocess_text)
    
    def get_preprocessing_summary(self) -> dict:
        """Get a summary of preprocessing settings."""
        return {
            'remove_stopwords': self.remove_stopwords,
            'use_stemming': self.use_stemming,
            'use_lemmatization': self.use_lemmatization,
            'remove_punctuation': self.remove_punctuation,
            'remove_numbers': self.remove_numbers,
            'lowercase': self.lowercase,
            'stop_words_count': len(self.stop_words)
        }

class SentimentSpecificPreprocessor(TextPreprocessor):
    """Specialized preprocessor for sentiment analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Sentiment-specific preprocessing
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
            'nowhere', 'hardly', 'barely', 'scarcely', 'doesnt', 'isnt',
            'wasnt', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 'cant',
            'dont', 'didnt', 'havent', 'hasnt', 'hadnt'
        }
        
        # Emoticons mapping
        self.emoticon_mapping = {
            ':)': ' positive_emoticon ',
            ':-)': ' positive_emoticon ',
            ':(': ' negative_emoticon ',
            ':-(': ' negative_emoticon ',
            ':D': ' positive_emoticon ',
            ':-D': ' positive_emoticon ',
            ';)': ' positive_emoticon ',
            ';-)': ' positive_emoticon ',
            ':P': ' positive_emoticon ',
            ':-P': ' positive_emoticon ',
            ':((': ' negative_emoticon ',
            ':-(((': ' negative_emoticon ',
            ':*': ' positive_emoticon ',
            ':-*': ' positive_emoticon ',
            '8)': ' positive_emoticon ',
            '8-)': ' positive_emoticon ',
            '8(': ' negative_emoticon ',
            '8-(': ' negative_emoticon ',
        }
    
    def handle_negations(self, text: str) -> str:
        """
        Handle negation words in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with negation handling
        """
        words = text.split()
        negated = False
        result = []
        
        for word in words:
            if word.lower() in self.negation_words:
                negated = not negated
                result.append(word)
            elif negated:
                result.append(f"NOT_{word}")
                negated = False
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def replace_emoticons(self, text: str) -> str:
        """
        Replace emoticons with sentiment indicators.
        
        Args:
            text: Input text
            
        Returns:
            Text with emoticons replaced
        """
        for emoticon, replacement in self.emoticon_mapping.items():
            text = text.replace(emoticon, replacement)
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete sentiment-specific text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Replace emoticons first
        text = self.replace_emoticons(text)
        
        # Clean text
        text = self.clean_text(text)
        
        # Handle negations
        text = self.handle_negations(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        
        # Apply stemming or lemmatization
        if self.use_stemming and not self.use_lemmatization:
            tokens = self.stem_tokens(tokens)
        elif self.use_lemmatization and not self.use_stemming:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = SentimentSpecificPreprocessor()
    
    test_texts = [
        "I love this movie! It's absolutely fantastic! :)",
        "This is terrible. I hate it. :(",
        "The movie was not bad, actually quite good.",
        "RT @user: This film is amazing! #movie #love",
        "I don't like this at all. Waste of time."
    ]
    
    print("Testing text preprocessing:")
    for text in test_texts:
        processed = preprocessor.preprocess_text(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print("-" * 50) 