"""
Feature extraction module for sentiment analysis.
Implements bag-of-words, TF-IDF, and Word2Vec embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import os
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Base class for feature extraction methods."""
    
    def __init__(self, **kwargs):
        self.vectorizer = None
        self.is_fitted = False
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractor':
        """Fit the feature extractor on training data."""
        raise NotImplementedError
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts to features."""
        raise NotImplementedError
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)
    
    def save_model(self, filepath: str):
        """Save the fitted model."""
        if self.is_fitted:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
        else:
            raise ValueError("Model must be fitted before saving")
    
    def load_model(self, filepath: str) -> 'FeatureExtractor':
        """Load a fitted model."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)
            logger.info(f"Model loaded from {filepath}")
            return self
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

class BagOfWordsExtractor(FeatureExtractor):
    """Bag-of-words feature extractor."""
    
    def __init__(self, max_features: int = 5000, 
                 min_df: int = 2, 
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 **kwargs):
        """
        Initialize bag-of-words extractor.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams to extract
        """
        super().__init__(**kwargs)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english'
        )
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'BagOfWordsExtractor':
        """Fit the bag-of-words vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Bag-of-words fitted with {len(self.vectorizer.vocabulary_)} features")
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts to bag-of-words features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out().tolist()
        return []

class TFIDFExtractor(FeatureExtractor):
    """TF-IDF feature extractor."""
    
    def __init__(self, max_features: int = 5000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_idf: bool = True,
                 norm: str = 'l2',
                 **kwargs):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams to extract
            use_idf: Whether to use inverse document frequency
            norm: Normalization method
        """
        super().__init__(**kwargs)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.norm = norm
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            use_idf=use_idf,
            norm=norm,
            stop_words='english'
        )
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'TFIDFExtractor':
        """Fit the TF-IDF vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF fitted with {len(self.vectorizer.vocabulary_)} features")
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out().tolist()
        return []

class Word2VecExtractor(FeatureExtractor):
    """Word2Vec feature extractor."""
    
    def __init__(self, vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 1,
                 workers: int = 4,
                 sg: int = 0,  # 0 for CBOW, 1 for Skip-gram
                 **kwargs):
        """
        Initialize Word2Vec extractor.
        
        Args:
            vector_size: Size of word vectors
            window: Context window size
            min_count: Minimum word count
            workers: Number of worker threads
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
        """
        super().__init__(**kwargs)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
    
    def _tokenize_texts(self, texts: Union[List[str], pd.Series]) -> List[List[str]]:
        """Tokenize texts into sentences of words."""
        tokenized_texts = []
        for text in texts:
            if isinstance(text, str):
                # Simple tokenization - split by whitespace
                tokens = text.lower().split()
                tokenized_texts.append(tokens)
            else:
                tokenized_texts.append([])
        return tokenized_texts
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'Word2VecExtractor':
        """Fit the Word2Vec model."""
        tokenized_texts = self._tokenize_texts(texts)
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        
        self.is_fitted = True
        logger.info(f"Word2Vec fitted with vocabulary size: {len(self.model.wv.key_to_index)}")
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts to Word2Vec features (document vectors)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        
        tokenized_texts = self._tokenize_texts(texts)
        doc_vectors = []
        
        for tokens in tokenized_texts:
            if tokens:
                # Calculate document vector as mean of word vectors
                word_vectors = []
                for word in tokens:
                    if word in self.model.wv:
                        word_vectors.append(self.model.wv[word])
                
                if word_vectors:
                    doc_vector = np.mean(word_vectors, axis=0)
                else:
                    # If no words found in vocabulary, use zero vector
                    doc_vector = np.zeros(self.vector_size)
            else:
                doc_vector = np.zeros(self.vector_size)
            
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a specific word."""
        if self.is_fitted and word in self.model.wv:
            return self.model.wv[word]
        return None

class Doc2VecExtractor(FeatureExtractor):
    """Doc2Vec feature extractor."""
    
    def __init__(self, vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 1,
                 workers: int = 4,
                 dm: int = 1,  # 1 for DM, 0 for DBOW
                 **kwargs):
        """
        Initialize Doc2Vec extractor.
        
        Args:
            vector_size: Size of document vectors
            window: Context window size
            min_count: Minimum word count
            workers: Number of worker threads
            dm: Training algorithm (1=DM, 0=DBOW)
        """
        super().__init__(**kwargs)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.dm = dm
        self.model = None
    
    def _prepare_documents(self, texts: Union[List[str], pd.Series]) -> List[TaggedDocument]:
        """Prepare documents for Doc2Vec training."""
        tagged_docs = []
        for i, text in enumerate(texts):
            if isinstance(text, str):
                tokens = text.lower().split()
                tagged_docs.append(TaggedDocument(tokens, [f'doc_{i}']))
        return tagged_docs
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'Doc2VecExtractor':
        """Fit the Doc2Vec model."""
        tagged_docs = self._prepare_documents(texts)
        
        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            dm=self.dm
        )
        
        self.is_fitted = True
        logger.info(f"Doc2Vec fitted with {len(tagged_docs)} documents")
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts to Doc2Vec features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        
        doc_vectors = []
        for i, text in enumerate(texts):
            if isinstance(text, str):
                tokens = text.lower().split()
                doc_vector = self.model.infer_vector(tokens)
            else:
                doc_vector = np.zeros(self.vector_size)
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)

class FeatureExtractionPipeline:
    """Pipeline for multiple feature extraction methods."""
    
    def __init__(self, extractors: dict):
        """
        Initialize feature extraction pipeline.
        
        Args:
            extractors: Dictionary of {name: extractor} pairs
        """
        self.extractors = extractors
        self.fitted_extractors = {}
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractionPipeline':
        """Fit all extractors."""
        for name, extractor in self.extractors.items():
            logger.info(f"Fitting {name} extractor...")
            self.fitted_extractors[name] = extractor.fit(texts)
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> dict:
        """Transform texts using all fitted extractors."""
        if not self.fitted_extractors:
            raise ValueError("Pipeline must be fitted before transforming")
        
        features = {}
        for name, extractor in self.fitted_extractors.items():
            logger.info(f"Transforming with {name} extractor...")
            features[name] = extractor.transform(texts)
        
        return features
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> dict:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)
    
    def save_models(self, base_path: str):
        """Save all fitted models."""
        for name, extractor in self.fitted_extractors.items():
            filepath = os.path.join(base_path, f"{name}_model.pkl")
            extractor.save_model(filepath)
    
    def load_models(self, base_path: str):
        """Load all models."""
        for name in self.extractors.keys():
            filepath = os.path.join(base_path, f"{name}_model.pkl")
            if os.path.exists(filepath):
                self.fitted_extractors[name] = self.extractors[name].load_model(filepath)

if __name__ == "__main__":
    # Test the feature extractors
    sample_texts = [
        "I love this movie! It's absolutely fantastic!",
        "This is terrible. I hate it.",
        "The movie was okay, nothing special.",
        "Amazing film with great acting.",
        "Disappointing and boring movie."
    ]
    
    # Test Bag-of-Words
    print("Testing Bag-of-Words extractor...")
    bow_extractor = BagOfWordsExtractor(max_features=100)
    bow_features = bow_extractor.fit_transform(sample_texts)
    print(f"BoW features shape: {bow_features.shape}")
    
    # Test TF-IDF
    print("\nTesting TF-IDF extractor...")
    tfidf_extractor = TFIDFExtractor(max_features=100)
    tfidf_features = tfidf_extractor.fit_transform(sample_texts)
    print(f"TF-IDF features shape: {tfidf_features.shape}")
    
    # Test Word2Vec
    print("\nTesting Word2Vec extractor...")
    w2v_extractor = Word2VecExtractor(vector_size=50)
    w2v_features = w2v_extractor.fit_transform(sample_texts)
    print(f"Word2Vec features shape: {w2v_features.shape}")
    
    # Test pipeline
    print("\nTesting feature extraction pipeline...")
    extractors = {
        'bow': BagOfWordsExtractor(max_features=50),
        'tfidf': TFIDFExtractor(max_features=50),
        'w2v': Word2VecExtractor(vector_size=25)
    }
    
    pipeline = FeatureExtractionPipeline(extractors)
    all_features = pipeline.fit_transform(sample_texts)
    
    for name, features in all_features.items():
        print(f"{name} features shape: {features.shape}") 