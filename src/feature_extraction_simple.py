"""
Simplified feature extraction for sentiment analysis.
Avoids Word2Vec to prevent scipy/gensim compatibility issues.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self, **kwargs):
        """Initialize the feature extractor."""
        self.vectorizer = None
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """Fit the feature extractor on the given texts."""
        raise NotImplementedError
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to features."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transforming")
        raise NotImplementedError
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the extractor and transform the texts."""
        return self.fit(texts).transform(texts)
    
    def save_model(self, filepath: str):
        """Save the feature extractor to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'FeatureExtractor':
        """Load a feature extractor from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class BagOfWordsExtractor(FeatureExtractor):
    """Bag-of-Words feature extractor."""
    
    def __init__(self, max_features: int = 1000, **kwargs):
        """Initialize the Bag-of-Words extractor."""
        super().__init__(**kwargs)
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    def fit(self, texts: List[str]) -> 'BagOfWordsExtractor':
        """Fit the Bag-of-Words extractor."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to Bag-of-Words features."""
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()

class TFIDFExtractor(FeatureExtractor):
    """TF-IDF feature extractor."""
    
    def __init__(self, max_features: int = 1000, **kwargs):
        """Initialize the TF-IDF extractor."""
        super().__init__(**kwargs)
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    
    def fit(self, texts: List[str]) -> 'TFIDFExtractor':
        """Fit the TF-IDF extractor."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()

class FeatureExtractionPipeline:
    """Pipeline for multiple feature extractors."""
    
    def __init__(self, extractors: Dict[str, FeatureExtractor]):
        """Initialize the pipeline with multiple extractors."""
        self.extractors = extractors
        self.features = {}
    
    def fit_transform(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Fit and transform texts using all extractors."""
        for name, extractor in self.extractors.items():
            logger.info(f"Extracting features using {name}")
            self.features[name] = extractor.fit_transform(texts)
        
        return self.features
    
    def transform(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Transform texts using all fitted extractors."""
        features = {}
        for name, extractor in self.extractors.items():
            features[name] = extractor.transform(texts)
        
        return features
    
    def save_pipeline(self, directory: str):
        """Save all extractors in the pipeline."""
        os.makedirs(directory, exist_ok=True)
        for name, extractor in self.extractors.items():
            extractor.save_model(os.path.join(directory, f"{name}_extractor.pkl"))
    
    @classmethod
    def load_pipeline(cls, directory: str, extractor_names: List[str]) -> 'FeatureExtractionPipeline':
        """Load a pipeline from saved extractors."""
        extractors = {}
        for name in extractor_names:
            filepath = os.path.join(directory, f"{name}_extractor.pkl")
            if os.path.exists(filepath):
                extractors[name] = FeatureExtractor.load_model(filepath)
        
        return cls(extractors) 