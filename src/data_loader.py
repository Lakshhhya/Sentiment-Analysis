"""
Data loader module for sentiment analysis datasets.
Supports Sentiment140 and IMDB movie review datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataLoader:
    """Data loader for sentiment analysis datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_sentiment140(self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Download and load Sentiment140 dataset.
        
        Args:
            sample_size: Number of samples to load (for faster processing)
            
        Returns:
            DataFrame with 'text' and 'sentiment' columns
        """
        url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        zip_path = os.path.join(self.data_dir, "sentiment140.zip")
        csv_path = os.path.join(self.data_dir, "training.1600000.processed.noemoticon.csv")
        
        if not os.path.exists(csv_path):
            logger.info("Downloading Sentiment140 dataset...")
            response = requests.get(url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up zip file
            os.remove(zip_path)
        
        # Load data
        columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(csv_path, encoding='latin-1', names=columns)
        
        # Map sentiment: 0 = negative, 4 = positive
        df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})
        
        # Sample data for faster processing
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Loaded {len(df)} samples from Sentiment140")
        return df[['text', 'sentiment']]
    
    def load_imdb_sample(self) -> pd.DataFrame:
        """
        Load a sample IMDB dataset (simulated for demo purposes).
        In a real scenario, you would download the full IMDB dataset.
        
        Returns:
            DataFrame with 'text' and 'sentiment' columns
        """
        # Create sample IMDB-like data
        positive_reviews = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "I loved every minute of this film. Highly recommended!",
            "Amazing cinematography and brilliant storytelling.",
            "One of the best movies I've ever seen. Outstanding performance.",
            "Incredible film with perfect direction and acting.",
            "This is a masterpiece. The acting is superb.",
            "Wonderful movie with great character development.",
            "Excellent film that kept me engaged throughout.",
            "Fantastic storytelling with beautiful visuals.",
            "A must-watch movie with outstanding performances."
        ]
        
        negative_reviews = [
            "This movie was terrible. Waste of time and money.",
            "I hated every minute of this film. Poor acting.",
            "Awful cinematography and boring plot.",
            "One of the worst movies I've ever seen. Terrible performance.",
            "Disappointing film with bad direction and acting.",
            "This is a disaster. The acting is horrible.",
            "Terrible movie with poor character development.",
            "Bad film that bored me throughout.",
            "Poor storytelling with ugly visuals.",
            "A waste of time with terrible performances."
        ]
        
        neutral_reviews = [
            "This movie was okay. Not great, not terrible.",
            "Average film with decent acting.",
            "The plot was predictable but watchable.",
            "It's a typical movie, nothing special.",
            "Decent film with mixed performances.",
            "The movie was fine, nothing remarkable.",
            "Average storytelling with standard visuals.",
            "It's watchable but forgettable.",
            "The acting was adequate, plot was mediocre.",
            "A standard film with nothing outstanding."
        ]
        
        data = []
        for review in positive_reviews:
            data.append({'text': review, 'sentiment': 'positive'})
        for review in negative_reviews:
            data.append({'text': review, 'sentiment': 'negative'})
        for review in neutral_reviews:
            data.append({'text': review, 'sentiment': 'neutral'})
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample IMDB dataset with {len(df)} reviews")
        return df
    
    def load_dataset(self, dataset_name: str = "sentiment140", 
                    sample_size: int = 10000) -> pd.DataFrame:
        """
        Load the specified dataset.
        
        Args:
            dataset_name: Name of dataset ('sentiment140' or 'imdb')
            sample_size: Number of samples to load
            
        Returns:
            DataFrame with 'text' and 'sentiment' columns
        """
        if dataset_name.lower() == "sentiment140":
            return self.download_sentiment140(sample_size)
        elif dataset_name.lower() == "imdb":
            return self.load_imdb_sample()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.1, random_state: int = 42) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df['text']
        y = df['sentiment']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, df: pd.DataFrame) -> dict:
        """Get the distribution of sentiment classes."""
        return df['sentiment'].value_counts().to_dict()

if __name__ == "__main__":
    # Test the data loader
    loader = SentimentDataLoader()
    
    # Load Sentiment140 dataset
    print("Loading Sentiment140 dataset...")
    df_140 = loader.load_dataset("sentiment140", sample_size=5000)
    print(f"Sentiment140 shape: {df_140.shape}")
    print(f"Class distribution: {loader.get_class_distribution(df_140)}")
    
    # Load IMDB dataset
    print("\nLoading IMDB dataset...")
    df_imdb = loader.load_dataset("imdb")
    print(f"IMDB shape: {df_imdb.shape}")
    print(f"Class distribution: {loader.get_class_distribution(df_imdb)}") 