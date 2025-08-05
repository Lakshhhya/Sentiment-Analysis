"""
Quick training script for sentiment analysis.
Trains models and saves them without complex evaluation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_loader import SentimentDataLoader
from preprocessing import SentimentSpecificPreprocessor
from feature_extraction_simple import BagOfWordsExtractor, TFIDFExtractor
from models import (
    NaiveBayesClassifier, SVMClassifier, LogisticRegressionClassifier,
    RandomForestClassifier, GradientBoostingClassifier, ModelFactory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_quick.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Quick training of sentiment analysis models."""
    print("=" * 60)
    print("QUICK SENTIMENT ANALYSIS TRAINING")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
    # 1. Load data
    print("\n1. Loading dataset...")
    data_loader = SentimentDataLoader()
    df = data_loader.load_dataset("imdb", sample_size=100)
    print(f"Loaded {len(df)} samples")
    
    # 2. Split data
    print("\n2. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        df, test_size=0.2, val_size=0.1, random_state=42
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    preprocessor = SentimentSpecificPreprocessor()
    X_train_processed = preprocessor.preprocess_series(X_train)
    X_val_processed = preprocessor.preprocess_series(X_val)
    X_test_processed = preprocessor.preprocess_series(X_test)
    
    # 4. Extract features
    print("\n4. Extracting features...")
    
    # TF-IDF features
    tfidf_extractor = TFIDFExtractor(max_features=500)
    X_train_tfidf = tfidf_extractor.fit_transform(X_train_processed)
    X_val_tfidf = tfidf_extractor.transform(X_val_processed)
    X_test_tfidf = tfidf_extractor.transform(X_test_processed)
    
    # Save TF-IDF extractor
    tfidf_extractor.save_model('models/tfidf_extractor.pkl')
    print("TF-IDF extractor saved")
    
    # Bag-of-Words features
    bow_extractor = BagOfWordsExtractor(max_features=500)
    X_train_bow = bow_extractor.fit_transform(X_train_processed)
    X_val_bow = bow_extractor.transform(X_val_processed)
    X_test_bow = bow_extractor.transform(X_test_processed)
    
    # Save BoW extractor
    bow_extractor.save_model('models/bow_extractor.pkl')
    print("BoW extractor saved")
    
    # 5. Train models
    print("\n5. Training models...")
    
    # Define models to train
    models_config = {
        'naive_bayes': {'nb_type': 'multinomial'},
        'svm': {'kernel': 'linear', 'C': 1.0},
        'logistic_regression': {'C': 1.0, 'max_iter': 200},
        'random_forest': {'n_estimators': 25},
        'gradient_boosting': {'n_estimators': 25}
    }
    
    # Train on TF-IDF features
    print("Training on TF-IDF features...")
    for model_name, params in models_config.items():
        print(f"Training {model_name}...")
        model = ModelFactory.create_model(model_name, **params)
        model.fit(X_train_tfidf, y_train)
        model.save_model(f'models/{model_name}_tfidf.pkl')
        print(f"  {model_name} saved")
    
    # Train on BoW features
    print("Training on BoW features...")
    for model_name, params in models_config.items():
        print(f"Training {model_name}...")
        model = ModelFactory.create_model(model_name, **params)
        model.fit(X_train_bow, y_train)
        model.save_model(f'models/{model_name}_bow.pkl')
        print(f"  {model_name} saved")
    
    # 6. Quick evaluation
    print("\n6. Quick evaluation...")
    
    # Test one model as example
    test_model = LogisticRegressionClassifier(C=1.0, max_iter=200)
    test_model.fit(X_train_tfidf, y_train)
    y_pred = test_model.predict(X_test_tfidf)
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    print("\nModels saved in 'models/' directory:")
    
    # List saved models
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    for file in sorted(model_files):
        print(f"  - {file}")
    
    print("\nNext steps:")
    print("1. Run 'python -m streamlit run web_app_simple.py' for web interface")
    print("2. Run 'python demo_simple.py' for a quick demo")

if __name__ == "__main__":
    main() 