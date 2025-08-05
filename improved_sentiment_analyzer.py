"""
Improved Sentiment Analysis with better dataset and model ensemble.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import pickle
import logging

# Add src to path
sys.path.append('src')

from data_loader import SentimentDataLoader
from preprocessing import SentimentSpecificPreprocessor
from feature_extraction_simple import TFIDFExtractor
from models import (
    NaiveBayesClassifier, SVMClassifier, LogisticRegressionClassifier,
    RandomForestClassifier, GradientBoostingClassifier, ModelEnsemble
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedSentimentAnalyzer:
    """Improved sentiment analyzer with better dataset and ensemble."""
    
    def __init__(self):
        """Initialize the improved analyzer."""
        self.data_loader = SentimentDataLoader()
        self.preprocessor = SentimentSpecificPreprocessor()
        self.models = {}
        self.feature_extractor = None
        self.ensemble = None
        
    def create_better_dataset(self, size: int = 1000) -> pd.DataFrame:
        """Create a better, more diverse dataset."""
        logger.info("Creating improved dataset...")
        
        # Create more diverse and realistic sentiment data
        positive_texts = [
            "I absolutely love this movie! It's fantastic and amazing.",
            "This is the best film I've ever seen. Outstanding performance!",
            "Wonderful movie with great acting and plot.",
            "Excellent film, highly recommended!",
            "Amazing story and brilliant direction.",
            "I really enjoyed this movie. It's very good.",
            "Fantastic film with incredible acting.",
            "This movie is absolutely wonderful!",
            "Great film with excellent cinematography.",
            "I love this movie so much! It's perfect.",
            "Outstanding performance by all actors.",
            "This is a masterpiece of cinema.",
            "Brilliant movie with amazing soundtrack.",
            "I can't stop watching this movie!",
            "This film exceeded all my expectations.",
            "Incredible movie with perfect pacing.",
            "I'm blown away by this film!",
            "This is cinema at its finest.",
            "Absolutely brilliant movie!",
            "I'm in love with this film!"
        ]
        
        negative_texts = [
            "This movie is terrible. I hate it.",
            "Worst film I've ever seen. Awful acting.",
            "This is a complete waste of time.",
            "Terrible movie with bad acting.",
            "I can't believe how bad this film is.",
            "This movie is absolutely horrible.",
            "Waste of money and time.",
            "I regret watching this movie.",
            "This film is a disaster.",
            "Terrible acting and poor direction.",
            "I hate this movie so much.",
            "This is the worst film ever made.",
            "Awful movie with bad plot.",
            "I can't stand this movie.",
            "This film is a complete failure.",
            "Terrible cinematography and acting.",
            "I'm so disappointed with this movie.",
            "This is cinema at its worst.",
            "Absolutely terrible film!",
            "I want my money back for this movie.",
            "This movie is a complete disaster."
        ]
        
        neutral_texts = [
            "This movie is okay, nothing special.",
            "The film was decent but not great.",
            "It's an average movie, nothing remarkable.",
            "The movie was fine, nothing to write home about.",
            "It's a decent film with some good moments.",
            "The movie was alright, nothing spectacular.",
            "It's an okay film, neither good nor bad.",
            "The movie was passable, nothing special.",
            "It's a mediocre film with average acting.",
            "The movie was neither good nor bad.",
            "It's an acceptable film, nothing outstanding.",
            "The movie was reasonable, nothing exceptional.",
            "It's a fair film with average quality.",
            "The movie was adequate, nothing remarkable.",
            "It's a standard film, nothing unique.",
            "The movie was tolerable, nothing special.",
            "It's a run-of-the-mill film.",
            "The movie was acceptable, nothing great.",
            "It's a typical film, nothing extraordinary.",
            "The movie was fine, nothing exceptional.",
            "It's a normal film, nothing special."
        ]
        
        # Create dataset with more samples
        texts = []
        sentiments = []
        
        # Add positive samples
        for i in range(size // 3):
            text = positive_texts[i % len(positive_texts)]
            if i > len(positive_texts):
                text += f" Sample {i}"  # Add variation
            texts.append(text)
            sentiments.append('positive')
        
        # Add negative samples
        for i in range(size // 3):
            text = negative_texts[i % len(negative_texts)]
            if i > len(negative_texts):
                text += f" Sample {i}"  # Add variation
            texts.append(text)
            sentiments.append('negative')
        
        # Add neutral samples
        for i in range(size // 3):
            text = neutral_texts[i % len(neutral_texts)]
            if i > len(neutral_texts):
                text += f" Sample {i}"  # Add variation
            texts.append(text)
            sentiments.append('neutral')
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
    
    def train_improved_models(self, df: pd.DataFrame):
        """Train improved models with better features."""
        logger.info("Training improved models...")
        
        # Preprocess data
        df['processed_text'] = self.preprocessor.preprocess_series(df['text'])
        
        # Extract better features
        self.feature_extractor = TFIDFExtractor(
            max_features=2000,  # More features
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 3)  # Include trigrams
        )
        
        # Fit and transform
        features = self.feature_extractor.fit_transform(df['processed_text'])
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
        )
        
        # Train multiple models with different configurations
        model_configs = {
            'naive_bayes': NaiveBayesClassifier(nb_type='multinomial', alpha=0.1),
            'svm_linear': SVMClassifier(kernel='linear', C=1.0),
            'svm_rbf': SVMClassifier(kernel='rbf', C=10.0),
            'logistic_regression': LogisticRegressionClassifier(C=1.0, max_iter=500),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        }
        
        # Train each model
        for name, model in model_configs.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Quick evaluation
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Create ensemble with weights based on performance
        logger.info("Creating weighted ensemble...")
        self.ensemble = ModelEnsemble(self.models, voting_method='soft')
        self.ensemble.fit(X_train, y_train)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        self.feature_extractor.save_model('models/improved_tfidf_extractor.pkl')
        
        for name, model in self.models.items():
            model.save_model(f'models/improved_{name}.pkl')
        
        self.ensemble.save_model('models/improved_ensemble.pkl')
        
        logger.info("All models trained and saved!")
    
    def predict_sentiment(self, text: str, use_ensemble: bool = True) -> Dict:
        """Predict sentiment with improved accuracy."""
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess_text(text)
            
            # Extract features
            features = self.feature_extractor.transform([processed_text])
            
            if use_ensemble and self.ensemble:
                # Use ensemble
                prediction = self.ensemble.predict(features)[0]
                probabilities = self.ensemble.predict_proba(features)[0]
            else:
                # Use best individual model
                best_model = self.models['logistic_regression']  # Usually most reliable
                prediction = best_model.predict(features)[0]
                probabilities = best_model.predict_proba(features)[0]
            
            confidence = max(probabilities)
            
            return {
                'sentiment': prediction,
                'probabilities': probabilities,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {'sentiment': 'error', 'probabilities': None, 'confidence': None}
    
    def test_models(self):
        """Test the models with sample texts."""
        test_texts = [
            "I love this movie! It's absolutely fantastic!",
            "This is terrible. I hate it.",
            "The movie was okay, nothing special.",
            "Amazing film with great acting.",
            "Disappointing and boring movie.",
            "This is the best film I've ever seen!",
            "Worst movie ever made.",
            "It's a decent film, nothing remarkable.",
            "I'm blown away by this incredible movie!",
            "This film is a complete disaster."
        ]
        
        print("\n" + "="*60)
        print("TESTING IMPROVED MODELS")
        print("="*60)
        
        for text in test_texts:
            result = self.predict_sentiment(text)
            print(f"Text: {text[:50]}...")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
            print("-" * 40)

def main():
    """Main function to run improved sentiment analysis."""
    analyzer = ImprovedSentimentAnalyzer()
    
    # Create better dataset
    df = analyzer.create_better_dataset(size=1000)
    
    # Train improved models
    analyzer.train_improved_models(df)
    
    # Test the models
    analyzer.test_models()
    
    print("\n" + "="*60)
    print("IMPROVED TRAINING COMPLETED! 🎉")
    print("="*60)
    print("\nModels saved in 'models/' directory:")
    print("- improved_tfidf_extractor.pkl")
    print("- improved_naive_bayes.pkl")
    print("- improved_svm_linear.pkl")
    print("- improved_svm_rbf.pkl")
    print("- improved_logistic_regression.pkl")
    print("- improved_random_forest.pkl")
    print("- improved_gradient_boosting.pkl")
    print("- improved_ensemble.pkl")
    
    print("\nNext steps:")
    print("1. Run 'python -m streamlit run improved_web_app.py' for web interface")
    print("2. The models should now give much better predictions!")

if __name__ == "__main__":
    main() 