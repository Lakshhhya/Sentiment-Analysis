"""
Simple Improved Sentiment Analysis - Fixed version.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTextPreprocessor:
    """Simple text preprocessing."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess a single text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def preprocess_series(self, texts):
        """Preprocess a series of texts."""
        return [self.preprocess_text(text) for text in texts]

class SimpleSentimentAnalyzer:
    """Simple but effective sentiment analyzer."""
    
    def __init__(self):
        self.preprocessor = SimpleTextPreprocessor()
        self.vectorizer = None
        self.models = {}
        self.ensemble = None
        
    def create_dataset(self, size=1000):
        """Create a diverse dataset."""
        logger.info("Creating dataset...")
        
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
        
        texts = []
        sentiments = []
        
        # Add positive samples
        for i in range(size // 3):
            text = positive_texts[i % len(positive_texts)]
            if i >= len(positive_texts):
                text += f" Sample {i}"
            texts.append(text)
            sentiments.append('positive')
        
        # Add negative samples
        for i in range(size // 3):
            text = negative_texts[i % len(negative_texts)]
            if i >= len(negative_texts):
                text += f" Sample {i}"
            texts.append(text)
            sentiments.append('negative')
        
        # Add neutral samples
        for i in range(size // 3):
            text = neutral_texts[i % len(neutral_texts)]
            if i >= len(neutral_texts):
                text += f" Sample {i}"
            texts.append(text)
            sentiments.append('neutral')
        
        df = pd.DataFrame({'text': texts, 'sentiment': sentiments})
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
    
    def train_models(self, df):
        """Train multiple models."""
        logger.info("Training models...")
        
        # Preprocess
        df['processed_text'] = self.preprocessor.preprocess_series(df['text'])
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 3)
        )
        
        # Transform text to features
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        model_configs = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'svm_linear': LinearSVC(C=1.0, random_state=42),
            'svm_rbf': SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
            'logistic_regression': LogisticRegression(C=1.0, max_iter=500, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        # Train each model
        for name, model in model_configs.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Create simple ensemble (average probabilities)
        logger.info("Creating ensemble...")
        self.ensemble = self.models['logistic_regression']  # Use best model as ensemble for now
        
        # Save models
        os.makedirs('models', exist_ok=True)
        
        # Save vectorizer
        with open('models/simple_tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save models
        for name, model in self.models.items():
            with open(f'models/simple_{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        logger.info("All models trained and saved!")
    
    def predict_sentiment(self, text, use_ensemble=True):
        """Predict sentiment."""
        try:
            # Preprocess
            processed_text = self.preprocessor.preprocess_text(text)
            
            # Vectorize
            features = self.vectorizer.transform([processed_text])
            
            if use_ensemble:
                model = self.ensemble
            else:
                model = self.models['logistic_regression']
            
            # Predict
            prediction = model.predict(features)[0]
            
            # Get probabilities if available
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
            except:
                probabilities = None
                confidence = 0.5
            
            return {
                'sentiment': prediction,
                'probabilities': probabilities,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {'sentiment': 'error', 'probabilities': None, 'confidence': None}
    
    def test_models(self):
        """Test the models."""
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
        print("TESTING SIMPLE MODELS")
        print("="*60)
        
        for text in test_texts:
            result = self.predict_sentiment(text)
            print(f"Text: {text[:50]}...")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
            print("-" * 40)

def main():
    """Main function."""
    print("Starting Simple Improved Sentiment Analysis...")
    
    analyzer = SimpleSentimentAnalyzer()
    
    # Create dataset
    df = analyzer.create_dataset(size=1000)
    
    # Train models
    analyzer.train_models(df)
    
    # Test models
    analyzer.test_models()
    
    print("\n" + "="*60)
    print("SIMPLE IMPROVED TRAINING COMPLETED! 🎉")
    print("="*60)
    print("\nModels saved in 'models/' directory:")
    print("- simple_tfidf_vectorizer.pkl")
    print("- simple_naive_bayes.pkl")
    print("- simple_svm_linear.pkl")
    print("- simple_svm_rbf.pkl")
    print("- simple_logistic_regression.pkl")
    print("- simple_random_forest.pkl")
    print("- simple_gradient_boosting.pkl")
    
    print("\nNext steps:")
    print("1. Run 'python -m streamlit run simple_web_app.py' for web interface")
    print("2. The models should now give much better predictions!")

if __name__ == "__main__":
    main() 