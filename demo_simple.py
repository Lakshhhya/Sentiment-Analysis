"""
Simplified demo script for sentiment analysis system.
Avoids Word2Vec to prevent scipy/gensim compatibility issues.
"""

import sys
import os

# Add src to path
sys.path.append('src')

from data_loader import SentimentDataLoader
from preprocessing import SentimentSpecificPreprocessor
from feature_extraction_simple import TFIDFExtractor
from models import LogisticRegressionClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    """Run a quick demo of the sentiment analysis system."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS SYSTEM DEMO (Simplified)")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading dataset...")
    data_loader = SentimentDataLoader()
    df = data_loader.load_dataset("imdb", sample_size=50)  # Small sample for demo
    print(f"Loaded {len(df)} samples")
    print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # 2. Preprocess data
    print("\n2. Preprocessing text...")
    preprocessor = SentimentSpecificPreprocessor()
    df['processed_text'] = preprocessor.preprocess_series(df['text'])
    
    # Show preprocessing example
    print("Preprocessing example:")
    print(f"Original: {df['text'].iloc[0]}")
    print(f"Processed: {df['processed_text'].iloc[0]}")
    
    # 3. Extract features (TF-IDF only)
    print("\n3. Extracting TF-IDF features...")
    feature_extractor = TFIDFExtractor(max_features=200)
    features = feature_extractor.fit_transform(df['processed_text'])
    print(f"Feature matrix shape: {features.shape}")
    
    # 4. Split data
    print("\n4. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, df['sentiment'], test_size=0.3, random_state=42, stratify=df['sentiment']
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 5. Train model
    print("\n5. Training Logistic Regression model...")
    model = LogisticRegressionClassifier(C=1.0, max_iter=200)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # 6. Make predictions
    print("\n6. Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 7. Evaluate model (simplified)
    print("\n7. Evaluating model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Test on sample texts
    print("\n8. Testing on sample texts...")
    sample_texts = [
        "I love this movie! It's absolutely fantastic!",
        "This is terrible. I hate it.",
        "The movie was okay, nothing special.",
        "Amazing film with great acting.",
        "Disappointing and boring movie."
    ]
    
    print("Sample predictions:")
    for text in sample_texts:
        # Preprocess
        processed = preprocessor.preprocess_text(text)
        # Extract features
        features = feature_extractor.transform([processed])
        # Predict
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])
        
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {prediction} (confidence: {confidence:.2%})")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python train_model_simple.py' for full training")
    print("2. Run 'python -m streamlit run web_app_simple.py' for web interface")
    print("3. Check 'models/', 'results/', and 'plots/' directories for outputs")

if __name__ == "__main__":
    main() 