"""
Simplified demo script for sentiment analysis system.
Avoids Word2Vec to prevent scipy/gensim compatibility issues.
"""

import sys
import os

# Add src to path
sys.path.append('src')

# Try to import heavy dependencies; if unavailable, fall back to a lightweight demo
HAS_DEPS = True
try:
    from data_loader import SentimentDataLoader
    from preprocessing import SentimentSpecificPreprocessor
    from feature_extraction_simple import TFIDFExtractor
    from models import LogisticRegressionClassifier
    from sklearn.metrics import accuracy_score, classification_report
except Exception as _e:
    HAS_DEPS = False
    # We intentionally avoid raising here; a fallback demo will run instead

def main():
    """Run a quick demo of the sentiment analysis system."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS SYSTEM DEMO (Simplified)")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading dataset...")
    if HAS_DEPS:
        data_loader = SentimentDataLoader()
        df = data_loader.load_dataset("imdb", sample_size=50)  # Small sample for demo
        print(f"Loaded {len(df)} samples")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    else:
        # Minimal in-memory sample when dependencies are missing
        print("Dependencies missing: running lightweight fallback demo (no pandas/nltk/sklearn).")
        sample_texts = [
            ("This movie was absolutely fantastic! Great acting and plot.", 'positive'),
            ("I hated every minute of this film. Poor acting.", 'negative'),
            ("This movie was okay. Not great, not terrible.", 'neutral'),
            ("Amazing cinematography and brilliant storytelling.", 'positive'),
            ("Terrible movie with poor character development.", 'negative'),
            ("Average film with decent acting.", 'neutral'),
        ]
        # Convert to a simple list structure similar to a DataFrame
        df = {'text': [t for t, s in sample_texts], 'sentiment': [s for t, s in sample_texts]}
    
    # 2. Preprocess data
    print("\n2. Preprocessing text...")
    if HAS_DEPS:
        preprocessor = SentimentSpecificPreprocessor()
        df['processed_text'] = preprocessor.preprocess_series(df['text'])
    else:
        # Very small preprocessing for fallback (lowercase + remove punctuation)
        import re, string
        def simple_preprocess(text):
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        df['processed_text'] = [simple_preprocess(t) for t in df['text']]
    
    # Show preprocessing example
    print("Preprocessing example:")
    if HAS_DEPS:
        print(f"Original: {df['text'].iloc[0]}")
        print(f"Processed: {df['processed_text'].iloc[0]}")
    else:
        print(f"Original: {df['text'][0]}")
        print(f"Processed: {df['processed_text'][0]}")
    
    # 3. Extract features (TF-IDF only)
    print("\n3. Extracting TF-IDF features...")
    if HAS_DEPS:
        feature_extractor = TFIDFExtractor(max_features=200)
        features = feature_extractor.fit_transform(df['processed_text'])
        print(f"Feature matrix shape: {features.shape}")
    else:
        # Fallback: represent features as the processed text itself (list of strings)
        features = df['processed_text']
        print(f"Using fallback feature representation for {len(features)} samples")
    
    # 4. Split data
    print("\n4. Splitting data...")
    from sklearn.model_selection import train_test_split
    if HAS_DEPS:
        X_train, X_test, y_train, y_test = train_test_split(
            features, df['sentiment'], test_size=0.3, random_state=42, stratify=df['sentiment']
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    else:
        # Simple split for fallback using lists
        split_idx = int(len(features) * 0.7)
        X_train = features[:split_idx]
        X_test = features[split_idx:]
        y_train = df['sentiment'][:split_idx]
        y_test = df['sentiment'][split_idx:]
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
    
    # 5. Train model
    print("\n5. Training Logistic Regression model...")
    if HAS_DEPS:
        model = LogisticRegressionClassifier(C=1.0, max_iter=200)
        model.fit(X_train, y_train)
        print("Model trained successfully!")
    else:
        # Fallback: simple rule-based classifier
        class RuleBasedClassifier:
            def __init__(self):
                self.pos_words = {'love','fantastic','great','amazing','excellent','incredible','masterpiece','outstanding'}
                self.neg_words = {'terrible','hate','awful','disappointing','horrible','worst','boring','waste'}
            def predict(self, X):
                preds = []
                for text in X:
                    tset = set(text.split())
                    pos = len(tset & self.pos_words)
                    neg = len(tset & self.neg_words)
                    if pos > neg:
                        preds.append('positive')
                    elif neg > pos:
                        preds.append('negative')
                    else:
                        preds.append('neutral')
                return preds
            def predict_proba(self, X):
                # crude confidence
                probs = []
                for text in X:
                    tset = set(text.split())
                    pos = len(tset & self.pos_words)
                    neg = len(tset & self.neg_words)
                    total = pos + neg
                    if total == 0:
                        probs.append([0.33,0.33,0.34])
                    elif pos >= neg:
                        probs.append([0.0, pos/total, 1 - pos/total])
                    else:
                        probs.append([neg/total, 0.0, 1 - neg/total])
                return probs

        model = RuleBasedClassifier()
        print("Fallback rule-based model ready")
    
    # 6. Make predictions
    print("\n6. Making predictions...")
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None
    
    # 7. Evaluate model (simplified)
    print("\n7. Evaluating model...")
    if HAS_DEPS:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        # Simple evaluation
        correct = sum(1 for a,b in zip(y_test, y_pred) if a==b)
        accuracy = correct / len(y_test)
        print(f"Accuracy (fallback): {accuracy:.4f} ({correct}/{len(y_test)})")
    
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
        if HAS_DEPS:
            processed = preprocessor.preprocess_text(text)
            # Extract features
            features = feature_extractor.transform([processed])
            # Predict
            prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0])
        else:
            processed = df['processed_text'][0] if len(df['processed_text'])>0 else text.lower()
            prediction = model.predict([simple_preprocess(text)])[0] if not HAS_DEPS else model.predict([processed])[0]
            proba = model.predict_proba([simple_preprocess(text)])[0] if not HAS_DEPS else model.predict_proba([processed])[0]
            confidence = max(proba) if proba is not None else 0.0

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