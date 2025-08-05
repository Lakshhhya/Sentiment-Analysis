"""
Simplified Streamlit web application for sentiment analysis.
Avoids Word2Vec to prevent scipy/gensim compatibility issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append('src')

from data_loader import SentimentDataLoader
from preprocessing import SentimentSpecificPreprocessor
from feature_extraction_simple import BagOfWordsExtractor, TFIDFExtractor
from models import ModelFactory

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleSentimentAnalysisApp:
    """Simplified Streamlit web application for sentiment analysis."""
    
    def __init__(self):
        """Initialize the application."""
        self.data_loader = SentimentDataLoader()
        self.preprocessor = SentimentSpecificPreprocessor()
        self.models = {}
        self.feature_extractors = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models and feature extractors."""
        try:
            # Load feature extractors (only BoW and TF-IDF)
            extractor_paths = {
                'bow': 'models/bow_extractor.pkl',
                'tfidf': 'models/tfidf_extractor.pkl'
            }
            
            for name, path in extractor_paths.items():
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.feature_extractors[name] = pickle.load(f)
            
            # Load models (only BoW and TF-IDF based models)
            model_paths = [
                'models/naive_bayes_bow.pkl',
                'models/svm_bow.pkl',
                'models/logistic_regression_bow.pkl',
                'models/naive_bayes_tfidf.pkl',
                'models/svm_tfidf.pkl',
                'models/logistic_regression_tfidf.pkl'
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        model_name = os.path.basename(path).replace('.pkl', '')
                        self.models[model_name] = pickle.load(f)
            
            st.success(f"Loaded {len(self.models)} models and {len(self.feature_extractors)} feature extractors")
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please run the training script first: `python train_quick.py`")
    
    def predict_sentiment(self, text: str, model_name: str) -> Dict:
        """Predict sentiment for a given text using specified model."""
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess_text(text)
            
            # Extract features
            feature_name = model_name.split('_')[-1]  # Extract feature type from model name
            if feature_name in self.feature_extractors:
                features = self.feature_extractors[feature_name].transform([processed_text])
                
                # Make prediction
                if model_name in self.models:
                    model = self.models[model_name]
                    prediction = model.predict(features)[0]
                    probabilities = None
                    try:
                        probabilities = model.predict_proba(features)[0]
                    except:
                        pass
                    
                    return {
                        'sentiment': prediction,
                        'probabilities': probabilities,
                        'confidence': max(probabilities) if probabilities is not None else None
                    }
            
            return {'sentiment': 'unknown', 'probabilities': None, 'confidence': None}
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return {'sentiment': 'error', 'probabilities': None, 'confidence': None}
    
    def analyze_batch(self, texts: List[str], model_name: str) -> pd.DataFrame:
        """Analyze a batch of texts."""
        results = []
        
        for i, text in enumerate(texts):
            result = self.predict_sentiment(text, model_name)
            results.append({
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence']
            })
        
        return pd.DataFrame(results)
    
    def create_sentiment_visualization(self, df: pd.DataFrame):
        """Create sentiment visualization."""
        if df.empty:
            return None
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sentiment Distribution', 'Confidence Distribution'),
            specs=[[{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   name="Sentiment Distribution"),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df['confidence'], name="Confidence Distribution", nbinsx=20),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Analysis Results")
        return fig
    
    def run(self):
        """Run the Streamlit application."""
        st.title("😊 Sentiment Analysis App (Simplified)")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.header("Settings")
        
        # Model selection
        available_models = list(self.models.keys())
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Select Model",
                available_models,
                index=0
            )
        else:
            st.sidebar.error("No trained models found!")
            st.info("Please run the training script first: `python train_quick.py`")
            return
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Single Text", "Batch Analysis", "Dataset Analysis", "About"])
        
        with tab1:
            st.header("Single Text Analysis")
            
            # Text input
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type your text here..."
            )
            
            if st.button("Analyze Sentiment", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        result = self.predict_sentiment(text_input, selected_model)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment", result['sentiment'].title())
                        
                        with col2:
                            if result['confidence'] is not None:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            else:
                                st.metric("Confidence", "N/A")
                        
                        with col3:
                            st.metric("Model", selected_model.replace('_', ' ').title())
                        
                        # Sentiment indicator
                        sentiment_emoji = {
                            'positive': '😊',
                            'negative': '😞',
                            'neutral': '😐',
                            'unknown': '❓',
                            'error': '⚠️'
                        }
                        
                        st.markdown(f"### {sentiment_emoji.get(result['sentiment'], '❓')} {result['sentiment'].title()}")
                        
                        # Probabilities
                        if result['probabilities'] is not None:
                            st.subheader("Sentiment Probabilities")
                            prob_df = pd.DataFrame({
                                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                                'Probability': result['probabilities']
                            })
                            st.bar_chart(prob_df.set_index('Sentiment'))
                else:
                    st.warning("Please enter some text to analyze.")
        
        with tab2:
            st.header("Batch Analysis")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column",
                type=['csv']
            )
            
            # Manual input
            st.subheader("Or enter multiple texts manually:")
            manual_texts = st.text_area(
                "Enter texts (one per line):",
                height=200,
                placeholder="Text 1\nText 2\nText 3..."
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    st.success(f"Loaded {len(df)} texts from file")
                    
                    if st.button("Analyze Batch", type="primary"):
                        with st.spinner("Analyzing batch..."):
                            results_df = self.analyze_batch(df['text'].tolist(), selected_model)
                            
                            # Display results
                            st.subheader("Analysis Results")
                            st.dataframe(results_df)
                            
                            # Visualization
                            fig = self.create_sentiment_visualization(results_df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("CSV file must contain a 'text' column")
            
            elif manual_texts.strip():
                texts = [text.strip() for text in manual_texts.split('\n') if text.strip()]
                
                if st.button("Analyze Manual Input", type="primary"):
                    with st.spinner("Analyzing texts..."):
                        results_df = self.analyze_batch(texts, selected_model)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(results_df)
                        
                        # Visualization
                        fig = self.create_sentiment_visualization(results_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Dataset Analysis")
            
            # Dataset selection
            dataset_option = st.selectbox(
                "Select Dataset",
                ["IMDB Sample"],  # Simplified to avoid Sentiment140 download issues
                index=0
            )
            
            sample_size = st.slider("Sample Size", 10, 100, 30)
            
            if st.button("Load and Analyze Dataset", type="primary"):
                with st.spinner("Loading dataset..."):
                    try:
                        df = self.data_loader.load_dataset("imdb", sample_size)
                        
                        st.success(f"Loaded {len(df)} samples from {dataset_option}")
                        
                        # Dataset overview
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Dataset Overview")
                            st.write(f"Total samples: {len(df)}")
                            st.write(f"Sentiment distribution:")
                            st.write(df['sentiment'].value_counts())
                        
                        with col2:
                            st.subheader("Sample Texts")
                            st.dataframe(df.head(10))
                        
                        # Analyze with selected model
                        st.subheader("Model Analysis")
                        with st.spinner("Analyzing with model..."):
                            # Use a sample for analysis to avoid long processing
                            sample_df = df.sample(min(20, len(df)), random_state=42)
                            results_df = self.analyze_batch(sample_df['text'].tolist(), selected_model)
                            
                            # Compare with true labels
                            results_df['true_sentiment'] = sample_df['sentiment'].values
                            results_df['correct'] = results_df['sentiment'] == results_df['true_sentiment']
                            
                            accuracy = results_df['correct'].mean()
                            st.metric("Accuracy", f"{accuracy:.2%}")
                            
                            # Confusion matrix
                            st.subheader("Confusion Matrix")
                            confusion = pd.crosstab(
                                results_df['true_sentiment'], 
                                results_df['sentiment'], 
                                margins=True
                            )
                            st.dataframe(confusion)
                        
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
        
        with tab4:
            st.header("About")
            
            st.markdown("""
            ## Sentiment Analysis Project (Simplified Version)
            
            This is a simplified version of the sentiment analysis application that avoids 
            Word2Vec dependencies to prevent compatibility issues.
            
            ### Features:
            - **Multiple Models**: Naive Bayes, Support Vector Machine (SVM), and Logistic Regression
            - **Feature Extraction**: Bag-of-words and TF-IDF (Word2Vec disabled)
            - **Real-time Analysis**: Interactive web interface for instant sentiment analysis
            - **Batch Processing**: Analyze multiple texts at once
            - **Visualization**: Interactive charts and graphs
            
            ### AI Concepts Implemented:
            1. **Natural Language Processing (NLP)**: Text preprocessing, tokenization, stemming
            2. **Machine Learning**: Supervised learning for text classification
            3. **Feature Engineering**: Text representation techniques
            4. **Model Evaluation**: Comprehensive performance metrics
                        
            ### Installation
            1. Clone the repository
            2. Install dependencies:
            ```bash
            pip install -r requirements.txt
            ```
            3. Download NLTK data:
            ```python
            python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
            ```

            ## Usage
            1. **Train Models**: Run `python train_model_simple.py` to train all models
            2. **Web Interface**: Run `python -m streamlit run web_app.py` for interactive demo
            3. **Notebooks**: Explore `notebooks/` for detailed analysis
            
            ### How to Use:
            1. **Single Text**: Enter text in the first tab for instant analysis
            2. **Batch Analysis**: Upload a CSV file or enter multiple texts manually
            3. **Dataset Analysis**: Test the model on sample datasets
            4. **Model Selection**: Choose different models from the sidebar
            
            ### Technical Details:
            - Built with Python, scikit-learn, and Streamlit
            - Supports IMDB sample dataset
            - Implements Bag-of-Words and TF-IDF feature extraction
            - Provides comprehensive evaluation metrics
            """)
            
            st.subheader("Model Information")
            if self.models:
                model_info = pd.DataFrame([
                    {
                        'Model': name.replace('_', ' ').title(),
                        'Feature Type': name.split('_')[-1].upper(),
                        'Status': 'Loaded'
                    }
                    for name in self.models.keys()
                ])
                st.dataframe(model_info)
            else:
                st.warning("No models loaded")

def main():
    """Main function to run the Streamlit app."""
    app = SimpleSentimentAnalysisApp()
    app.run()

if __name__ == "__main__":
    main() 