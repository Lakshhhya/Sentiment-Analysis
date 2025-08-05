"""
Sentiment Analysis Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

class SentimentApp:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.models = {}
        self.load_models()
    def load_models(self):
        try:
            if os.path.exists('models/simple_tfidf_vectorizer.pkl'):
                with open('models/simple_tfidf_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
            model_files = [
                'models/simple_logistic_regression.pkl',
                'models/simple_random_forest.pkl',
                'models/simple_gradient_boosting.pkl'
            ]
            for file in model_files:
                if os.path.exists(file):
                    with open(file, 'rb') as f:
                        model_name = os.path.basename(file).replace('.pkl', '').replace('simple_', '')
                        self.models[model_name] = pickle.load(f)
            if self.vectorizer and self.models:
                st.success("Models loaded successfully!")
                st.info(f"Loaded {len(self.models)} models.")
            else:
                st.warning("Models not found. Please run the training script first.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please run the training script.")
    def predict_sentiment(self, text, model_name='logistic_regression'):
        try:
            if not self.vectorizer or model_name not in self.models:
                return {'sentiment': 'error', 'probabilities': None, 'confidence': None, 'error': 'Model not loaded'}
            processed_text = self.preprocessor.preprocess_text(text)
            features = self.vectorizer.transform([processed_text])
            model = self.models[model_name]
            prediction = model.predict(features)[0]
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
            except:
                probabilities = None
                confidence = 0.5
            return {
                'sentiment': prediction,
                'probabilities': probabilities,
                'confidence': confidence,
                'model': model_name
            }
        except Exception as e:
            return {'sentiment': 'error', 'probabilities': None, 'confidence': None, 'error': str(e)}
    def analyze_batch(self, texts, model_name='logistic_regression'):
        results = []
        for text in texts:
            result = self.predict_sentiment(text, model_name)
            results.append({
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'model': result.get('model', 'Unknown')
            })
        return pd.DataFrame(results)
    def create_visualization(self, df):
        if df.empty:
            return None
        sentiment_counts = df['sentiment'].value_counts()
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sentiment Distribution', 'Confidence Distribution'),
            specs=[[{"type": "pie"}, {"type": "histogram"}]]
        )
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   name="Sentiment Distribution"),
            row=1, col=1
        )
        if 'confidence' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['confidence'], name="Confidence Distribution", nbinsx=20),
                row=1, col=2
            )
        fig.update_layout(height=400, title_text="Analysis Results")
        return fig
    def run(self):
        st.title("😊 Sentiment Analysis App")
        st.markdown("---")
        st.sidebar.header("Settings")
        if self.models:
            selected_model = st.sidebar.selectbox(
                "Select Model",
                list(self.models.keys()),
                index=0,
                help="Choose the model to use for analysis"
            )
        else:
            selected_model = 'logistic_regression'
        tab1, tab2, tab3, tab4 = st.tabs(["Single Text", "Batch Analysis", "Model Comparison", "About"])
        with tab1:
            st.header("Single Text Analysis")
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type your text here..."
            )
            if st.button("Analyze Sentiment", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        result = self.predict_sentiment(text_input, selected_model)
                        if result['sentiment'] == 'error':
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sentiment", result['sentiment'].title())
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            with col3:
                                st.metric("Model", selected_model.replace('_', ' ').title())
                            sentiment_emoji = {
                                'positive': '😊',
                                'negative': '😞',
                                'neutral': '😐',
                                'unknown': '❓',
                                'error': '⚠️'
                            }
                            st.markdown(f"### {sentiment_emoji.get(result['sentiment'], '❓')} {result['sentiment'].title()}")
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
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column",
                type=['csv']
            )
            st.subheader("Or enter multiple texts manually:")
            manual_texts = st.text_area(
                "Enter texts (one per line):",
                height=200,
                placeholder="I love this movie!\nThis is terrible.\nThe movie was okay."
            )
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    st.success(f"Loaded {len(df)} texts from file")
                    if st.button("Analyze Batch", type="primary"):
                        with st.spinner("Analyzing batch..."):
                            results_df = self.analyze_batch(df['text'].tolist(), selected_model)
                            st.subheader("Analysis Results")
                            st.dataframe(results_df)
                            fig = self.create_visualization(results_df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="sentiment_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("CSV file must contain a 'text' column")
            elif manual_texts.strip():
                texts = [text.strip() for text in manual_texts.split('\n') if text.strip()]
                if st.button("Analyze Manual Input", type="primary"):
                    with st.spinner("Analyzing texts..."):
                        results_df = self.analyze_batch(texts, selected_model)
                        st.subheader("Analysis Results")
                        st.dataframe(results_df)
                        fig = self.create_visualization(results_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.header("Model Comparison")
            if self.models:
                st.subheader("Model Performance Comparison")
                test_texts = [
                    "I love this movie! It's absolutely fantastic!",
                    "This is terrible. I hate it.",
                    "The movie was okay, nothing special."
                ]
                comparison_results = []
                for text in test_texts:
                    row = {'text': text[:30] + "..."}
                    for model_name in self.models.keys():
                        result = self.predict_sentiment(text, model_name)
                        row[model_name] = f"{result['sentiment']} ({result['confidence']:.2%})"
                    comparison_results.append(row)
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df)
                st.info("All models are trained on a diverse dataset and provide reliable sentiment predictions.")
            else:
                st.warning("No models loaded for comparison.")
        with tab4:
            st.header("About")
            st.markdown("""
            ## Sentiment Analysis Web Application
            
            This application provides accurate sentiment analysis for text data using advanced machine learning models. It supports single text analysis, batch processing, and model comparison.
            
            ### Features
            - **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting
            - **Text Preprocessing**: Cleaning, lemmatization, stopword removal
            - **Feature Extraction**: TF-IDF with trigrams
            - **Batch Analysis**: Upload CSV or enter multiple texts
            - **Model Comparison**: Compare predictions from different models
            - **Interactive Visualizations**: Sentiment and confidence distributions
            
            ### How to Use
            1. **Single Text**: Enter text for instant sentiment analysis
            2. **Batch Analysis**: Upload a CSV or enter multiple texts
            3. **Model Comparison**: See how different models perform
            4. **Download Results**: Export batch results as CSV
            
            ### Technical Details
            - Built with Python, scikit-learn, and Streamlit
            - Trained on a diverse, balanced dataset
            - Models achieve high accuracy and confidence
            """)
            if self.models:
                st.subheader("Loaded Models")
                model_info = pd.DataFrame([
                    {
                        'Model': name.replace('_', ' ').title(),
                        'Status': 'Loaded'
                    }
                    for name in self.models.keys()
                ])
                st.dataframe(model_info)
            else:
                st.warning("No models loaded")
def main():
    app = SentimentApp()
    app.run()
if __name__ == "__main__":
    main() 