"""
Improved Streamlit web application for sentiment analysis.
Uses better trained models and ensemble predictions.
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
from feature_extraction_simple import TFIDFExtractor
from models import ModelEnsemble

# Configure page
st.set_page_config(
    page_title="Improved Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ImprovedSentimentAnalysisApp:
    """Improved Streamlit web application for sentiment analysis."""
    
    def __init__(self):
        """Initialize the application."""
        self.data_loader = SentimentDataLoader()
        self.preprocessor = SentimentSpecificPreprocessor()
        self.models = {}
        self.feature_extractor = None
        self.ensemble = None
        self.load_models()
    
    def load_models(self):
        """Load improved models and feature extractors."""
        try:
            # Load feature extractor
            if os.path.exists('models/improved_tfidf_extractor.pkl'):
                with open('models/improved_tfidf_extractor.pkl', 'rb') as f:
                    self.feature_extractor = pickle.load(f)
            
            # Load ensemble
            if os.path.exists('models/improved_ensemble.pkl'):
                with open('models/improved_ensemble.pkl', 'rb') as f:
                    self.ensemble = pickle.load(f)
            
            # Load individual models
            model_files = [
                'models/improved_naive_bayes.pkl',
                'models/improved_svm_linear.pkl',
                'models/improved_svm_rbf.pkl',
                'models/improved_logistic_regression.pkl',
                'models/improved_random_forest.pkl',
                'models/improved_gradient_boosting.pkl'
            ]
            
            for file in model_files:
                if os.path.exists(file):
                    with open(file, 'rb') as f:
                        model_name = os.path.basename(file).replace('.pkl', '').replace('improved_', '')
                        self.models[model_name] = pickle.load(f)
            
            if self.feature_extractor and self.ensemble:
                st.success("✅ Improved models loaded successfully!")
                st.info(f"Loaded {len(self.models)} individual models + ensemble")
            else:
                st.warning("⚠️ Improved models not found. Please run 'python improved_sentiment_analyzer.py' first.")
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please run: python improved_sentiment_analyzer.py")
    
    def predict_sentiment(self, text: str, use_ensemble: bool = True) -> Dict:
        """Predict sentiment with improved accuracy."""
        try:
            if not self.feature_extractor:
                return {'sentiment': 'error', 'probabilities': None, 'confidence': None, 'error': 'Models not loaded'}
            
            # Preprocess text
            processed_text = self.preprocessor.preprocess_text(text)
            
            # Extract features
            features = self.feature_extractor.transform([processed_text])
            
            if use_ensemble and self.ensemble:
                # Use ensemble
                prediction = self.ensemble.predict(features)[0]
                probabilities = self.ensemble.predict_proba(features)[0]
                method = "Ensemble"
            else:
                # Use best individual model
                best_model = self.models.get('logistic_regression', list(self.models.values())[0])
                prediction = best_model.predict(features)[0]
                probabilities = best_model.predict_proba(features)[0]
                method = "Individual Model"
            
            confidence = max(probabilities)
            
            return {
                'sentiment': prediction,
                'probabilities': probabilities,
                'confidence': confidence,
                'method': method
            }
            
        except Exception as e:
            return {'sentiment': 'error', 'probabilities': None, 'confidence': None, 'error': str(e)}
    
    def analyze_batch(self, texts: List[str], use_ensemble: bool = True) -> pd.DataFrame:
        """Analyze a batch of texts."""
        results = []
        
        for text in texts:
            result = self.predict_sentiment(text, use_ensemble)
            results.append({
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'method': result.get('method', 'Unknown')
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
        if 'confidence' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['confidence'], name="Confidence Distribution", nbinsx=20),
                row=1, col=2
            )
        
        fig.update_layout(height=400, title_text="Analysis Results")
        return fig
    
    def run(self):
        """Run the improved Streamlit application."""
        st.title("😊 Improved Sentiment Analysis App")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.header("Settings")
        
        # Model selection
        use_ensemble = st.sidebar.checkbox("Use Ensemble Model", value=True, 
                                          help="Ensemble combines all models for better accuracy")
        
        if not use_ensemble and self.models:
            selected_model = st.sidebar.selectbox(
                "Select Individual Model",
                list(self.models.keys()),
                index=0
            )
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Single Text", "Batch Analysis", "Model Comparison", "About"])
        
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
                        result = self.predict_sentiment(text_input, use_ensemble)
                        
                        if result['sentiment'] == 'error':
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                        else:
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Sentiment", result['sentiment'].title())
                            
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            with col3:
                                st.metric("Method", result.get('method', 'Unknown'))
                            
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
                placeholder="I love this movie!\nThis is terrible.\nThe movie was okay."
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    st.success(f"Loaded {len(df)} texts from file")
                    
                    if st.button("Analyze Batch", type="primary"):
                        with st.spinner("Analyzing batch..."):
                            results_df = self.analyze_batch(df['text'].tolist(), use_ensemble)
                            
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
                                file_name="improved_sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("CSV file must contain a 'text' column")
            
            elif manual_texts.strip():
                texts = [text.strip() for text in manual_texts.split('\n') if text.strip()]
                
                if st.button("Analyze Manual Input", type="primary"):
                    with st.spinner("Analyzing texts..."):
                        results_df = self.analyze_batch(texts, use_ensemble)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(results_df)
                        
                        # Visualization
                        fig = self.create_sentiment_visualization(results_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Model Comparison")
            
            if self.models:
                st.subheader("Individual Model Performance")
                
                # Test with sample texts
                test_texts = [
                    "I love this movie! It's absolutely fantastic!",
                    "This is terrible. I hate it.",
                    "The movie was okay, nothing special."
                ]
                
                comparison_results = []
                
                for text in test_texts:
                    row = {'text': text[:30] + "..."}
                    
                    # Test ensemble
                    ensemble_result = self.predict_sentiment(text, use_ensemble=True)
                    row['ensemble'] = f"{ensemble_result['sentiment']} ({ensemble_result['confidence']:.2%})"
                    
                    # Test individual models
                    for model_name, model in self.models.items():
                        try:
                            processed = self.preprocessor.preprocess_text(text)
                            features = self.feature_extractor.transform([processed])
                            pred = model.predict(features)[0]
                            prob = max(model.predict_proba(features)[0])
                            row[model_name] = f"{pred} ({prob:.2%})"
                        except:
                            row[model_name] = "Error"
                    
                    comparison_results.append(row)
                
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df)
                
                st.info("💡 Tip: Ensemble model usually provides the most reliable predictions!")
            else:
                st.warning("No models loaded for comparison.")
        
        with tab4:
            st.header("About Improved Version")
            
            st.markdown("""
            ## Improved Sentiment Analysis Project
            
            This version addresses the issues with the previous implementation:
            
            ### 🔧 Improvements Made:
            
            1. **Better Dataset**: 
               - Larger, more diverse training data (1000+ samples)
               - More realistic and varied text examples
               - Better balance between positive, negative, and neutral sentiments
            
            2. **Enhanced Feature Extraction**:
               - Increased feature vocabulary (2000 features)
               - Added trigrams for better context capture
               - Improved TF-IDF parameters
            
            3. **Model Ensemble**:
               - Combines multiple models for better accuracy
               - Weighted voting based on individual model performance
               - Reduces overfitting and improves reliability
            
            4. **Multiple Model Types**:
               - Naive Bayes (Multinomial)
               - SVM (Linear and RBF kernels)
               - Logistic Regression
               - Random Forest
               - Gradient Boosting
               - Ensemble (Combined)
            
            ### 🎯 Key Features:
            - **Higher Accuracy**: Better trained models with more data
            - **Ensemble Predictions**: Combines multiple models for reliability
            - **Confidence Scores**: Shows prediction confidence
            - **Model Comparison**: Compare different algorithms
            - **Batch Processing**: Analyze multiple texts at once
            
            ### 📊 Expected Performance:
            - Much better distinction between positive, negative, and neutral texts
            - Higher confidence scores for clear cases
            - More reliable predictions across different text types
            
            ### 🚀 How to Use:
            1. **Single Text**: Enter text for instant analysis
            2. **Batch Analysis**: Upload CSV or enter multiple texts
            3. **Model Comparison**: See how different models perform
            4. **Ensemble Mode**: Use combined model for best results
            """)
            
            if self.models:
                st.subheader("Loaded Models")
                model_info = pd.DataFrame([
                    {
                        'Model': name.replace('_', ' ').title(),
                        'Type': 'Individual',
                        'Status': 'Loaded'
                    }
                    for name in self.models.keys()
                ])
                
                if self.ensemble:
                    model_info = pd.concat([
                        model_info,
                        pd.DataFrame([{
                            'Model': 'Ensemble',
                            'Type': 'Combined',
                            'Status': 'Loaded'
                        }])
                    ])
                
                st.dataframe(model_info)
            else:
                st.warning("No models loaded")

def main():
    """Main function to run the improved Streamlit app."""
    app = ImprovedSentimentAnalysisApp()
    app.run()

if __name__ == "__main__":
    main() 