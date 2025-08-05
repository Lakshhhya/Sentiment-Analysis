"""
Visualization module for sentiment analysis.
Includes word clouds, sentiment distribution plots, and model performance visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """Comprehensive visualizer for sentiment analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize sentiment visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#FFD700',   # Gold
            'overall': '#4682B4'    # Steel Blue
        }
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, 
                                  sentiment_col: str = 'sentiment',
                                  title: str = "Sentiment Distribution",
                                  save_path: Optional[str] = None):
        """
        Plot sentiment distribution.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_col: Name of sentiment column
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Count sentiments
        sentiment_counts = df[sentiment_col].value_counts()
        
        # Create bar plot
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[self.colors.get(sent, self.colors['overall']) 
                            for sent in sentiment_counts.index])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sentiment_pie(self, df: pd.DataFrame, 
                          sentiment_col: str = 'sentiment',
                          title: str = "Sentiment Distribution",
                          save_path: Optional[str] = None):
        """
        Plot sentiment distribution as pie chart.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_col: Name of sentiment column
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Count sentiments
        sentiment_counts = df[sentiment_col].value_counts()
        
        # Create pie chart
        colors = [self.colors.get(sent, self.colors['overall']) 
                 for sent in sentiment_counts.index]
        
        wedges, texts, autotexts = plt.pie(sentiment_counts.values, 
                                          labels=sentiment_counts.index,
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_wordcloud(self, texts: List[str], 
                        sentiment: Optional[str] = None,
                        title: str = "Word Cloud",
                        save_path: Optional[str] = None):
        """
        Create word cloud from text data.
        
        Args:
            texts: List of text documents
            sentiment: Sentiment label for coloring
            title: Plot title
            save_path: Path to save the plot
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=200,
            colormap='viridis' if sentiment is None else None,
            color_func=lambda *args, **kwargs: self.colors.get(sentiment, self.colors['overall']) if sentiment else None,
            random_state=42
        ).generate(combined_text)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sentiment_wordclouds(self, df: pd.DataFrame, 
                                text_col: str = 'text',
                                sentiment_col: str = 'sentiment',
                                save_path: Optional[str] = None):
        """
        Create word clouds for each sentiment category.
        
        Args:
            df: DataFrame with text and sentiment data
            text_col: Name of text column
            sentiment_col: Name of sentiment column
            save_path: Path to save the plot
        """
        sentiments = df[sentiment_col].unique()
        n_sentiments = len(sentiments)
        
        fig, axes = plt.subplots(1, n_sentiments, figsize=(6*n_sentiments, 6))
        if n_sentiments == 1:
            axes = [axes]
        
        for i, sentiment in enumerate(sentiments):
            sentiment_texts = df[df[sentiment_col] == sentiment][text_col].tolist()
            
            if sentiment_texts:
                # Create word cloud
                combined_text = ' '.join(sentiment_texts)
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=100,
                    color_func=lambda *args, **kwargs: self.colors.get(sentiment, self.colors['overall']),
                    random_state=42
                ).generate(combined_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.title()} Sentiment', fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                            metrics: List[str] = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                            title: str = "Model Performance Comparison",
                            save_path: Optional[str] = None):
        """
        Plot model performance comparison.
        
        Args:
            results: Dictionary of {model_name: metrics} pairs
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot
        """
        # Prepare data
        models = list(results.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [results[model].get(metric, 0) for model in models]
                
                bars = axes[i].bar(models, values, 
                                 color=[self.colors['overall']] * len(models))
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)
                axes[i].set_ylim(0, 1)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, train_sizes: List[int], 
                            train_scores: List[float],
                            val_scores: List[float],
                            model_name: str = "Model",
                            save_path: Optional[str] = None):
        """
        Plot learning curves.
        
        Args:
            train_sizes: List of training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(train_sizes, train_scores, 'o-', color=self.colors['overall'], 
                label='Training Score', linewidth=2, markersize=8)
        plt.plot(train_sizes, val_scores, 'o-', color=self.colors['negative'], 
                label='Validation Score', linewidth=2, markersize=8)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {model_name}', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: List[float],
                              top_n: int = 20,
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save the plot
        """
        # Sort features by importance
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = feature_importance[:top_n]
        names, scores = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(names)), scores, color=self.colors['overall'])
        
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance Score')
        plt.title(title, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sentiment_timeline(self, df: pd.DataFrame,
                              date_col: str = 'date',
                              sentiment_col: str = 'sentiment',
                              freq: str = 'D',
                              title: str = "Sentiment Over Time",
                              save_path: Optional[str] = None):
        """
        Plot sentiment over time.
        
        Args:
            df: DataFrame with date and sentiment data
            date_col: Name of date column
            sentiment_col: Name of sentiment column
            freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
            title: Plot title
            save_path: Path to save the plot
        """
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create sentiment counts over time
        sentiment_over_time = df.groupby([pd.Grouper(key=date_col, freq=freq), sentiment_col]).size().unstack(fill_value=0)
        
        plt.figure(figsize=(15, 8))
        
        for sentiment in sentiment_over_time.columns:
            plt.plot(sentiment_over_time.index, sentiment_over_time[sentiment], 
                    label=sentiment.title(), color=self.colors.get(sentiment, self.colors['overall']),
                    linewidth=2, marker='o', markersize=4)
        
        plt.title(title, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame,
                                   text_col: str = 'text',
                                   sentiment_col: str = 'sentiment',
                                   save_path: Optional[str] = None):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            df: DataFrame with text and sentiment data
            text_col: Name of text column
            sentiment_col: Name of sentiment column
            save_path: Path to save the HTML file
        """
        # Sentiment distribution
        sentiment_counts = df[sentiment_col].value_counts()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment Pie Chart', 
                          'Text Length Distribution', 'Sentiment by Text Length'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                   name="Sentiment Count", marker_color=[self.colors.get(s, self.colors['overall']) 
                                                       for s in sentiment_counts.index]),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   name="Sentiment Distribution"),
            row=1, col=2
        )
        
        # Text length distribution
        text_lengths = df[text_col].str.len()
        fig.add_trace(
            go.Histogram(x=text_lengths, name="Text Length", nbinsx=30),
            row=2, col=1
        )
        
        # Sentiment by text length
        for sentiment in df[sentiment_col].unique():
            sentiment_data = df[df[sentiment_col] == sentiment]
            fig.add_trace(
                go.Scatter(x=sentiment_data[text_col].str.len(), 
                          y=[sentiment] * len(sentiment_data),
                          mode='markers', name=sentiment.title(),
                          marker=dict(color=self.colors.get(sentiment, self.colors['overall']))),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Sentiment Analysis Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

if __name__ == "__main__":
    # Test the visualizer
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sentiments = np.random.choice(['positive', 'negative', 'neutral'], n_samples, p=[0.4, 0.3, 0.3])
    texts = [f"Sample text {i} with sentiment {sentiment}" for i, sentiment in enumerate(sentiments)]
    
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments,
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    })
    
    # Test visualizer
    visualizer = SentimentVisualizer()
    
    # Plot sentiment distribution
    visualizer.plot_sentiment_distribution(df)
    
    # Create word cloud
    visualizer.create_wordcloud(df['text'].tolist())
    
    # Plot sentiment word clouds
    visualizer.plot_sentiment_wordclouds(df)
    
    # Test model comparison
    results = {
        'Naive Bayes': {'accuracy': 0.85, 'precision_macro': 0.84, 'recall_macro': 0.83, 'f1_macro': 0.84},
        'SVM': {'accuracy': 0.87, 'precision_macro': 0.86, 'recall_macro': 0.85, 'f1_macro': 0.86},
        'Logistic Regression': {'accuracy': 0.86, 'precision_macro': 0.85, 'recall_macro': 0.84, 'f1_macro': 0.85}
    }
    
    visualizer.plot_model_comparison(results) 