"""
Main training script for sentiment analysis.
Orchestrates the entire pipeline: data loading, preprocessing, feature extraction, model training, and evaluation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from data_loader import SentimentDataLoader
from preprocessing import SentimentSpecificPreprocessor
from feature_extraction import (
    BagOfWordsExtractor, TFIDFExtractor, Word2VecExtractor, FeatureExtractionPipeline
)
from models import (
    NaiveBayesClassifier, SVMClassifier, LogisticRegressionClassifier,
    RandomForestClassifier, GradientBoostingClassifier, ModelFactory
)
from evaluation import SentimentAnalysisEvaluator
from visualization import SentimentVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    """Complete sentiment analysis pipeline."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.data_loader = SentimentDataLoader()
        self.preprocessor = SentimentSpecificPreprocessor()
        self.evaluator = SentimentAnalysisEvaluator()
        self.visualizer = SentimentVisualizer()
        
        # Initialize components
        self.feature_extractors = {}
        self.models = {}
        self.results = {}
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'dataset': 'sentiment140',  # or 'imdb'
            'sample_size': 10000,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'feature_extractors': {
                'bow': {'max_features': 5000},
                'tfidf': {'max_features': 5000},
                'w2v': {'vector_size': 100}
            },
            'models': {
                'naive_bayes': {'nb_type': 'multinomial'},
                'svm': {'kernel': 'linear', 'C': 1.0},
                'logistic_regression': {'C': 1.0, 'max_iter': 1000},
                'random_forest': {'n_estimators': 100},
                'gradient_boosting': {'n_estimators': 100}
            }
        }
    
    def load_data(self) -> tuple:
        """Load and split the dataset."""
        logger.info("Loading dataset...")
        
        # Load dataset
        df = self.data_loader.load_dataset(
            dataset_name=self.config['dataset'],
            sample_size=self.config['sample_size']
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            df, 
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(self, X_train, X_val, X_test):
        """Preprocess the text data."""
        logger.info("Preprocessing data...")
        
        # Preprocess training data
        X_train_processed = self.preprocessor.preprocess_series(X_train)
        X_val_processed = self.preprocessor.preprocess_series(X_val)
        X_test_processed = self.preprocessor.preprocess_series(X_test)
        
        logger.info("Data preprocessing completed")
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def extract_features(self, X_train, X_val, X_test):
        """Extract features from preprocessed text."""
        logger.info("Extracting features...")
        
        # Initialize feature extractors
        self.feature_extractors = {
            'bow': BagOfWordsExtractor(**self.config['feature_extractors']['bow']),
            'tfidf': TFIDFExtractor(**self.config['feature_extractors']['tfidf']),
            'w2v': Word2VecExtractor(**self.config['feature_extractors']['w2v'])
        }
        
        # Extract features
        features = {}
        for name, extractor in self.feature_extractors.items():
            logger.info(f"Extracting {name} features...")
            
            # Fit and transform training data
            X_train_features = extractor.fit_transform(X_train)
            X_val_features = extractor.transform(X_val)
            X_test_features = extractor.transform(X_test)
            
            features[name] = {
                'train': X_train_features,
                'val': X_val_features,
                'test': X_test_features
            }
            
            # Save feature extractor
            extractor.save_model(f'models/{name}_extractor.pkl')
        
        logger.info("Feature extraction completed")
        return features
    
    def train_models(self, features, y_train, y_val):
        """Train all models on all feature sets."""
        logger.info("Training models...")
        
        # Initialize models
        self.models = {}
        for name, params in self.config['models'].items():
            self.models[name] = ModelFactory.create_model(name, **params)
        
        # Train models on each feature set
        for feature_name, feature_data in features.items():
            logger.info(f"Training models on {feature_name} features...")
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name} on {feature_name}...")
                
                # Train model
                model.fit(feature_data['train'], y_train)
                
                # Save model
                model.save_model(f'models/{model_name}_{feature_name}.pkl')
                
                # Evaluate on validation set
                y_val_pred = model.predict(feature_data['val'])
                y_val_proba = None
                try:
                    y_val_proba = model.predict_proba(feature_data['val'])
                except:
                    pass
                
                # Store results
                model_key = f"{model_name}_{feature_name}"
                self.results[model_key] = {
                    'model': model,
                    'feature_name': feature_name,
                    'val_predictions': y_val_pred,
                    'val_probabilities': y_val_proba
                }
        
        logger.info("Model training completed")
    
    def evaluate_models(self, y_val, y_test, features):
        """Evaluate all models."""
        logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        # Evaluate on validation set
        for model_key, result in self.results.items():
            logger.info(f"Evaluating {model_key}...")
            
            # Validation evaluation
            val_metrics = self.evaluator.evaluate_sentiment_model(
                y_val, 
                result['val_predictions'], 
                result['val_probabilities'],
                model_key
            )
            
            # Test evaluation
            feature_name = result['feature_name']
            X_test_features = features[feature_name]['test']
            y_test_pred = result['model'].predict(X_test_features)
            y_test_proba = None
            try:
                y_test_proba = result['model'].predict_proba(X_test_features)
            except:
                pass
            
            test_metrics = self.evaluator.evaluate_sentiment_model(
                y_test, 
                y_test_pred, 
                y_test_proba,
                f"{model_key}_test"
            )
            
            evaluation_results[model_key] = {
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'test_predictions': y_test_pred,
                'test_probabilities': y_test_proba
            }
        
        # Save evaluation results
        self.evaluator.save_results('results/evaluation_results.json')
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def create_visualizations(self, df, evaluation_results):
        """Create visualizations."""
        logger.info("Creating visualizations...")
        
        # Sentiment distribution
        self.visualizer.plot_sentiment_distribution(
            df, 
            save_path='plots/sentiment_distribution.png'
        )
        
        # Word clouds
        self.visualizer.plot_sentiment_wordclouds(
            df, 
            save_path='plots/sentiment_wordclouds.png'
        )
        
        # Model comparison
        model_comparison = {}
        for model_key, result in evaluation_results.items():
            model_comparison[model_key] = result['test_metrics']
        
        self.visualizer.plot_model_comparison(
            model_comparison,
            save_path='plots/model_comparison.png'
        )
        
        # Confusion matrices
        for model_key, result in evaluation_results.items():
            self.evaluator.plot_confusion_matrix(
                y_test, 
                result['test_predictions'],
                model_key,
                save_path=f'plots/confusion_matrix_{model_key}.png'
            )
        
        logger.info("Visualizations created")
    
    def generate_report(self, evaluation_results, df):
        """Generate comprehensive report."""
        logger.info("Generating report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'dataset_info': {
                'total_samples': len(df),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
                'preprocessing_summary': self.preprocessor.get_preprocessing_summary()
            },
            'model_performance': {}
        }
        
        # Add model performance
        for model_key, result in evaluation_results.items():
            report['model_performance'][model_key] = {
                'validation_metrics': result['val_metrics'],
                'test_metrics': result['test_metrics']
            }
        
        # Find best model
        best_model, best_score = self.evaluator.get_best_model('f1_macro')
        report['best_model'] = {
            'name': best_model,
            'score': best_score
        }
        
        # Save report
        with open('results/training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Dataset: {self.config['dataset']}")
        print(f"Total samples: {len(df)}")
        print(f"Best model: {best_model} (F1: {best_score:.4f})")
        print("="*50)
        
        logger.info("Report generated")
        return report
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        logger.info("Starting sentiment analysis pipeline...")
        
        try:
            # Load data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
            
            # Preprocess data
            X_train_processed, X_val_processed, X_test_processed = self.preprocess_data(
                X_train, X_val, X_test
            )
            
            # Extract features
            features = self.extract_features(X_train_processed, X_val_processed, X_test_processed)
            
            # Train models
            self.train_models(features, y_train, y_val)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(y_val, y_test, features)
            
            # Create visualizations
            df = pd.DataFrame({
                'text': pd.concat([X_train, X_val, X_test]),
                'sentiment': pd.concat([y_train, y_val, y_test])
            })
            self.create_visualizations(df, evaluation_results)
            
            # Generate report
            report = self.generate_report(evaluation_results, df)
            
            logger.info("Pipeline completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the training pipeline."""
    # Configuration
    config = {
        'dataset': 'sentiment140',
        'sample_size': 5000,  # Reduced for faster training
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42,
        'feature_extractors': {
            'bow': {'max_features': 2000},
            'tfidf': {'max_features': 2000},
            'w2v': {'vector_size': 50}
        },
        'models': {
            'naive_bayes': {'nb_type': 'multinomial'},
            'svm': {'kernel': 'linear', 'C': 1.0},
            'logistic_regression': {'C': 1.0, 'max_iter': 500},
            'random_forest': {'n_estimators': 50},
            'gradient_boosting': {'n_estimators': 50}
        }
    }
    
    # Create and run pipeline
    pipeline = SentimentAnalysisPipeline(config)
    report = pipeline.run_pipeline()
    
    print("\nTraining completed! Check the following directories:")
    print("- models/: Trained models")
    print("- results/: Evaluation results and reports")
    print("- plots/: Visualizations")

if __name__ == "__main__":
    main() 