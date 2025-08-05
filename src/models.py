"""
Models module for sentiment analysis.
Implements various machine learning models for sentiment classification.
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class SentimentClassifier:
    """Base class for sentiment classification models."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the sentiment classifier."""
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SentimentClassifier':
        """Fit the model to the data."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        try:
            return self.model.predict_proba(X)
        except:
            # Return uniform probabilities if predict_proba not available
            n_classes = len(self.classes_)
            return np.ones((X.shape[0], n_classes)) / n_classes
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SentimentClassifier':
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class NaiveBayesClassifier(SentimentClassifier):
    """Naive Bayes classifier for sentiment analysis."""
    
    def __init__(self, nb_type: str = 'multinomial', **kwargs):
        """
        Initialize Naive Bayes classifier.
        
        Args:
            nb_type: Type of Naive Bayes ('multinomial', 'bernoulli', 'gaussian')
        """
        super().__init__('naive_bayes', **kwargs)
        self.nb_type = nb_type
        
        if nb_type == 'multinomial':
            self.model = MultinomialNB(**kwargs)
        elif nb_type == 'bernoulli':
            self.model = BernoulliNB(**kwargs)
        elif nb_type == 'gaussian':
            self.model = GaussianNB(**kwargs)
        else:
            raise ValueError(f"Unknown Naive Bayes type: {nb_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """Fit the Naive Bayes model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        logger.info("Naive Bayes fitted successfully")
        return self

class SVMClassifier(SentimentClassifier):
    """Support Vector Machine classifier for sentiment analysis."""
    
    def __init__(self, kernel: str = 'linear', C: float = 1.0, 
                 probability: bool = True, **kwargs):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            probability: Whether to enable probability estimates
        """
        super().__init__('svm', **kwargs)
        self.kernel = kernel
        self.C = C
        self.probability = probability
        
        if kernel == 'linear':
            self.model = LinearSVC(C=C, **kwargs)
        else:
            self.model = SVC(kernel=kernel, C=C, probability=probability, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Fit the SVM model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        logger.info("SVM fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For LinearSVC, return decision function values
            decision_values = self.model.decision_function(X)
            if decision_values.ndim == 1:
                # Binary case
                proba = np.zeros((X.shape[0], 2))
                proba[:, 1] = 1 / (1 + np.exp(-decision_values))
                proba[:, 0] = 1 - proba[:, 1]
                return proba
            else:
                # Multi-class case
                return np.exp(decision_values) / np.sum(np.exp(decision_values), axis=1, keepdims=True)

class LogisticRegressionClassifier(SentimentClassifier):
    """Logistic Regression classifier for sentiment analysis."""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, 
                 solver: str = 'lbfgs', **kwargs):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            C: Regularization parameter
            max_iter: Maximum number of iterations
            solver: Optimization algorithm
        """
        super().__init__('logistic_regression', **kwargs)
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver, **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionClassifier':
        """Fit the Logistic Regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        logger.info("Logistic Regression fitted successfully")
        return self

class RandomForestClassifier(SentimentClassifier):
    """Random Forest classifier for sentiment analysis."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42, **kwargs):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
        """
        super().__init__('random_forest', **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = SklearnRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Fit the Random Forest model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        logger.info("Random Forest fitted successfully")
        return self

class GradientBoostingClassifier(SentimentClassifier):
    """Gradient Boosting classifier for sentiment analysis."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: int = 42, **kwargs):
        """
        Initialize Gradient Boosting classifier.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of trees
            random_state: Random seed
        """
        super().__init__('gradient_boosting', **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = SklearnGradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """Fit the Gradient Boosting model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        logger.info("Gradient Boosting fitted successfully")
        return self

class NeuralNetworkClassifier(SentimentClassifier):
    """Neural Network classifier for sentiment analysis."""
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100, 50),
                 activation: str = 'relu', solver: str = 'adam',
                 max_iter: int = 500, random_state: int = 42, **kwargs):
        """
        Initialize Neural Network classifier.
        
        Args:
            hidden_layer_sizes: Sizes of hidden layers
            activation: Activation function
            solver: Optimization algorithm
            max_iter: Maximum number of iterations
            random_state: Random seed
        """
        super().__init__('neural_network', **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkClassifier':
        """Fit the Neural Network model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        logger.info("Neural Network fitted successfully")
        return self

class ModelFactory:
    """Factory class for creating sentiment classification models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> SentimentClassifier:
        """
        Create a sentiment classification model.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for the model
            
        Returns:
            Initialized model instance
        """
        model_mapping = {
            'naive_bayes': NaiveBayesClassifier,
            'svm': SVMClassifier,
            'logistic_regression': LogisticRegressionClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'neural_network': NeuralNetworkClassifier
        }
        
        if model_type not in model_mapping:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_mapping[model_type](**kwargs)
    
    @staticmethod
    def get_hyperparameter_grid(model_type: str) -> Dict[str, list]:
        """
        Get hyperparameter grid for grid search.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of hyperparameter grids
        """
        grids = {
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        return grids.get(model_type, {})

class ModelEnsemble:
    """Ensemble of multiple sentiment classification models."""
    
    def __init__(self, models: Dict[str, SentimentClassifier], 
                 voting_method: str = 'hard'):
        """
        Initialize model ensemble.
        
        Args:
            models: Dictionary of trained models
            voting_method: Voting method ('hard' or 'soft')
        """
        self.models = models
        self.voting_method = voting_method
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelEnsemble':
        """Fit all models in the ensemble."""
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        if self.voting_method == 'hard':
            # Majority voting
            all_predictions = np.array(list(predictions.values()))
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions
            )
        else:
            # Soft voting (average probabilities)
            probas = {}
            for name, model in self.models.items():
                probas[name] = model.predict_proba(X)
            
            avg_proba = np.mean(list(probas.values()), axis=0)
            return np.argmax(avg_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probas = {}
        for name, model in self.models.items():
            probas[name] = model.predict_proba(X)
        
        return np.mean(list(probas.values()), axis=0)
    
    def get_model_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get performance of individual models."""
        performance = {}
        for name, model in self.models.items():
            y_pred = model.predict(X)
            performance[name] = accuracy_score(y, y_pred)
        return performance 