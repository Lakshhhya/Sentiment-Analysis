"""
Model evaluation module for sentiment analysis.
Implements accuracy, precision, recall, F1-score, and confusion matrix evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluator for sentiment analysis."""
    
    def __init__(self, classes: Optional[List[str]] = None):
        """
        Initialize model evaluator.
        
        Args:
            classes: List of class names (e.g., ['negative', 'neutral', 'positive'])
        """
        self.classes = classes
        self.results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None,
                      model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        if self.classes:
            for i, class_name in enumerate(self.classes):
                if len(np.unique(y_true)) > 1:  # At least 2 classes
                    metrics[f'precision_{class_name}'] = precision_score(
                        y_true, y_pred, pos_label=class_name, zero_division=0
                    )
                    metrics[f'recall_{class_name}'] = recall_score(
                        y_true, y_pred, pos_label=class_name, zero_division=0
                    )
                    metrics[f'f1_{class_name}'] = f1_score(
                        y_true, y_pred, pos_label=class_name, zero_division=0
                    )
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                logger.warning("Could not calculate ROC AUC")
        
        # Multi-class AUC
        elif len(np.unique(y_true)) > 2 and y_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
            except:
                logger.warning("Could not calculate multi-class ROC AUC")
        
        # Store results
        self.results[model_name] = metrics
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, model_name: str = "model") -> Dict[str, List[float]]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Trained model with fit/predict methods
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            model_name: Name of the model
            
        Returns:
            Dictionary of cross-validation results
        """
        cv_results = {}
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_results['accuracy'] = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
        cv_results['precision_macro'] = cross_val_score(model, X, y, cv=cv_strategy, scoring='precision_macro')
        cv_results['recall_macro'] = cross_val_score(model, X, y, cv=cv_strategy, scoring='recall_macro')
        cv_results['f1_macro'] = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1_macro')
        
        # Calculate mean and std
        cv_summary = {}
        for metric, scores in cv_results.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
        
        self.results[f"{model_name}_cv"] = cv_summary
        
        logger.info(f"Cross-validation completed for {model_name}")
        return cv_results
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                           normalize: bool = True) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model", normalize: bool = True,
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
        """
        cm = self.get_confusion_matrix(y_true, y_pred, normalize=normalize)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=True,
                   xticklabels=self.classes if self.classes else None,
                   yticklabels=self.classes if self.classes else None)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if len(np.unique(y_true)) != 2:
            logger.warning("ROC curve only available for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                  model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if len(np.unique(y_true)) != 2:
            logger.warning("Precision-Recall curve only available for binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary of {model_name: metrics} pairs
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "Model") -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Classification report as string
        """
        report = f"Classification Report for {model_name}\n"
        report += "=" * 50 + "\n\n"
        report += classification_report(y_true, y_pred, target_names=self.classes)
        return report
    
    def get_best_model(self, metric: str = 'f1_macro') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (best_model_name, best_score)
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in self.results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        return best_model, best_score
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load evaluation results from file."""
        import json
        
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")

class SentimentAnalysisEvaluator(ModelEvaluator):
    """Specialized evaluator for sentiment analysis."""
    
    def __init__(self):
        """Initialize sentiment analysis evaluator."""
        super().__init__(classes=['negative', 'neutral', 'positive'])
    
    def evaluate_sentiment_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               model_name: str = "sentiment_model") -> Dict[str, float]:
        """
        Evaluate sentiment analysis model with sentiment-specific metrics.
        
        Args:
            y_true: True sentiment labels
            y_pred: Predicted sentiment labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = self.evaluate_model(y_true, y_pred, y_proba, model_name)
        
        # Sentiment-specific metrics
        if len(np.unique(y_true)) >= 2:
            # Binary sentiment (positive vs negative)
            if len(np.unique(y_true)) == 2:
                positive_class = 'positive' if 'positive' in y_true else y_true[0]
                metrics['positive_precision'] = precision_score(
                    y_true, y_pred, pos_label=positive_class, zero_division=0
                )
                metrics['positive_recall'] = recall_score(
                    y_true, y_pred, pos_label=positive_class, zero_division=0
                )
                metrics['positive_f1'] = f1_score(
                    y_true, y_pred, pos_label=positive_class, zero_division=0
                )
            
            # Multi-class sentiment
            elif len(np.unique(y_true)) == 3:
                for sentiment in ['negative', 'neutral', 'positive']:
                    if sentiment in y_true:
                        metrics[f'{sentiment}_precision'] = precision_score(
                            y_true, y_pred, pos_label=sentiment, zero_division=0
                        )
                        metrics[f'{sentiment}_recall'] = recall_score(
                            y_true, y_pred, pos_label=sentiment, zero_division=0
                        )
                        metrics[f'{sentiment}_f1'] = f1_score(
                            y_true, y_pred, pos_label=sentiment, zero_division=0
                        )
        
        return metrics
    
    def plot_sentiment_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot sentiment distribution comparison.
        
        Args:
            y_true: True sentiment labels
            y_pred: Predicted sentiment labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        true_counts = pd.Series(y_true).value_counts()
        true_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('True Sentiment Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts()
        pred_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Predicted Sentiment Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Sentiment Distribution - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Test the evaluator
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                              n_informative=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    evaluator = SentimentAnalysisEvaluator()
    metrics = evaluator.evaluate_sentiment_model(y_test, y_pred, y_proba, "Random Forest")
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(y_test, y_pred, "Random Forest")
    
    # Generate report
    report = evaluator.generate_report(y_test, y_pred, "Random Forest")
    print("\n" + report) 