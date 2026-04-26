"""
Advanced Evaluation and Visualization Features

Provides advanced analysis tools for model evaluation and comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, Tuple, Optional


class AdvancedVisualizations:
    """
    Advanced visualization and analysis tools for neural networks.
    
    Provides:
    - Confusion matrices
    - ROC curves
    - Feature importance
    - Model comparison
    """
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix"
    ) -> plt.Figure:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or probabilities
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        # Convert predictions to class labels if needed
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
        ax.set_ylabel('True', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve"
    ) -> plt.Figure:
        """
        Plot ROC curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        # Handle multi-class case
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(
        model,
        feature_names: Optional[list] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Compute and plot feature importance from first layer weights.
        
        Args:
            model: Trained Keras model
            feature_names: Names of features
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        first_layer_weights = model.layers[0].get_weights()[0]
        feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
        feature_importance = feature_importance / np.sum(feature_importance)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
        bars = ax.barh(feature_names, feature_importance, color=colors)
        ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
        ax.set_title('Feature Importance (Based on First Layer Weights)',
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, feature_importance):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   va='center', ha='left', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def compare_models(
        models_dict: Dict[str, object],
        X_test: np.ndarray,
        y_test: np.ndarray,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Compare multiple models side by side.
        
        Args:
            models_dict: Dictionary mapping model names to trained models
            X_test: Test features
            y_test: Test labels
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        model_names = list(models_dict.keys())
        losses = []
        accuracies = []
        
        for model in models_dict.values():
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            losses.append(loss)
            accuracies.append(acc)
        
        # Plot losses
        axes[0].bar(model_names, losses, color='coral', alpha=0.7,
                   edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Loss', fontweight='bold', fontsize=12)
        axes[0].set_title('Model Comparison: Test Loss', fontweight='bold', fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot accuracies
        axes[1].bar(model_names, accuracies, color='lightgreen', alpha=0.7,
                   edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        axes[1].set_title('Model Comparison: Test Accuracy', fontweight='bold', fontsize=13)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig