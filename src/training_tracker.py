"""
Training Visualization and Tracking Module

Tracks training metrics and provides visualization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class TrainingVisualizer:
    """
    Tracks and visualizes neural network training metrics.
    
    Provides methods to:
    - Track loss and accuracy over epochs
    - Visualize training vs validation performance
    - Compare different training runs
    """
    
    def __init__(self):
        """Initialize training history dictionary."""
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch': []
        }
    
    def update(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None
    ) -> None:
        """
        Update training history with new epoch data.
        
        Args:
            epoch: Epoch number
            loss: Training loss
            accuracy: Training accuracy
            val_loss: Validation loss (optional)
            val_accuracy: Validation accuracy (optional)
        """
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_accuracy is not None:
            self.history['val_accuracy'].append(val_accuracy)
    
    def plot_training_curves(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot training and validation curves side by side.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.history['epoch'], self.history['loss'],
                    'b-o', label='Train Loss', linewidth=2, markersize=4)
        if self.history['val_loss']:
            axes[0].plot(self.history['epoch'], self.history['val_loss'],
                        'r-s', label='Val Loss', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history['epoch'], self.history['accuracy'],
                    'g-o', label='Train Accuracy', linewidth=2, markersize=4)
        if self.history['val_accuracy']:
            axes[1].plot(self.history['epoch'], self.history['val_accuracy'],
                        'orange', marker='s', label='Val Accuracy', linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Training Accuracy Over Time', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_rate_effect(
        self,
        histories_dict: Dict[float, 'TrainingVisualizer'],
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Compare training runs with different learning rates.
        
        Args:
            histories_dict: Dictionary mapping learning rates to TrainingVisualizer objects
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(histories_dict)))
        
        for (lr, history), color in zip(histories_dict.items(), colors):
            axes[0].plot(history.history['epoch'], history.history['loss'],
                        label=f'LR={lr}', color=color, linewidth=2)
            axes[1].plot(history.history['epoch'], history.history['accuracy'],
                        label=f'LR={lr}', color=color, linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Learning Rate Impact on Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontweight='bold')
        axes[1].set_title('Learning Rate Impact on Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig