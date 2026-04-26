"""
Layer Analysis and Visualization Module

Analyzes and visualizes layer weights, activations, and gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


class LayerAnalyzer:
    """
    Analyzes neural network layers for weights, activations, and gradients.
    
    Provides visualization for:
    - Weight distributions per layer
    - Neuron activation heatmaps
    - Gradient flow analysis
    """
    
    @staticmethod
    def plot_activation_heatmap(
        activations_dict: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Visualize neuron activations as heatmaps.
        
        Args:
            activations_dict: Dictionary mapping layer names to activation matrices
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        num_layers = len(activations_dict)
        fig, axes = plt.subplots(1, num_layers, figsize=figsize)
        
        if num_layers == 1:
            axes = [axes]
        
        for idx, (layer_name, activations) in enumerate(activations_dict.items()):
            sns.heatmap(activations, cmap='viridis', ax=axes[idx], cbar=True)
            axes[idx].set_title(f'{layer_name}\nActivations', fontweight='bold', fontsize=11)
            axes[idx].set_xlabel('Neuron Index')
            axes[idx].set_ylabel('Sample Index')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_weight_distribution(
        weights_dict: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Analyze and visualize weight distributions across layers.
        
        Args:
            weights_dict: Dictionary mapping layer names to weight matrices
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        num_layers = len(weights_dict)
        fig, axes = plt.subplots(1, num_layers, figsize=figsize)
        
        if num_layers == 1:
            axes = [axes]
        
        for idx, (layer_name, weights) in enumerate(weights_dict.items()):
            axes[idx].hist(weights.flatten(), bins=50, color='skyblue',
                          edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{layer_name}\nWeight Distribution',
                               fontweight='bold', fontsize=11)
            axes[idx].set_xlabel('Weight Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(weights.mean(), color='red', linestyle='--',
                             label=f'Mean: {weights.mean():.3f}', linewidth=2)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_gradient_flow(
        gradients_dict: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Visualize gradient magnitude flow through layers.
        
        Helps diagnose vanishing/exploding gradient problems.
        
        Args:
            gradients_dict: Dictionary mapping layer names to gradient matrices
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        layer_names = list(gradients_dict.keys())
        mean_grads = [np.mean(np.abs(g)) for g in gradients_dict.values()]
        max_grads = [np.max(np.abs(g)) for g in gradients_dict.values()]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        ax.bar(x - width/2, mean_grads, width, label='Mean Gradient',
              alpha=0.8, color='skyblue')
        ax.bar(x + width/2, max_grads, width, label='Max Gradient',
              alpha=0.8, color='coral')
        
        ax.set_xlabel('Layer', fontweight='bold', fontsize=12)
        ax.set_ylabel('Gradient Magnitude', fontweight='bold', fontsize=12)
        ax.set_title('Gradient Flow Through Network',
                    fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig