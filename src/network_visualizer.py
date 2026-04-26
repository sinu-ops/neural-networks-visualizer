"""
Neural Network Architecture Visualization Module

Visualizes network structure, connections, weights, and activations.

Classes:
    NeuralNetworkVisualizer: Main visualization class
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Optional, Tuple


class NeuralNetworkVisualizer:
    """
    Visualizes neural network architecture with connections, weights, and activations.
    
    This class provides methods to visualize:
    - Network structure with neurons and connections
    - Weight magnitudes through line width and color
    - Neuron activations through color coding
    - 2D decision boundaries
    
    Example:
        >>> visualizer = NeuralNetworkVisualizer([2, 64, 32, 16, 2])
        >>> fig = visualizer.plot_network_architecture()
        >>> plt.show()
    """
    
    def __init__(self, layer_sizes: List[int], figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize the visualizer.
        
        Args:
            layer_sizes: List of neuron counts per layer
                        [input, hidden1, hidden2, ..., output]
            figsize: Figure size as (width, height) tuple
            
        Raises:
            ValueError: If layer_sizes is empty or invalid
        """
        if not layer_sizes or len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 layers (input and output)")
        
        self.layer_sizes = layer_sizes
        self.figsize = figsize
        self.num_layers = len(layer_sizes)
    
    def plot_network_architecture(
        self,
        weights: Optional[List[np.ndarray]] = None,
        activations: Optional[List[np.ndarray]] = None,
        title: str = "Neural Network Architecture"
    ) -> plt.Figure:
        """
        Plot the network structure with optional weight and activation visualization.
        
        Args:
            weights: List of weight matrices for each layer
            activations: List of activation arrays for each neuron
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate neuron positions
        layer_positions = np.linspace(0, 1, self.num_layers)
        neuron_y_positions = []
        neuron_x_positions = []
        
        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            y_positions = np.linspace(0, 1, num_neurons)
            neuron_y_positions.append(y_positions)
            neuron_x_positions.append([layer_positions[layer_idx]] * num_neurons)
        
        # Draw connections between layers
        for layer_idx in range(self.num_layers - 1):
            for neuron_idx in range(self.layer_sizes[layer_idx]):
                for next_neuron_idx in range(self.layer_sizes[layer_idx + 1]):
                    
                    x1 = neuron_x_positions[layer_idx][neuron_idx]
                    y1 = neuron_y_positions[layer_idx][neuron_idx]
                    x2 = neuron_x_positions[layer_idx + 1][next_neuron_idx]
                    y2 = neuron_y_positions[layer_idx + 1][next_neuron_idx]
                    
                    # Visualize weight magnitude
                    if weights is not None:
                        weight_magnitude = np.abs(weights[layer_idx][neuron_idx, next_neuron_idx])
                        max_weight = np.max(np.abs(weights[layer_idx]))
                        line_width = 0.5 + 2 * (weight_magnitude / max_weight)
                        color = 'red' if weights[layer_idx][neuron_idx, next_neuron_idx] < 0 else 'blue'
                        alpha = min(0.8, weight_magnitude / max_weight)
                    else:
                        line_width = 1
                        color = 'gray'
                        alpha = 0.3
                    
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                           linewidth=line_width, zorder=1)
        
        # Draw neurons
        for layer_idx in range(self.num_layers):
            for neuron_idx in range(self.layer_sizes[layer_idx]):
                x = neuron_x_positions[layer_idx][neuron_idx]
                y = neuron_y_positions[layer_idx][neuron_idx]
                
                # Color neurons based on activation
                if activations is not None:
                    activation = np.clip(activations[layer_idx][neuron_idx], 0, 1)
                    color = plt.cm.RdYlGn(activation)
                    size = 300 + 200 * activation
                else:
                    color = 'lightblue'
                    size = 300
                
                circle = Circle((x, y), 0.03, color=color, ec='black',
                              linewidth=2, zorder=2)
                ax.add_patch(circle)
        
        # Add layer labels
        layer_labels = ['Input'] + [f'Hidden {i}' for i in range(1, self.num_layers-1)] + ['Output']
        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            label = layer_labels[layer_idx]
            ax.text(layer_positions[layer_idx], -0.15,
                   f"{label}\n({num_neurons} neurons)",
                   ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_decision_boundary(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str = "Decision Boundary"
    ) -> Optional[plt.Figure]:
        """
        Plot 2D decision boundary for classification models.
        
        Works only with 2D input data.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix (must be 2D)
            y: Class labels
            title: Plot title
            
        Returns:
            matplotlib Figure object or None if data is not 2D
        """
        if X.shape[1] != 2:
            print("❌ Decision boundary visualization requires 2D data")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh grid for background
        h = 0.02  # Step size in mesh
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        if len(Z.shape) > 1 and Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, levels=15, cmap=plt.cm.RdBu, alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        # Plot training data
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu,
                            edgecolors='k', linewidth=1.5, s=100, alpha=0.8)
        
        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Class')
        
        plt.tight_layout()
        return fig