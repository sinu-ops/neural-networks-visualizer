"""
Neural Networks Visualizer Package

A comprehensive tool for visualizing neural network architectures,
training dynamics, and layer activations.

Author: Your Name
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Interactive visualization tool for neural networks"
__license__ = "MIT"

# Import main classes for easy access
from .network_visualizer import NeuralNetworkVisualizer
from .training_tracker import TrainingVisualizer
from .layer_analyzer import LayerAnalyzer
from .advanced_features import AdvancedVisualizations
from .data_utils import generate_dataset
from .model_builder import build_model

# Define what gets imported with: from src import *
__all__ = [
    'NeuralNetworkVisualizer',
    'TrainingVisualizer',
    'LayerAnalyzer',
    'AdvancedVisualizations',
    'generate_dataset',
    'build_model',
]

# Print welcome message when package is imported
print(f"✅ Neural Networks Visualizer v{__version__} loaded successfully!")