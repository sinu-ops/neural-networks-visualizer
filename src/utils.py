"""
Utility Functions and Helpers

General utility functions for the project.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def create_output_directories() -> None:
    """Create necessary output directories if they don't exist."""
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to outputs folder.
    
    Args:
        fig: matplotlib Figure object
        filename: Name of the file (without path)
        dpi: Resolution in dots per inch
    """
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✅ Figure saved to {filepath}")


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range.
    
    Args:
        arr: Input array
        
    Returns:
        Normalized array
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def print_model_info(model) -> None:
    """
    Print comprehensive model information.
    
    Args:
        model: Keras model
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    model.summary()
    
    print(f"\nTotal Layers: {len(model.layers)}")
    print(f"Total Parameters: {model.count_params():,}")
    
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable_params = sum([np.prod(w.shape) for w in model.non_trainable_weights])
    
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    