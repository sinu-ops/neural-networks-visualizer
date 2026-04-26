"""
Data Utilities and Dataset Generation

Utilities for generating and preparing datasets.
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


def generate_dataset(
    dataset_type: str = 'moons',
    n_samples: int = 300,
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D classification datasets.
    
    Args:
        dataset_type: Type of dataset ('moons', 'circles', or 'digits')
        n_samples: Number of samples to generate
        noise: Noise level for synthetic data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_all, y_all)
        
    Raises:
        ValueError: If invalid dataset_type is provided
    """
    print(f"📊 Generating '{dataset_type}' dataset with {n_samples} samples...")
    
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset_type == 'digits':
        digits = load_digits()
        X, y = digits.data, digits.target
        # Limit to first 300 samples for visualization
        X, y = X[:300], y[:300]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'moons', 'circles', or 'digits'")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X, y


def create_custom_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split custom dataset into train and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test