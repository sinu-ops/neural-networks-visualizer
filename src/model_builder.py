"""
Neural Network Model Building Utilities

Utilities for creating and compiling neural network models.
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional


def build_model(
    layer_sizes: List[int] = [2, 64, 32, 16, 2],
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2,
    activation: str = 'relu'
) -> keras.Model:
    """
    Build a sequential neural network.
    
    Args:
        layer_sizes: List of neurons per layer [input, hidden1, hidden2, ..., output]
        learning_rate: Learning rate for the optimizer
        dropout_rate: Dropout rate for regularization
        activation: Activation function for hidden layers
        
    Returns:
        Compiled Keras Sequential model
        
    Raises:
        ValueError: If layer_sizes is invalid
    """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least 2 layers (input and output)")
    
    model = keras.Sequential()
    
    # Input and first hidden layer
    model.add(layers.Dense(layer_sizes[1], activation=activation, input_dim=layer_sizes[0]))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Hidden layers
    for units in layer_sizes[2:-1]:
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    num_classes = layer_sizes[-1]
    output_activation = 'softmax' if num_classes > 1 else 'sigmoid'
    model.add(layers.Dense(layer_sizes[-1], activation=output_activation))
    
    # Compile
    loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def build_cnn_model(
    input_shape: tuple = (28, 28, 1),
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build a Convolutional Neural Network.
    
    Args:
        input_shape: Input shape (height, width, channels)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras CNN model
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model