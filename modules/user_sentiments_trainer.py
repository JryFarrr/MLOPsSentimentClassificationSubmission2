"""Module for building and training the user sentiments model."""

from tensorflow import keras

def build_model():
    """Builds and returns a compiled sentiment analysis model."""
    layers = keras.layers
    model = keras.Sequential([
        layers.Embedding(input_dim=5000, output_dim=16),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
