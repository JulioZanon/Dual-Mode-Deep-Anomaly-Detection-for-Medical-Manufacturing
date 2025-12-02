"""MOCCA (Multi-Objective Contextual Clustering Analysis) aligned model for anomaly detection.

This implementation follows the PyTorch-aligned approach for MOCCA anomaly detection.
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path

class FeatureExtractionMethod(Enum):
    """Feature extraction methods for MOCCA."""
    PYTORCH_ALIGNED = "pytorch_aligned"
    STANDARD = "standard"

class MOCCA_OneClass:
    """One-class MOCCA model for anomaly detection.
    
    This model implements the one-class learning approach for MOCCA,
    following the PyTorch-aligned implementation.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], code_length: int, 
                 use_selectors: bool = True, idx_list: List[int] = None,
                 feature_extraction_method: FeatureExtractionMethod = FeatureExtractionMethod.PYTORCH_ALIGNED):
        """Initialize MOCCA OneClass model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            code_length: Length of the latent code
            use_selectors: Whether to use feature selectors
            idx_list: List of layer indices to use for feature extraction
            feature_extraction_method: Method for feature extraction
        """
        self.input_shape = input_shape
        self.code_length = code_length
        self.use_selectors = use_selectors
        self.idx_list = idx_list or [4, 5, 6, 7]
        self.feature_extraction_method = feature_extraction_method
        
        # Initialize model components
        self.encoder = None
        self.selectors = {}
        self.centers = {}
        self.radiuses = {}
        self.threshold = None
        
        # Build the model
        self._build_model()
        
        logger.info(f"Initialized MOCCA OneClass model with {len(self.idx_list)} feature layers")
        logger.info(f"Feature extraction method: {self.feature_extraction_method.value}")
    
    def _build_model(self):
        """Build the MOCCA model architecture."""
        # Input layer
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Encoder (simplified for this implementation)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D(2)(x)  # 112x112
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)  # 56x56
        x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)  # 28x28
        x = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)  # 14x14
        x = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)  # 7x7
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Flatten to 1D
        encoded = tf.keras.layers.Dense(self.code_length, name='fc2')(x)
        
        # Create encoder model
        self.encoder = tf.keras.Model(inputs=inputs, outputs=encoded, name='encoder')
        
        # Decoder for reconstruction
        decoder_input = tf.keras.Input(shape=(self.code_length,))
        x = tf.keras.layers.Dense(7 * 7 * 128, activation='relu')(decoder_input)
        x = tf.keras.layers.Reshape((7, 7, 128))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)  # 14x14
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)  # 28x28
        x = tf.keras.layers.Conv2DTranspose(16, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)  # 56x56
        x = tf.keras.layers.Conv2DTranspose(8, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)  # 112x112
        x = tf.keras.layers.Conv2DTranspose(4, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)  # 224x224
        decoded = tf.keras.layers.Conv2DTranspose(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        
        # Create decoder model
        self.decoder = tf.keras.Model(inputs=decoder_input, outputs=decoded, name='decoder')
        
        # Full autoencoder model
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = tf.keras.Model(inputs=inputs, outputs=decoded, name='autoencoder')
        
        # Initialize selectors if enabled
        if self.use_selectors:
            for idx in self.idx_list:
                selector_name = f'selector_{idx}'
                self.selectors[selector_name] = tf.keras.layers.Dense(
                    self.code_length, 
                    activation='relu',
                    name=selector_name
                )
        
        logger.info("Built MOCCA model architecture")
    
    def set_centers(self, centers: Dict[str, np.ndarray]):
        """Set the centers for anomaly detection.
        
        Args:
            centers: Dictionary mapping layer indices to center vectors
        """
        self.centers = centers
        logger.info(f"Set {len(centers)} centers for anomaly detection")
    
    def set_radiuses(self, radiuses: Dict[str, np.ndarray]):
        """Set the radiuses for anomaly detection.
        
        Args:
            radiuses: Dictionary mapping layer indices to radius values
        """
        self.radiuses = radiuses
        logger.info(f"Set {len(radiuses)} radiuses for anomaly detection")
    
    def predict(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict anomaly scores for inputs.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Tuple of (reconstructions, anomaly_scores)
        """
        # Extract features through encoder
        features = self.encoder(inputs, training=False)
        
        # Compute anomaly scores using centers and radiuses
        if not self.centers or not self.radiuses:
            logger.warning("No centers or radiuses set, returning zero scores")
            return inputs, tf.zeros(tf.shape(inputs)[0])
        
        # For simplicity, use the first center/radius pair
        # In a full implementation, you'd use all layers
        first_key = list(self.centers.keys())[0]
        center = self.centers[first_key]
        radius = self.radiuses[first_key]
        
        # Compute distances to center
        distances = tf.norm(features - center, axis=1)
        
        # Convert distances to anomaly scores (higher distance = more anomalous)
        anomaly_scores = distances
        
        return inputs, anomaly_scores
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Model output (reconstructions for autoencoder training)
        """
        return self.autoencoder(inputs, training=training)
    
    def __call__(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Make the model callable."""
        return self.call(inputs, training=training)
    
    @property
    def trainable_variables(self):
        """Get trainable variables for the autoencoder."""
        return self.autoencoder.trainable_variables
    
    def save_weights(self, filepath, overwrite=True):
        """Save model weights."""
        return self.autoencoder.save_weights(filepath, overwrite)
    
    def load_weights(self, filepath, by_name=False, skip_mismatch=False):
        """Load model weights."""
        return self.autoencoder.load_weights(filepath, by_name, skip_mismatch)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return {
            'input_shape': self.input_shape,
            'code_length': self.code_length,
            'use_selectors': self.use_selectors,
            'idx_list': self.idx_list,
            'feature_extraction_method': self.feature_extraction_method.value
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MOCCA_OneClass':
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            MOCCA OneClass model instance
        """
        # Convert enum string back to enum
        if 'feature_extraction_method' in config:
            config['feature_extraction_method'] = FeatureExtractionMethod(config['feature_extraction_method'])
        
        return cls(**config) 
