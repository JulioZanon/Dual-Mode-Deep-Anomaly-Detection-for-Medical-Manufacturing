"""CPC-AE (Contrastive Predictive Coding Autoencoder) model for anomaly detection.

This implementation provides a TensorFlow 2.13+ compatible version of CPC-AE
with patch-based processing and anomaly detection capabilities.
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path


class CPC_AE:
    """CPC-AE (Contrastive Predictive Coding Autoencoder) model for anomaly detection.
    
    This model implements patch-based processing with positional encoding
    for improved anomaly detection performance.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], patch_size: Tuple[int, int] = (64, 64),
                 stride: int = 32, latent_dim: int = 256, use_positional_encoding: bool = True):
        """Initialize CPC-AE model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            patch_size: Size of patches to extract (height, width)
            stride: Stride for patch extraction
            latent_dim: Dimension of the latent space
            use_positional_encoding: Whether to use positional encoding for patches
        """
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.use_positional_encoding = use_positional_encoding
        
        # Initialize model components
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.centers = {}
        self.radiuses = {}
        self.threshold = None
        
        # Build the model
        self._build_model()
        
        logger.info(f"Initialized CPC-AE model with patch size {patch_size} and latent dim {latent_dim}")
    
    def _build_model(self):
        """Build the CPC-AE model architecture."""
        # Input layers
        patch_input = tf.keras.Input(shape=self.patch_size + (self.input_shape[-1],), name='patch_input')
        index_input = tf.keras.Input(shape=(2,), name='index_input')
        
        # Encoder
        x = tf.keras.layers.Conv2D(32, (5, 5), strides=1, activation='relu', padding='same')(patch_input)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        
        # Concatenate with positional encoding
        if self.use_positional_encoding:
            # Normalize patch indices
            normalized_indices = index_input / tf.cast(self.input_shape[:2], tf.float32)
            x = tf.keras.layers.Concatenate()([x, normalized_indices])
        
        # Final encoding layer
        encoded = tf.keras.layers.Dense(self.latent_dim, activation='relu', name='encoded')(x)
        
        # Create encoder model
        self.encoder = tf.keras.Model(
            inputs=[patch_input, index_input], 
            outputs=encoded, 
            name='cpcae_encoder'
        )
        
        # Decoder
        decoder_input = tf.keras.Input(shape=(self.latent_dim,), name='decoder_input')
        decoder_index_input = tf.keras.Input(shape=(2,), name='decoder_index_input')
        
        # Concatenate with positional encoding for decoder
        if self.use_positional_encoding:
            normalized_indices = decoder_index_input / tf.cast(self.input_shape[:2], tf.float32)
            x = tf.keras.layers.Concatenate()([decoder_input, normalized_indices])
        else:
            x = decoder_input
        
        # Decoder layers
        x = tf.keras.layers.Dense(4 * 4 * 512, activation='relu')(x)
        x = tf.keras.layers.Reshape((4, 4, 512))(x)
        x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=1, activation='relu', padding='same')(x)
        decoded = tf.keras.layers.Conv2DTranspose(
            self.input_shape[-1], (5, 5), strides=1, activation='sigmoid', padding='same'
        )(x)
        
        # Create decoder model
        self.decoder = tf.keras.Model(
            inputs=[decoder_input, decoder_index_input], 
            outputs=decoded, 
            name='cpcae_decoder'
        )
        
        # Full autoencoder model
        encoded = self.encoder([patch_input, index_input])
        decoded = self.decoder([encoded, index_input])
        self.autoencoder = tf.keras.Model(
            inputs=[patch_input, index_input], 
            outputs=decoded, 
            name='cpcae_autoencoder'
        )
        
        logger.info("Built CPC-AE model architecture")
    
    def extract_patches(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Extract patches from images.
        
        Args:
            images: Input images tensor of shape (batch, height, width, channels)
            
        Returns:
            Tuple of (patches, patch_indices)
        """
        batch_size = tf.shape(images)[0]
        patches = []
        patch_indices = []
        
        for i in range(batch_size):
            image = images[i]
            image_patches = []
            image_indices = []
            
            # Extract patches with stride
            for y in range(0, self.input_shape[0] - self.patch_size[0] + 1, self.stride):
                for x in range(0, self.input_shape[1] - self.patch_size[1] + 1, self.stride):
                    patch = image[y:y + self.patch_size[0], x:x + self.patch_size[1], :]
                    image_patches.append(patch)
                    image_indices.append([y, x])
            
            patches.extend(image_patches)
            patch_indices.extend(image_indices)
        
        return tf.stack(patches), tf.stack(patch_indices)
    
    def reconstruct_image(self, patches: tf.Tensor, patch_indices: tf.Tensor, 
                         image_shape: Tuple[int, int, int]) -> tf.Tensor:
        """Reconstruct image from patches.
        
        Args:
            patches: Reconstructed patches
            patch_indices: Patch positions
            image_shape: Target image shape
            
        Returns:
            Reconstructed image
        """
        reconstructed = tf.zeros(image_shape)
        patch_count = tf.zeros(image_shape)
        
        for i in range(tf.shape(patches)[0]):
            patch = patches[i]
            y, x = patch_indices[i][0], patch_indices[i][1]
            
            # Add patch to reconstructed image
            reconstructed = tf.tensor_scatter_nd_add(
                reconstructed,
                [[y + dy, x + dx, c] for dy in range(self.patch_size[0]) 
                 for dx in range(self.patch_size[1]) for c in range(image_shape[2])],
                tf.reshape(patch, [-1])
            )
            
            # Update patch count
            patch_count = tf.tensor_scatter_nd_add(
                patch_count,
                [[y + dy, x + dx, c] for dy in range(self.patch_size[0]) 
                 for dx in range(self.patch_size[1]) for c in range(image_shape[2])],
                tf.ones([self.patch_size[0] * self.patch_size[1] * image_shape[2]])
            )
        
        # Avoid division by zero
        patch_count = tf.where(patch_count == 0, tf.ones_like(patch_count), patch_count)
        return reconstructed / patch_count
    
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
        # Extract patches
        patches, patch_indices = self.extract_patches(inputs)
        
        # Get reconstructions
        reconstructions = self.autoencoder([patches, patch_indices], training=False)
        
        # Compute reconstruction error as anomaly score
        reconstruction_error = tf.reduce_mean(tf.square(patches - reconstructions), axis=[1, 2, 3])
        
        # Reconstruct full images
        batch_size = tf.shape(inputs)[0]
        reconstructed_images = []
        
        for i in range(batch_size):
            # Get patches and indices for this image
            start_idx = i * (self.input_shape[0] // self.stride) * (self.input_shape[1] // self.stride)
            end_idx = (i + 1) * (self.input_shape[0] // self.stride) * (self.input_shape[1] // self.stride)
            
            image_patches = patches[start_idx:end_idx]
            image_indices = patch_indices[start_idx:end_idx]
            image_reconstructions = reconstructions[start_idx:end_idx]
            
            # Reconstruct image
            reconstructed_image = self.reconstruct_image(
                image_reconstructions, image_indices, self.input_shape
            )
            reconstructed_images.append(reconstructed_image)
        
        reconstructed_images = tf.stack(reconstructed_images)
        
        # Compute image-level anomaly scores
        image_anomaly_scores = tf.reduce_mean(tf.square(inputs - reconstructed_images), axis=[1, 2, 3])
        
        return reconstructed_images, image_anomaly_scores
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Model output (reconstructions for autoencoder training)
        """
        # Extract patches
        patches, patch_indices = self.extract_patches(inputs)
        
        # Get reconstructions
        reconstructions = self.autoencoder([patches, patch_indices], training=training)
        
        # Reconstruct full images
        batch_size = tf.shape(inputs)[0]
        reconstructed_images = []
        
        for i in range(batch_size):
            # Get patches and indices for this image
            start_idx = i * (self.input_shape[0] // self.stride) * (self.input_shape[1] // self.stride)
            end_idx = (i + 1) * (self.input_shape[0] // self.stride) * (self.input_shape[1] // self.stride)
            
            image_patches = patches[start_idx:end_idx]
            image_indices = patch_indices[start_idx:end_idx]
            image_reconstructions = reconstructions[start_idx:end_idx]
            
            # Reconstruct image
            reconstructed_image = self.reconstruct_image(
                image_reconstructions, image_indices, self.input_shape
            )
            reconstructed_images.append(reconstructed_image)
        
        return tf.stack(reconstructed_images)
    
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
            'patch_size': self.patch_size,
            'stride': self.stride,
            'latent_dim': self.latent_dim,
            'use_positional_encoding': self.use_positional_encoding
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CPC_AE':
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            CPC-AE model instance
        """
        return cls(**config)







