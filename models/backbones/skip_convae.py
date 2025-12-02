"""High-Performance Convolutional Autoencoder with Skip Connections.

This implementation provides a U-Net style autoencoder with:
- Skip connections between encoder and decoder for better reconstruction
- Residual blocks for deeper learning
- Optional attention gates for feature refinement
- Same interface as ConvAutoencoder for compatibility
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import attention layers
try:
    from models.layers.attention import AttentionGate, ResidualBlock
    ATTENTION_AVAILABLE = True
except ImportError:
    logger.warning("Attention layers not available, will skip attention mechanisms")
    ATTENTION_AVAILABLE = False
    # Define minimal ResidualBlock if not available
    class ResidualBlock(layers.Layer):
        def __init__(self, filters, **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
        def build(self, input_shape):
            self.conv1 = layers.Conv2D(self.filters, 3, padding='same')
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv2D(self.filters, 3, padding='same')
            self.bn2 = layers.BatchNormalization()
            self.add = layers.Add()
        def call(self, x, training=None):
            residual = x
            x = self.conv1(x, training=training)
            x = self.bn1(x, training=training)
            x = layers.ReLU()(x)
            x = self.conv2(x, training=training)
            x = self.bn2(x, training=training)
            return self.add([x, residual])


class SkipConvAutoencoder(tf.keras.Model):
    """High-performance convolutional autoencoder with skip connections.
    
    This model uses U-Net style skip connections for better reconstruction quality.
    Architecture features:
    - Encoder with downsampling convolutions
    - Dense bottleneck (latent representation)
    - Decoder with upsampling convolutions
    - Skip connections from encoder to decoder
    - Optional residual blocks for deeper learning
    - Optional attention gates for feature refinement
    """
    
    def __init__(self, config):
        """Initialize the autoencoder with given configuration.
        
        Args:
            config (dict): Configuration containing model architecture parameters
        """
        super(SkipConvAutoencoder, self).__init__()
        
        # Store configuration
        self.model_config = config
        self.input_shape = tuple(config["input_shape"])
        self.latent_dim = config["latent_dim"]
        self.filters = config.get("encoder_filters", [32, 64, 128])
        self.kernel_size = config.get("kernel_size", 3)
        self.strides = config.get("strides", 2)
        self.padding = config.get("padding", "same")
        self.activation = config.get("activation", "relu")
        self.final_activation = config.get("final_activation", "sigmoid")
        self.use_batch_norm = config.get("use_batch_norm", True)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        # Skip connection and enhancement options
        self.use_skip_connections = config.get("use_skip_connections", True)
        self.use_residual_blocks = config.get("use_residual_blocks", True)
        self.use_attention = config.get("use_attention", False) and ATTENTION_AVAILABLE
        
        # Set initializer
        self.kernel_initializer = config.get("kernel_initializer", "he_normal")
        
        # Calculate decoder shapes
        self._calculate_shapes()
        
        # Build encoder and decoder
        self._build_encoder()
        self._build_decoder()
        
        # Set up for reconstruction error calculation
        self.input_data_range = config.get('input_data_range', [0, 1])
    
    def _calculate_shapes(self):
        """Calculate intermediate shapes for skip connections."""
        h, w = self.input_shape[0], self.input_shape[1]
        self.encoder_shapes = []
        
        for filters in self.filters:
            h = h // self.strides
            w = w // self.strides
            self.encoder_shapes.append((h, w, filters))
        
        # Final encoder shape (before bottleneck)
        self.encoder_output_shape = self.encoder_shapes[-1]
        self.initial_units = h * w * self.filters[-1]
        
        logger.info(f"Encoder output shape: {self.encoder_output_shape}")
        logger.info(f"Initial decoder units: {self.initial_units}")
    
    def _build_encoder(self):
        """Build the encoder layers."""
        self.encoder_blocks = []
        
        for i, filters in enumerate(self.filters):
            block = []
            
            # Conv2D for downsampling
            conv = layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                name=f'encoder_conv_{i}'
            )
            block.append(conv)
            
            # Batch normalization
            if self.use_batch_norm:
                block.append(layers.BatchNormalization(name=f'encoder_bn_{i}'))
            
            # Activation
            block.append(layers.Activation(self.activation, name=f'encoder_act_{i}'))
            
            # Residual block if enabled
            if self.use_residual_blocks:
                block.append(ResidualBlock(filters, name=f'encoder_residual_{i}'))
            
            # Dropout
            if self.dropout_rate > 0:
                block.append(layers.Dropout(self.dropout_rate, name=f'encoder_drop_{i}'))
            
            self.encoder_blocks.append(block)
        
        # Flatten for bottleneck
        self.flatten = layers.Flatten(name='encoder_flatten')
        
        # Dense bottleneck
        self.encoder_dense = layers.Dense(
            self.latent_dim,
            kernel_initializer=self.kernel_initializer,
            name='encoder_dense'
        )
    
    def _build_decoder(self):
        """Build the decoder layers."""
        self.decoder_blocks = []
        
        # Dense layer to expand from latent
        self.decoder_dense = layers.Dense(
            self.initial_units,
            kernel_initializer=self.kernel_initializer,
            name='decoder_dense'
        )
        
        # Reshape to spatial dimensions
        self.decoder_reshape = layers.Reshape(
            self.encoder_output_shape,
            name='decoder_reshape'
        )
        
        # Build decoder blocks - need one for each encoder downsampling step
        # We reshape from the last encoder output (after all encoder blocks)
        # Skip connections are stored after encoder blocks 0..n-2 (excluding the last one)
        # Decoder filters should match skip connection filters (reversed)
        # For example: encoder filters [64, 128, 256, 512] with skips after [64, 128, 256]
        # Decoder should have blocks for [256, 128, 64] + one final block to get to input size
        reversed_skip_filters = list(reversed(self.filters[:-1]))  # All but last, reversed
        
        # Build decoder blocks matching skip connections
        for i, skip_filters in enumerate(reversed_skip_filters):
            block = []
            
            # Conv2DTranspose for upsampling - filters match the skip connection we'll concatenate
            conv_transpose = layers.Conv2DTranspose(
                filters=skip_filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                name=f'decoder_conv_{i}'
            )
            block.append(conv_transpose)
            
            # Attention gate if enabled (before skip connection)
            if self.use_attention and self.use_skip_connections:
                block.append(AttentionGate(skip_filters // 2, name=f'decoder_attention_{i}'))
            
            # Skip connection concatenation happens in call()
            # After concatenation, we'll have 2x channels, so add a conv to reduce back
            if self.use_skip_connections:
                block.append(layers.Conv2D(
                    skip_filters,
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    name=f'decoder_reduce_{i}'
                ))
            
            # Batch normalization
            if self.use_batch_norm:
                block.append(layers.BatchNormalization(name=f'decoder_bn_{i}'))
            
            # Activation
            block.append(layers.Activation(self.activation, name=f'decoder_act_{i}'))
            
            # Residual block if enabled
            if self.use_residual_blocks:
                block.append(ResidualBlock(skip_filters, name=f'decoder_residual_{i}'))
            
            # Dropout
            if self.dropout_rate > 0:
                block.append(layers.Dropout(self.dropout_rate, name=f'decoder_drop_{i}'))
            
            self.decoder_blocks.append(block)
        
        # Add final decoder block if we need one more upsampling step to reach input size
        # This happens when we have skip connections but still need one more upsampling
        # Calculate how many upsampling steps we need total
        encoder_output_h = self.encoder_output_shape[0]
        input_h = self.input_shape[0]
        num_upsampling_steps_needed = int(np.log2(input_h / encoder_output_h))
        
        if len(self.decoder_blocks) < num_upsampling_steps_needed:
            # Need one more block
            final_filters = self.filters[0]  # Use first encoder filter count
            block = []
            conv_transpose = layers.Conv2DTranspose(
                filters=final_filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                name=f'decoder_conv_final'
            )
            block.append(conv_transpose)
            
            if self.use_batch_norm:
                block.append(layers.BatchNormalization(name=f'decoder_bn_final'))
            block.append(layers.Activation(self.activation, name=f'decoder_act_final'))
            if self.use_residual_blocks:
                block.append(ResidualBlock(final_filters, name=f'decoder_residual_final'))
            if self.dropout_rate > 0:
                block.append(layers.Dropout(self.dropout_rate, name=f'decoder_drop_final'))
            
            self.decoder_blocks.append(block)
        
        # Final output layer
        self.final_layer = layers.Conv2D(
            self.input_shape[-1],
            kernel_size=1,
            padding='same',
            activation=self.final_activation,
            name='decoder_output'
        )
    
    def call(self, inputs, training=False):
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            tensor: Reconstructed output
        """
        # Encoder path - collect skip connections
        x = inputs
        skip_connections = []
        
        for i, block in enumerate(self.encoder_blocks):
            # Apply all layers in encoder block
            for layer in block:
                x = layer(x, training=training)
            
            # Store skip connection (before last block, since we reshape from last block's output)
            if self.use_skip_connections and i < len(self.encoder_blocks) - 1:
                skip_connections.append(x)
        
        # Bottleneck
        x = self.flatten(x)
        latent = self.encoder_dense(x)
        
        # Decode with skip connections
        x = self.decoder_dense(latent)
        x = self.decoder_reshape(x)
        
        # Decoder path with skip connections (reversed order)
        skip_connections = list(reversed(skip_connections))
        
        for i, block in enumerate(self.decoder_blocks):
            # Upsample first
            x = block[0](x, training=training)  # Conv2DTranspose
            
            # Apply skip connection if enabled
            if self.use_skip_connections and i < len(skip_connections):
                skip = skip_connections[i]
                
                # Resize skip if needed (should match after upsampling)
                if x.shape[1:3] != skip.shape[1:3]:
                    skip = tf.image.resize(skip, x.shape[1:3], method='bilinear')
                
                # Apply attention if enabled (attention takes [skip, decoder] and returns (att_map, attended_skip))
                block_idx = 1
                if self.use_attention and len(block) > block_idx and hasattr(block[block_idx], '__class__') and 'Attention' in block[block_idx].__class__.__name__:
                    _, attended_skip = block[block_idx]([skip, x], training=training)
                    # Concatenate attended skip with decoder features
                    x = layers.Concatenate()([attended_skip, x])
                    block_idx += 1
                else:
                    # Concatenate skip connection directly
                    x = layers.Concatenate()([skip, x])
                
                # Reduce channels after concatenation (if we have a reduce layer)
                if len(block) > block_idx and isinstance(block[block_idx], layers.Conv2D) and 'reduce' in block[block_idx].name:
                    x = block[block_idx](x, training=training)
                    block_idx += 1
                
                start_idx = block_idx
            else:
                start_idx = 1
            
            # Apply remaining layers in block
            for j in range(start_idx, len(block)):
                x = block[j](x, training=training)
        
        # Final output
        return self.final_layer(x, training=training)
    
    def get_encoder_features(self, inputs):
        """Extract features from encoder layers for anomaly detection.
        
        Args:
            inputs: Input tensor
            
        Returns:
            List of feature tensors from different encoder layers
        """
        features = []
        x = inputs
        
        # Extract features from each encoder block
        for i, block in enumerate(self.encoder_blocks):
            # Apply all layers in block
            for layer in block:
                x = layer(x, training=False)
            
            # Store feature after each encoder block
            features.append(x)
        
        # Add latent/bottleneck features
        x = self.flatten(x)
        latent = self.encoder_dense(x)
        features.append(latent)
        
        return features
