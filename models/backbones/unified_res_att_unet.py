"""Unified Residual Attention U-Net implementation.

This implementation uses ConvAE as the base architecture and adds optional layers:
- Residual blocks
- Attention gates  
- Skip connections

The key insight is that ConvAE provides a solid foundation, and we can enhance it
with additional features rather than having completely separate architectures.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from loguru import logger

# Import corrected SSIM library
try:
    from metrics.ssimlib import SSIMLibrary
    SSIM_LIBRARY_AVAILABLE = True
    logger.info("Corrected SSIM library imported successfully")
except ImportError as e:
    SSIM_LIBRARY_AVAILABLE = False
    logger.warning(f"Corrected SSIM library not available: {e}")

try:
    from models.layers.attention import AttentionGate, ResidualBlock
except ImportError:
    # Fallback import path
    raise ImportError("Could not find attention layers")


class UnifiedResidualAttentionUNet(tf.keras.Model):
    """Unified Residual Attention U-Net with ConvAE as base architecture.
    
    This implementation uses ConvAE as the foundation and adds optional enhancements:
    - Residual blocks in encoder/decoder
    - Attention gates in decoder
    - Skip connections between encoder and decoder
    
    Architecture Flow:
    1. ConvAE Encoder: Conv2D + Downsample → Flatten → Dense (bottleneck)
    2. ConvAE Decoder: Dense → Reshape → Conv2DTranspose layers
    3. Optional Enhancements:
       - Residual blocks: Wrapped around Conv2D/Conv2DTranspose
       - Skip connections: Concatenate encoder features with decoder
       - Attention gates: Apply attention before skip connections
    
    This approach ensures:
    - Consistent bottleneck (dense layer)
    - Flexible feature extraction
    - Gradual complexity addition
    - Better parameter efficiency
    """
    
    def __init__(self, input_shape, filters=[32, 64, 128], use_attention=True, 
                 use_residual=True, use_skip_connections=True, output_channels=None, 
                 loss_function="mse", dropout_rate=0.0, latent_dim=16, **kwargs):
        """Initialize Unified Residual Attention U-Net.
        
        Args:
            input_shape: Input shape (height, width, channels)
            filters: List of filter sizes for encoder/decoder layers
            use_attention: Whether to use attention gates in decoder
            use_residual: Whether to use residual blocks in encoder/decoder
            use_skip_connections: Whether to use skip connections
            output_channels: Number of output channels (default: input channels)
            loss_function: Loss function ("mse", "mae", "ssim", "ssim_mae", "bce")
            dropout_rate: Dropout rate for regularization
            latent_dim: Latent dimension for dense bottleneck
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.filters = filters
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_skip_connections = use_skip_connections
        self.output_channels = output_channels if output_channels is not None else input_shape[-1]
        self.loss_function = loss_function
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        
        # Calculate shapes for decoder
        self._calculate_decoder_shapes()
        
        # Build the model layers
        self._build_layers()
        
        # Initialize attention extractor as None
        self._attention_extractor_model = None
        self._attention_layer_names = []
        # Initialize SSIM library
        if SSIM_LIBRARY_AVAILABLE:
            self.ssim_library = SSIMLibrary()
            logger.info("SSIM library initialized for corrected implementations")
        else:
            self.ssim_library = None
            logger.warning("SSIM library not available, will use fallback implementations")
            
    def _calculate_decoder_shapes(self):
        """Calculate the shapes needed for decoder reconstruction."""
        # Calculate encoder output shape
        h, w = self.input_shape[0], self.input_shape[1]
        for _ in self.filters:
            h = h // 2
            w = w // 2
        
        self.encoder_output_shape = (h, w, self.filters[-1])
        self.initial_units = h * w * self.filters[-1]
        
        logger.info(f"Encoder output shape: {self.encoder_output_shape}")
        logger.info(f"Initial decoder units: {self.initial_units}")
    
    def _build_encoder(self, inputs):
        """Build encoder with optional residual blocks and skip connections.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Tuple of (bottleneck_tensor, skip_connections)
        """
        x = inputs
        skip_connections = []
        
        # Encoder layers with optional residual blocks
        for i, filters in enumerate(self.filters):
            # Downsampling convolution
            x = layers.Conv2D(
                filters, 3, strides=2, padding='same', 
                name=f'encoder_conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'encoder_bn_{i}')(x)
            x = layers.Activation('relu', name=f'encoder_relu_{i}')(x)
            
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'encoder_dropout_{i}')(x)
            
            # Apply residual block if enabled
            if self.use_residual:
                x = ResidualBlock(filters, name=f'encoder_residual_{i}')(x)
            
            # Store skip connection if enabled
            # Store for all encoder layers except the last one (bottleneck level)
            if self.use_skip_connections and i < len(self.filters) - 1:
                skip_connections.append(x)
        
        # Dense bottleneck (ConvAE-style)
        x = layers.Flatten(name='encoder_flatten')(x)
        x = layers.Dense(self.latent_dim, activation='relu', name='encoder_bottleneck')(x)
        
        return x, skip_connections
    
    def _build_decoder(self, x, skip_connections):
        """Build decoder with optional residual blocks, attention, and skip connections.
        
        Args:
            x: Input tensor from bottleneck
            skip_connections: List of skip connection tensors
            
        Returns:
            Decoded tensor
        """
        # Initial dense layer to reshape to spatial dimensions
        x = layers.Dense(self.initial_units, activation='relu', name='decoder_dense')(x)
        x = layers.Reshape(self.encoder_output_shape, name='decoder_reshape')(x)
        
        # Apply dropout if enabled
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name='decoder_initial_dropout')(x)
        
        # Decoder layers - we need to upsample to match the encoder's downsampling
        # For vanilla ConvAE (no skip connections), we still need to upsample the same number of times
        num_upsampling_steps = len(self.filters)  # Same as number of encoder downsampling steps
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i in range(num_upsampling_steps):
            # Determine the target filter size for this decoder layer
            # We go backwards through the filters
            target_filters = self.filters[-(i+1)] if i+1 <= len(self.filters) else self.filters[0]
            
            # Upsampling
            x = layers.Conv2DTranspose(
                target_filters, 3, strides=2, padding='same', 
                name=f'decoder_convt_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
            x = layers.Activation('relu', name=f'decoder_relu_{i}')(x)
            
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'decoder_dropout_{i}')(x)
            
            # Apply skip connection if available and enabled
            if self.use_skip_connections and i < len(skip_connections):
                skip = skip_connections[i]
                
                # Apply attention if enabled
                if self.use_attention:
                    _, x = AttentionGate(target_filters, name=f'decoder_attention_{i}')([x, skip])
                
                # Concatenate skip connection
                x = layers.Concatenate(name=f'decoder_concat_{i}')([x, skip])
                
                # Additional convolution to reduce channels after concatenation
                x = layers.Conv2D(
                    target_filters, 3, padding='same', 
                    name=f'decoder_post_concat_conv_{i}'
                )(x)
                x = layers.BatchNormalization(name=f'decoder_post_concat_bn_{i}')(x)
                x = layers.Activation('relu', name=f'decoder_post_concat_relu_{i}')(x)
            
            # Apply residual block if enabled
            if self.use_residual:
                x = ResidualBlock(target_filters, name=f'decoder_residual_{i}')(x)
        
        return x
    
    def _build_model(self):
        """Build the complete unified model architecture."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Encoder with optional enhancements
        bottleneck, skip_connections = self._build_encoder(inputs)
        
        # Decoder with optional enhancements
        x = self._build_decoder(bottleneck, skip_connections)
        
        # Final output layer - ensure correct number of channels
        outputs = layers.Conv2D(
            self.output_channels, 1, padding='same', 
            activation='sigmoid', name='output'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='unified_res_att_unet')
        
        # Build the model with a dummy input
        dummy_input = tf.zeros((1,) + tuple(self.input_shape))
        _ = self.model(dummy_input, training=False)
        
        # Store layer references
        self.layer_dict = {layer.name: layer for layer in self.model.layers}
        
        logger.info(f"Unified ResAttUNet created with {self.model.count_params():,} parameters")
        logger.info(f"Features: residual={self.use_residual}, attention={self.use_attention}, skip_connections={self.use_skip_connections}")
    
    def _build_layers(self):
        """Build all the layers for the model."""
        # Encoder layers
        self.encoder_layers = []
        for i, filters in enumerate(self.filters):
            layer_group = []
            layer_group.append(layers.Conv2D(
                filters, 3, strides=2, padding='same', 
                name=f'encoder_conv_{i}'
            ))
            layer_group.append(layers.BatchNormalization(name=f'encoder_bn_{i}'))
            layer_group.append(layers.Activation('relu', name=f'encoder_relu_{i}'))
            
            if self.dropout_rate > 0:
                layer_group.append(layers.Dropout(self.dropout_rate, name=f'encoder_dropout_{i}'))
            
            if self.use_residual:
                layer_group.append(ResidualBlock(filters, name=f'encoder_residual_{i}'))
            
            self.encoder_layers.append(layer_group)
        
        # Bottleneck layers
        self.flatten_layer = layers.Flatten(name='encoder_flatten')
        self.bottleneck_layer = layers.Dense(self.latent_dim, activation='relu', name='encoder_bottleneck')
        
        # Decoder layers
        self.decoder_layers = []
        for i in range(len(self.filters)):
            layer_group = []
            target_filters = self.filters[-(i+1)] if i+1 <= len(self.filters) else self.filters[0]
            
            layer_group.append(layers.Conv2DTranspose(
                target_filters, 3, strides=2, padding='same', 
                name=f'decoder_convt_{i}'
            ))
            layer_group.append(layers.BatchNormalization(name=f'decoder_bn_{i}'))
            layer_group.append(layers.Activation('relu', name=f'decoder_relu_{i}'))
            
            if self.dropout_rate > 0:
                layer_group.append(layers.Dropout(self.dropout_rate, name=f'decoder_dropout_{i}'))
            
            if self.use_skip_connections and i < len(self.filters) - 1:
                if self.use_attention:
                    layer_group.append(AttentionGate(target_filters, name=f'decoder_attention_{i}'))
                layer_group.append(layers.Concatenate(name=f'decoder_concat_{i}'))
                layer_group.append(layers.Conv2D(
                    target_filters, 3, padding='same', 
                    name=f'decoder_post_concat_conv_{i}'
                ))
                layer_group.append(layers.BatchNormalization(name=f'decoder_post_concat_bn_{i}'))
                layer_group.append(layers.Activation('relu', name=f'decoder_post_concat_relu_{i}'))
            
            if self.use_residual:
                layer_group.append(ResidualBlock(target_filters, name=f'decoder_residual_{i}'))
            
            self.decoder_layers.append(layer_group)
        
        # Output layer
        self.output_layer = layers.Conv2D(
            self.output_channels, 1, padding='same', 
            activation='sigmoid', name='output'
        )
        
        # Dense layers for decoder
        self.decoder_dense = layers.Dense(self.initial_units, activation='relu', name='decoder_dense')
        self.decoder_reshape = layers.Reshape(self.encoder_output_shape, name='decoder_reshape')
        if self.dropout_rate > 0:
            self.decoder_initial_dropout = layers.Dropout(self.dropout_rate, name='decoder_initial_dropout')
    
    def _create_attention_extractor(self):
        """Create attention feature extractor."""
        if not self.use_attention:
            return

        attention_outputs = []
        attention_layer_names = []
        for layer in self.layers:
            if 'decoder_attention_' in layer.name:
                attention_outputs.append(layer.output[0])  # Get gate signal
                attention_layer_names.append(layer.name)
        
        if not attention_outputs:
            logger.warning("No attention gates found in model")
            return
            
        self._attention_extractor_model = tf.keras.Model(
            inputs=self.inputs,
            outputs=attention_outputs,
            name="unified_attention_extractor"
        )
        self._attention_layer_names = attention_layer_names
        logger.info(f"Created attention extractor for: {self._attention_layer_names}")
    
    def get_loss_function(self):
        """Get the loss function based on configuration."""
        if self.loss_function == "mse":
            return tf.keras.losses.MeanSquaredError()
        elif self.loss_function == "mae":
            return tf.keras.losses.MeanAbsoluteError()
        elif self.loss_function == "ssim":
            return self._ssim_loss  # Keep basic SSIM
        elif self.loss_function == "g_ssim":
            return self._g_ssim_loss  # NEW: Gradient SSIM
        elif self.loss_function == "ssim_4":
            return self._ssim_4_loss  # NEW: Four-component SSIM  
        elif self.loss_function == "ssim_4_g":
            return self._ssim_4_g_loss  # NEW: Four-component Gradient SSIM
        elif self.loss_function == "ms_ssim_4_g":
            return self._ms_ssim_4_g_loss  # NEW: Multi-scale Four-component Gradient SSIM
        elif self.loss_function == "ssim_mae":
            return self._ssim_mae_loss
        elif self.loss_function in ["bce", "binary_crossentropy"]:
            return tf.keras.losses.BinaryCrossentropy()
        else:
            logger.warning(f"Unknown loss function: {self.loss_function}, using MSE")
            return tf.keras.losses.MeanSquaredError()
    
    def build_model(self):
        """Ensure model is built."""
        if self.model is None:
            self._build_model()
            # Setup attention extractor if needed
            if self.use_attention:
                self._create_attention_extractor()
    
    def _ssim_loss(self, y_true, y_pred):
        """SSIM-based loss function."""
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        # Handle channel mismatch using TensorFlow operations
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        
        # Use tf.cond instead of Python if to avoid graph mode issues
        min_channels = tf.minimum(y_true_shape[-1], y_pred_shape[-1])
        
        # Always slice to the minimum channels - this handles both cases safely
        y_true = y_true[..., :min_channels]
        y_pred = y_pred[..., :min_channels]
        
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return tf.reduce_mean(1.0 - ssim)
    
    def _ssim_mae_loss(self, y_true, y_pred):
        """Combined SSIM and MAE loss function."""
        ssim_loss = self._ssim_loss(y_true, y_pred)
        mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        return 0.5 * ssim_loss + 0.5 * mae_loss
    
    def _g_ssim_loss(self, y_true, y_pred):
        """Gradient-based SSIM loss using corrected implementation."""
        if not SSIM_LIBRARY_AVAILABLE or self.ssim_library is None:
            logger.warning("SSIM library not available, falling back to regular SSIM")
            return self._ssim_loss(y_true, y_pred)
        
        # Convert TensorFlow tensors to numpy for SSIM library
        y_true_np = tf.py_function(lambda x: x.numpy(), [y_true], tf.float32)
        y_pred_np = tf.py_function(lambda x: x.numpy(), [y_pred], tf.float32)
        
        def compute_g_ssim_batch(y_true_batch, y_pred_batch):
            batch_scores = []
            for i in range(y_true_batch.shape[0]):
                # Convert to grayscale if needed
                true_img = y_true_batch[i]
                pred_img = y_pred_batch[i]
                
                if len(true_img.shape) == 3 and true_img.shape[-1] > 1:
                    true_img = tf.image.rgb_to_grayscale(true_img)[:,:,0]
                    pred_img = tf.image.rgb_to_grayscale(pred_img)[:,:,0]
                elif len(true_img.shape) == 3:
                    true_img = true_img[:,:,0]
                    pred_img = pred_img[:,:,0]
                
                g_ssim_score = self.ssim_library.calculate_g_ssim(true_img.numpy(), pred_img.numpy())
                batch_scores.append(1.0 - g_ssim_score)  # Convert to loss
            
            return tf.constant(batch_scores, dtype=tf.float32)
        
        loss_values = tf.py_function(compute_g_ssim_batch, [y_true_np, y_pred_np], tf.float32)
        return tf.reduce_mean(loss_values)

    def _ssim_4_loss(self, y_true, y_pred):
        """Four-component SSIM loss using corrected implementation."""
        if not SSIM_LIBRARY_AVAILABLE or self.ssim_library is None:
            logger.warning("SSIM library not available, falling back to regular SSIM")
            return self._ssim_loss(y_true, y_pred)
        
        def compute_4_ssim_batch(y_true_batch, y_pred_batch):
            batch_scores = []
            for i in range(y_true_batch.shape[0]):
                true_img = y_true_batch[i]
                pred_img = y_pred_batch[i]
                
                if len(true_img.shape) == 3 and true_img.shape[-1] > 1:
                    true_img = tf.image.rgb_to_grayscale(true_img)[:,:,0]
                    pred_img = tf.image.rgb_to_grayscale(pred_img)[:,:,0]
                elif len(true_img.shape) == 3:
                    true_img = true_img[:,:,0]
                    pred_img = pred_img[:,:,0]
                
                ssim_4_score = self.ssim_library.calculate_4_ssim(true_img.numpy(), pred_img.numpy())
                batch_scores.append(1.0 - ssim_4_score)
            
            return tf.constant(batch_scores, dtype=tf.float32)
        
        y_true_np = tf.py_function(lambda x: x.numpy(), [y_true], tf.float32)
        y_pred_np = tf.py_function(lambda x: x.numpy(), [y_pred], tf.float32)
        loss_values = tf.py_function(compute_4_ssim_batch, [y_true_np, y_pred_np], tf.float32)
        return tf.reduce_mean(loss_values)

    def _ssim_4_g_loss(self, y_true, y_pred):
        """Four-component Gradient SSIM loss using corrected implementation."""
        if not SSIM_LIBRARY_AVAILABLE or self.ssim_library is None:
            logger.warning("SSIM library not available, falling back to regular SSIM")
            return self._ssim_loss(y_true, y_pred)
        
        def compute_4_g_ssim_batch(y_true_batch, y_pred_batch):
            batch_scores = []
            for i in range(y_true_batch.shape[0]):
                true_img = y_true_batch[i]
                pred_img = y_pred_batch[i]
                
                if len(true_img.shape) == 3 and true_img.shape[-1] > 1:
                    true_img = tf.image.rgb_to_grayscale(true_img)[:,:,0]
                    pred_img = tf.image.rgb_to_grayscale(pred_img)[:,:,0]
                elif len(true_img.shape) == 3:
                    true_img = true_img[:,:,0]
                    pred_img = pred_img[:,:,0]
                
                ssim_4_g_score = self.ssim_library.calculate_4_g_ssim(true_img.numpy(), pred_img.numpy())
                batch_scores.append(1.0 - ssim_4_g_score)
            
            return tf.constant(batch_scores, dtype=tf.float32)
        
        y_true_np = tf.py_function(lambda x: x.numpy(), [y_true], tf.float32)
        y_pred_np = tf.py_function(lambda x: x.numpy(), [y_pred], tf.float32)
        loss_values = tf.py_function(compute_4_g_ssim_batch, [y_true_np, y_pred_np], tf.float32)
        return tf.reduce_mean(loss_values)

    def _ms_ssim_4_g_loss(self, y_true, y_pred):
        """Multi-scale Four-component Gradient SSIM loss using corrected implementation."""
        if not SSIM_LIBRARY_AVAILABLE or self.ssim_library is None:
            logger.warning("SSIM library not available, falling back to regular MS-SSIM")
            return self._ms_ssim_loss(y_true, y_pred)
        
        def compute_ms_4_g_ssim_batch(y_true_batch, y_pred_batch):
            batch_scores = []
            for i in range(y_true_batch.shape[0]):
                true_img = y_true_batch[i]
                pred_img = y_pred_batch[i]
                
                if len(true_img.shape) == 3 and true_img.shape[-1] > 1:
                    true_img = tf.image.rgb_to_grayscale(true_img)[:,:,0]
                    pred_img = tf.image.rgb_to_grayscale(pred_img)[:,:,0]
                elif len(true_img.shape) == 3:
                    true_img = true_img[:,:,0]
                    pred_img = pred_img[:,:,0]
                
                ms_ssim_4_g_score = self.ssim_library.calculate_4_ms_g_ssim(true_img.numpy(), pred_img.numpy())
                batch_scores.append(1.0 - ms_ssim_4_g_score)
            
            return tf.constant(batch_scores, dtype=tf.float32)
        
        y_true_np = tf.py_function(lambda x: x.numpy(), [y_true], tf.float32)
        y_pred_np = tf.py_function(lambda x: x.numpy(), [y_pred], tf.float32)
        loss_values = tf.py_function(compute_ms_4_g_ssim_batch, [y_true_np, y_pred_np], tf.float32)
        return tf.reduce_mean(loss_values)
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs
        skip_connections = []
        
        # Encoder
        for i, layer_group in enumerate(self.encoder_layers):
            for layer in layer_group:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            
            # Store skip connection if enabled
            if self.use_skip_connections and i < len(self.encoder_layers) - 1:
                skip_connections.append(x)
        
        # Bottleneck
        x = self.flatten_layer(x)
        x = self.bottleneck_layer(x)
        
        # Decoder
        x = self.decoder_dense(x)
        x = self.decoder_reshape(x)
        
        if self.dropout_rate > 0:
            x = self.decoder_initial_dropout(x, training=training)
        
        # Decoder layers
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for i, layer_group in enumerate(self.decoder_layers):
            for layer in layer_group:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                elif isinstance(layer, layers.Concatenate):
                    if i < len(skip_connections):
                        x = layer([x, skip_connections[i]])
                elif isinstance(layer, AttentionGate):
                    if i < len(skip_connections):
                        _, x = layer([x, skip_connections[i]])
                else:
                    x = layer(x)
        
        # Output
        outputs = self.output_layer(x)
        
        return outputs
    
    def get_encoder_features(self, x):
        """Extract encoder features for analysis."""
        features = []
        current = x
        
        # Extract features from each encoder layer
        for i, filters in enumerate(self.filters):
            current = self.get_layer(f'encoder_conv_{i}')(current)
            current = self.get_layer(f'encoder_bn_{i}')(current)
            current = self.get_layer(f'encoder_relu_{i}')(current)
            
            if self.dropout_rate > 0:
                current = self.get_layer(f'encoder_dropout_{i}')(current)
            
            if self.use_residual:
                current = self.get_layer(f'encoder_residual_{i}')(current)
            
            features.append(tf.convert_to_tensor(current))
        
        return features
    
    def get_attention_features(self, x):
        """Extract attention features for analysis."""
        if not self.use_attention or self._attention_extractor_model is None:
            logger.warning("Attention features not available")
            return {}
        
        features = self._attention_extractor_model(x, training=False)
        
        if not isinstance(features, list):
            features = [features]
        
        return {name: feature for name, feature in zip(self._attention_layer_names, features)}
    
    def get_bottleneck_features(self, x):
        """Extract bottleneck features (dense layer output)."""
        # Run through encoder to get bottleneck
        current = x
        for i, filters in enumerate(self.filters):
            current = self.get_layer(f'encoder_conv_{i}')(current)
            current = self.get_layer(f'encoder_bn_{i}')(current)
            current = self.get_layer(f'encoder_relu_{i}')(current)
            
            if self.dropout_rate > 0:
                current = self.get_layer(f'encoder_dropout_{i}')(current)
            
            if self.use_residual:
                current = self.get_layer(f'encoder_residual_{i}')(current)
        
        # Get bottleneck features
        current = self.get_layer('encoder_flatten')(current)
        bottleneck = self.get_layer('encoder_bottleneck')(current)
        
        return bottleneck

    def get_config(self):
        """Get configuration for Keras serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'filters': self.filters,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'use_skip_connections': self.use_skip_connections,
            'output_channels': self.output_channels,
            'loss_function': self.loss_function,
            'dropout_rate': self.dropout_rate,
            'latent_dim': self.latent_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Handle missing parameters gracefully by providing defaults
        # This is needed because older saved models might not have all parameters
        
        # Extract input_shape - this is required
        input_shape = config.get('input_shape')
        if input_shape is None:
            # Try to infer from other parameters or use default
            # Check if we can find input_shape in the config
            if 'input_shape' not in config:
                # Use default input shape if not specified
                input_shape = (256, 256, 1)
                print(f"Warning: input_shape not found in config, using default: {input_shape}")
        
        # Extract other parameters with defaults
        filters = config.get('filters', [32, 64, 128])
        use_attention = config.get('use_attention', False)
        use_residual = config.get('use_residual', False)
        use_skip_connections = config.get('use_skip_connections', False)
        output_channels = config.get('output_channels')
        loss_function = config.get('loss_function', 'mse')
        dropout_rate = config.get('dropout_rate', 0.0)
        latent_dim = config.get('latent_dim', 16)
        
        # If output_channels is not specified, infer from input_shape
        if output_channels is None and input_shape is not None:
            output_channels = input_shape[-1]
        
        # Create the model with the extracted parameters
        model = cls(
            input_shape=input_shape,
            filters=filters,
            use_attention=use_attention,
            use_residual=use_residual,
            use_skip_connections=use_skip_connections,
            output_channels=output_channels,
            loss_function=loss_function,
            dropout_rate=dropout_rate,
            latent_dim=latent_dim
        )
        
        # Ensure the model is built (this is crucial for loading weights)
        if not hasattr(model, 'model') or model.model is None:
            print("Building model from config...")
            model._build_model()
        
        return model


# Create alias for compatibility
UnifiedResAttUNet = UnifiedResidualAttentionUNet 
