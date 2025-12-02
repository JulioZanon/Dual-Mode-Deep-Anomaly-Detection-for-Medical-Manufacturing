"""RAG-PaDiM: Residual Attention Guided U-Net with PaDiM-based anomaly detection.

This implementation combines:
1. Fine-tuned Residual Attention Guided U-Net (RAG-U-Net) for reconstruction
2. Parameter estimation module for Gaussian modeling
3. PaDiM-style anomaly detection using learned parameters

The architecture follows the diagram:
- Encoder: Conv2D + Pool → Residual blocks → Feature extraction
- Decoder: Upsampling → Attention gates → Skip connections → Reconstruction
- Parameter Estimation: Extract embeddings → Learn Gaussian parameters (μ, Σ)
- Anomaly Detection: Mahalanobis distance using learned parameters
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from loguru import logger
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from tqdm import tqdm

try:
    from models.layers.attention import AttentionGate, ResidualBlock
except ImportError:
    # Fallback import path
    raise ImportError("Could not find attention layers")

try:
    from metrics.ssimlib import SSIMLibrary
    SSIM_LIBRARY_AVAILABLE = True
    logger.info("SSIM library imported successfully")
except ImportError as e:
    SSIM_LIBRARY_AVAILABLE = False
    logger.warning(f"SSIM library not available: {e}")


class ParameterEstimationModule(tf.keras.layers.Layer):
    """Parameter estimation module for learning Gaussian parameters from embeddings."""
    
    def __init__(self, embedding_dim, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        
        # Learnable parameters for Gaussian modeling
        self.mean_estimator = layers.Dense(embedding_dim, name='mean_estimator')
        self.cov_estimator = layers.Dense(embedding_dim * embedding_dim, name='cov_estimator')
        
        # Optional attention mechanism for parameter estimation
        if self.use_attention:
            self.attention_weights = layers.Dense(1, activation='sigmoid', name='param_attention')
    
    def call(self, embeddings, training=None):
        """
        Estimate Gaussian parameters from embeddings.
        
        Args:
            embeddings: Feature embeddings [B, H, W, C] or [B, N, C]
            
        Returns:
            mean: Estimated mean vectors [B, H, W, C] or [B, N, C]
            cov: Estimated covariance matrices [B, H, W, C, C] or [B, N, C, C]
        """
        original_shape = tf.shape(embeddings)
        B, H, W, C = original_shape[0], original_shape[1], original_shape[2], original_shape[3]
        
        # Flatten spatial dimensions
        embeddings_flat = tf.reshape(embeddings, [B, H * W, C])
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention_weights(embeddings_flat)  # [B, H*W, 1]
            embeddings_flat = embeddings_flat * attention_weights
        
        # Estimate mean parameters
        mean_flat = self.mean_estimator(embeddings_flat)  # [B, H*W, C]
        
        # Estimate covariance parameters
        cov_flat = self.cov_estimator(embeddings_flat)  # [B, H*W, C*C]
        cov_flat = tf.reshape(cov_flat, [B, H * W, C, C])
        
        # Ensure covariance is positive definite
        # Add small regularization to diagonal
        identity = tf.eye(C, dtype=cov_flat.dtype)
        cov_flat = cov_flat + 1e-3 * identity
        
        # Reshape back to spatial dimensions
        mean = tf.reshape(mean_flat, [B, H, W, C])
        cov = tf.reshape(cov_flat, [B, H, W, C, C])
        
        return mean, cov


class RAGPaDiM(tf.keras.Model):
    """RAG-PaDiM: Residual Attention Guided U-Net with PaDiM-based anomaly detection."""
    
    def __init__(self, input_shape, filters=[64, 128, 256, 512], use_attention=True, 
                 use_residual=True, use_skip_connections=True, output_channels=None,
                 loss_function="mse", dropout_rate=0.0, latent_dim=16, 
                 n_features=550, use_scipy_gaussian=False, normalize_anomaly_map=False,
                 per_position_stats=False, **kwargs):
        """Initialize RAG-PaDiM model.
        
        Args:
            input_shape: Input shape (height, width, channels)
            filters: List of filter sizes for encoder/decoder layers
            use_attention: Whether to use attention gates in decoder
            use_residual: Whether to use residual blocks in encoder/decoder
            use_skip_connections: Whether to use skip connections
            output_channels: Number of output channels (default: input channels)
            loss_function: Loss function for reconstruction
            dropout_rate: Dropout rate for regularization
            latent_dim: Latent dimension for dense bottleneck
            n_features: Number of features for PaDiM
            use_scipy_gaussian: Whether to use scipy's gaussian_filter
            normalize_anomaly_map: Whether to normalize anomaly map to [0, 1]
            per_position_stats: Whether to use per-position statistics
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
        self.n_features = n_features
        self.use_scipy_gaussian = use_scipy_gaussian
        self.normalize_anomaly_map = normalize_anomaly_map
        self.per_position_stats = per_position_stats
        
        # Calculate shapes for decoder
        self._calculate_decoder_shapes()
        
        # Build the model layers
        self._build_layers()
        
        # Initialize parameter estimation module
        self.parameter_estimation = ParameterEstimationModule(
            embedding_dim=self.n_features,
            use_attention=True,
            name='parameter_estimation'
        )
        
        # Initialize SSIM library
        if SSIM_LIBRARY_AVAILABLE:
            self.ssim_library = SSIMLibrary()
            logger.info("SSIM library initialized")
        else:
            self.ssim_library = None
            logger.warning("SSIM library not available")
        
        # Initialize statistics for anomaly detection
        self.feature_means = None
        self.feature_covs = None
        self.random_indices = None
        self.threshold = None
        
    def _calculate_decoder_shapes(self):
        """Calculate the shapes needed for decoder reconstruction."""
        h, w = self.input_shape[0], self.input_shape[1]
        for _ in self.filters:
            h = h // 2
            w = w // 2
        
        self.encoder_output_shape = (h, w, self.filters[-1])
        self.initial_units = h * w * self.filters[-1]
        
        logger.info(f"Encoder output shape: {self.encoder_output_shape}")
        logger.info(f"Initial decoder units: {self.initial_units}")
    
    def _build_encoder(self, inputs):
        """Build encoder with residual blocks and skip connections."""
        x = inputs
        skip_connections = []
        
        # Encoder layers with residual blocks
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
            
            # Store skip connection if enabled (except for the last layer)
            if self.use_skip_connections and i < len(self.filters) - 1:
                skip_connections.append(x)
        
        # Dense bottleneck (ConvAE-style)
        x = layers.Flatten(name='encoder_flatten')(x)
        x = layers.Dense(self.latent_dim, activation='relu', name='encoder_bottleneck')(x)
        
        return x, skip_connections
    
    def _build_decoder(self, x, skip_connections):
        """Build decoder with attention gates and skip connections."""
        # Initial dense layer to reshape to spatial dimensions
        x = layers.Dense(self.initial_units, activation='relu', name='decoder_dense')(x)
        x = layers.Reshape(self.encoder_output_shape, name='decoder_reshape')(x)
        
        # Apply dropout if enabled
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name='decoder_initial_dropout')(x)
        
        # Decoder layers
        num_upsampling_steps = len(self.filters)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i in range(num_upsampling_steps):
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
    
    def _extract_embeddings(self, inputs, training=False):
        """Extract embeddings from encoder layers for parameter estimation."""
        x = inputs
        embeddings = []
        
        # Extract features from each encoder layer (before residual blocks)
        for i, layer_group in enumerate(self.encoder_layers):
            for layer in layer_group:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                elif not isinstance(layer, ResidualBlock):
                    x = layer(x)
            
            # Store embedding before residual block
            if i < len(self.encoder_layers) - 1:  # Don't include the last layer (bottleneck)
                embeddings.append(x)
        
        # Concatenate embeddings from different scales
        if len(embeddings) > 1:
            # Use the largest feature map as base
            concat_embeddings = embeddings[-1]
            
            # Concatenate with smaller feature maps
            for emb in reversed(embeddings[:-1]):
                concat_embeddings = self._embedding_concat(emb, concat_embeddings)
        else:
            concat_embeddings = embeddings[0]
        
        return concat_embeddings
    
    def _embedding_concat(self, l1, l2):
        """Concatenate embeddings from different scales (PaDiM-style)."""
        bs, h1, w1, c1 = tf.shape(l1)[0], tf.shape(l1)[1], tf.shape(l1)[2], tf.shape(l1)[3]
        _, h2, w2, c2 = tf.shape(l2)[0], tf.shape(l2)[1], tf.shape(l2)[2], tf.shape(l2)[3]
        
        # Calculate scale factor
        s = h1 // h2
        
        # Extract patches from l1 to match l2's spatial dimensions
        x = tf.image.extract_patches(
            l1,
            sizes=[1, s, s, 1],
            strides=[1, s, s, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        
        # Reshape to [B, H2, W2, s*s, C1]
        x = tf.reshape(x, (bs, h2, w2, s*s, c1))
        
        # Reshape to [B, H2, W2, s*s*C1]
        x = tf.reshape(x, (bs, h2, w2, s*s*c1))
        
        # Concatenate with l2 along channel dimension
        z = tf.concat([x, l2], axis=-1)
        
        return z
    
    def call(self, inputs, training=False):
        """Forward pass of the model."""
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
        reconstruction = self.output_layer(x)
        
        # Extract embeddings for parameter estimation
        embeddings = self._extract_embeddings(inputs, training=training)
        
        # Estimate Gaussian parameters
        mean, cov = self.parameter_estimation(embeddings, training=training)
        
        return {
            'reconstruction': reconstruction,
            'embeddings': embeddings,
            'mean': mean,
            'covariance': cov
        }
    
    def get_loss_function(self):
        """Get the loss function based on configuration."""
        if self.loss_function == "mse":
            return tf.keras.losses.MeanSquaredError()
        elif self.loss_function == "mae":
            return tf.keras.losses.MeanAbsoluteError()
        elif self.loss_function == "ssim":
            return self._ssim_loss
        elif self.loss_function == "ssim_mae":
            return self._ssim_mae_loss
        else:
            logger.warning(f"Unknown loss function: {self.loss_function}, using MSE")
            return tf.keras.losses.MeanSquaredError()
    
    def _ssim_loss(self, y_true, y_pred):
        """SSIM-based loss function."""
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return tf.reduce_mean(1.0 - ssim)
    
    def _ssim_mae_loss(self, y_true, y_pred):
        """Combined SSIM and MAE loss function."""
        ssim_loss = self._ssim_loss(y_true, y_pred)
        mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        return 0.5 * ssim_loss + 0.5 * mae_loss
    
    def fit_parameters(self, dataset):
        """Fit Gaussian parameters using normal training data."""
        logger.info("Fitting Gaussian parameters for anomaly detection...")
        
        # Collect all embeddings from normal data
        all_embeddings = []
        
        for batch_idx, (x, _) in enumerate(dataset):
            outputs = self(x, training=False)
            embeddings = outputs['embeddings']
            all_embeddings.append(embeddings)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx} batches")
        
        # Concatenate all embeddings
        all_embeddings = tf.concat(all_embeddings, axis=0)
        logger.info(f"Collected embeddings shape: {all_embeddings.shape}")
        
        # Select random features if needed
        if self.random_indices is None:
            total_channels = all_embeddings.shape[-1]
            if self.n_features > total_channels:
                logger.warning(f"Requested {self.n_features} features but only {total_channels} are available")
                self.n_features = total_channels
                self.random_indices = tf.range(total_channels)
            else:
                indices = np.random.permutation(total_channels)[:self.n_features]
                self.random_indices = tf.convert_to_tensor(indices, dtype=tf.int32)
                logger.info(f"Selected {self.n_features} random features from {total_channels} total")
        
        # Select random channels
        selected_embeddings = tf.gather(all_embeddings, self.random_indices, axis=-1)
        
        # Calculate statistics
        if self.per_position_stats:
            logger.info("Computing per-position statistics...")
            B, H, W, C = selected_embeddings.shape
            embeddings_np = selected_embeddings.numpy()
            
            means = np.zeros((H, W, C), dtype=np.float32)
            inv_covs = np.zeros((H, W, C, C), dtype=np.float32)
            
            for i in range(H):
                for j in range(W):
                    patch_vecs = embeddings_np[:, i, j, :]
                    mu = np.mean(patch_vecs, axis=0)
                    means[i, j, :] = mu
                    
                    diff = patch_vecs - mu
                    cov = np.cov(diff, rowvar=False)
                    
                    # Add regularization
                    reg = 1e-3 * np.eye(C)
                    cov_reg = cov + reg
                    
                    try:
                        L = np.linalg.cholesky(cov_reg)
                        L_inv = np.linalg.inv(L)
                        inv_cov = np.dot(L_inv.T, L_inv)
                        inv_covs[i, j, :, :] = inv_cov
                    except np.linalg.LinAlgError:
                        logger.warning(f"Cholesky failed at position ({i}, {j}), using regular inverse")
                        try:
                            inv_cov = np.linalg.inv(cov_reg)
                            inv_covs[i, j, :, :] = inv_cov
                        except np.linalg.LinAlgError:
                            inv_covs[i, j, :, :] = np.linalg.pinv(cov_reg)
            
            self.feature_means = tf.convert_to_tensor(means, dtype=tf.float32)
            self.feature_covs = tf.convert_to_tensor(inv_covs, dtype=tf.float32)
        else:
            logger.info("Computing global statistics...")
            B, H, W, C = selected_embeddings.shape
            embeddings_flat = tf.reshape(selected_embeddings, [B * H * W, C])
            
            # Compute mean
            means = tf.reduce_mean(embeddings_flat, axis=0)
            
            # Compute covariance with regularization
            diff = embeddings_flat - means
            cov = tf.matmul(tf.transpose(diff), diff) / (B * H * W - 1)
            reg = 1e-5 * tf.eye(C, dtype=cov.dtype)
            cov_reg = cov + reg
            
            # Compute inverse using Cholesky decomposition
            try:
                L = tf.linalg.cholesky(cov_reg)
                L_inv = tf.linalg.inv(L)
                inv_cov = tf.matmul(tf.transpose(L_inv), L_inv)
            except tf.errors.InvalidArgumentError:
                logger.warning("Cholesky failed, using regular inverse")
                inv_cov = tf.linalg.inv(cov_reg)
            
            self.feature_means = means
            self.feature_covs = inv_cov
        
        # Calculate threshold
        self._calculate_threshold(dataset)
        
        logger.success("Parameter fitting completed!")
    
    def _calculate_threshold(self, dataset):
        """Calculate anomaly threshold using training data."""
        logger.info("Calculating anomaly threshold...")
        
        all_distances = []
        
        for batch_idx, (x, _) in enumerate(dataset):
            outputs = self(x, training=False)
            embeddings = outputs['embeddings']
            
            # Select random channels
            selected_embeddings = tf.gather(embeddings, self.random_indices, axis=-1)
            
            # Compute Mahalanobis distances
            if self.per_position_stats:
                distances = self._compute_per_position_mahalanobis(selected_embeddings)
            else:
                distances = self._compute_global_mahalanobis(selected_embeddings)
            
            # Get image-level scores (max of pixel-level distances)
            image_scores = tf.reduce_max(distances, axis=[1, 2])
            all_distances.extend(image_scores.numpy())
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx} batches")
        
        # Calculate threshold using quantile
        all_distances = np.array(all_distances)
        self.threshold = np.percentile(all_distances, 95)
        
        logger.info(f"Threshold statistics:")
        logger.info(f"Mean Mahalanobis distance: {np.mean(all_distances):.4f}")
        logger.info(f"Std Mahalanobis distance: {np.std(all_distances):.4f}")
        logger.info(f"Threshold (95th quantile): {self.threshold:.4f}")
    
    def _compute_per_position_mahalanobis(self, embeddings):
        """Compute per-position Mahalanobis distance."""
        B, H, W, C = embeddings.shape
        embeddings_np = embeddings.numpy()
        means_np = self.feature_means.numpy()
        inv_covs_np = self.feature_covs.numpy()
        
        distances = np.zeros((B, H, W), dtype=np.float32)
        
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    x = embeddings_np[b, i, j, :]
                    mu = means_np[i, j, :]
                    inv_cov = inv_covs_np[i, j, :, :]
                    diff = x - mu
                    m = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
                    distances[b, i, j] = m
        
        return tf.convert_to_tensor(distances, dtype=tf.float32)
    
    def _compute_global_mahalanobis(self, embeddings):
        """Compute global Mahalanobis distance."""
        B, H, W, C = embeddings.shape
        embeddings_flat = tf.reshape(embeddings, [B * H * W, C])
        
        diff = embeddings_flat - self.feature_means
        left = tf.matmul(diff, self.feature_covs)
        mahalanobis = tf.reduce_sum(left * diff, axis=-1)
        mahalanobis = tf.sqrt(tf.maximum(mahalanobis, 0.0))
        mahalanobis = tf.reshape(mahalanobis, [B, H, W])
        
        return mahalanobis
    
    def predict_anomalies(self, inputs, return_scores=False):
        """Predict whether samples are anomalous."""
        outputs = self(inputs, training=False)
        embeddings = outputs['embeddings']
        
        # Select random channels
        selected_embeddings = tf.gather(embeddings, self.random_indices, axis=-1)
        
        # Compute Mahalanobis distances
        if self.per_position_stats:
            distances = self._compute_per_position_mahalanobis(selected_embeddings)
        else:
            distances = self._compute_global_mahalanobis(selected_embeddings)
        
        # Get image-level scores
        scores = tf.reduce_max(distances, axis=[1, 2])
        
        # Make predictions
        threshold = self.threshold if self.threshold is not None else 0.5
        predictions = tf.cast(scores > threshold, tf.int32)
        
        if return_scores:
            return predictions, scores, distances
        return predictions, scores
    
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
            'n_features': self.n_features,
            'use_scipy_gaussian': self.use_scipy_gaussian,
            'normalize_anomaly_map': self.normalize_anomaly_map,
            'per_position_stats': self.per_position_stats,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
