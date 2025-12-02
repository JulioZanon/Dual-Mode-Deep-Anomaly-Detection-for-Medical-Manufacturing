"""Convolutional Autoencoder (ConvAE) implementation.

This module provides a convolutional autoencoder implementation using TensorFlow 
and Keras. The autoencoder consists of an encoder that compresses the input into 
a latent representation and a decoder that reconstructs the input from this 
representation.

Classes:
    ConvAutoencoder: Main autoencoder implementation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, initializers, constraints
import numpy as np
import logging

# Import SSIM library for advanced SSIM loss functions
try:
    from metrics.ssimlib import SSIMLibrary
    SSIM_LIBRARY_AVAILABLE = True
except ImportError:
    SSIM_LIBRARY_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnitLengthConstraint(constraints.Constraint):
    """Constraint to enforce unit length (L2 norm = 1) on weights."""
    
    def __call__(self, w):
        return w / (tf.keras.backend.epsilon() + tf.sqrt(tf.reduce_sum(tf.square(w), axis=[0, 1, 2], keepdims=True)))

class ConvAutoencoder(tf.keras.Model):
    """A convolutional autoencoder implementation.
    
    This model uses convolutional layers for encoding and decoding images,
    with optional batch normalization and dropout.
    """
    
    def __init__(self, config):
        """Initialize the autoencoder with given configuration.
        
        Args:
            config (dict): Configuration containing model architecture parameters
        """
        super(ConvAutoencoder, self).__init__()
        
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
        
        # Add new weight constraint configs
        self.use_tied_weights = config.get("use_tied_weights", False)
        self.use_orthogonal_weights = config.get("use_orthogonal_weights", False)
        self.use_unit_length_weights = config.get("use_unit_length_weights", False)
        
        # Classification and threshold parameters
        self.task = config.get("task", "classification")
        self.threshold_data = config.get("threshold_data", "train")
        self.threshold_percentile = config.get("threshold_percentile", 95.0)
        self.reconstruction_threshold = None
        self.training_errors = None
        self.validation_errors = None
        
        # Set initializer based on configuration
        if self.use_orthogonal_weights:
            self.kernel_initializer = initializers.Orthogonal()
        else:
            self.kernel_initializer = config.get("kernel_initializer", "he_normal")
        
        # Set constraint based on configuration
        if self.use_unit_length_weights:
            self.kernel_constraint = UnitLengthConstraint()
        else:
            self.kernel_constraint = None
        
        # Build encoder and decoder
        self._build_encoder()
        self._build_decoder()
        
        # Set up for reconstruction error calculation
        self.input_data_range = config.get('input_data_range', [0, 1])  # Default range for inputs
        
        # Initialize SSIM library if needed for advanced loss functions
        if SSIM_LIBRARY_AVAILABLE:
            self.ssim_library = SSIMLibrary()
        else:
            self.ssim_library = None
    
    def _build_encoder(self):
        """Build the encoder part of the autoencoder."""
        self.encoder_layers = []
        self.encoder_conv_layers = []  # Store Conv2D layers separately for tied weights
        
        for filters in self.filters:
            # Create Conv2D layer with constraints
            conv_layer = layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                kernel_constraint=self.kernel_constraint,
                name=f'encoder_conv_{filters}'
            )
            self.encoder_conv_layers.append(conv_layer)
            
            # Add all layers to the layer list
            self.encoder_layers.extend([
                conv_layer,
                layers.BatchNormalization(name=f'encoder_bn_{filters}') if self.use_batch_norm else None,
                layers.Activation(self.activation, name=f'encoder_act_{filters}'),
                layers.Dropout(self.dropout_rate, name=f'encoder_drop_{filters}') if self.dropout_rate > 0 else None
            ])
        self.encoder_layers = [layer for layer in self.encoder_layers if layer is not None]
        
        self.flatten = layers.Flatten(name='encoder_flatten')
        self.encoder_dense = layers.Dense(
            self.latent_dim, 
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            name='encoder_dense'
        )
        
        # Calculate initial shape for decoder
        h = self.input_shape[0]
        w = self.input_shape[1]
        for _ in self.filters:
            h = (h + 2 * int(self.padding == 'same') - self.kernel_size) // self.strides + 1
            w = (w + 2 * int(self.padding == 'same') - self.kernel_size) // self.strides + 1
        self.decoder_initial_shape = (h, w, self.filters[-1])
        self.initial_units = h * w * self.filters[-1]
    
    def _build_decoder(self):
        """Build the decoder part of the autoencoder."""
        self.decoder_dense = layers.Dense(
            self.initial_units, 
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            name='decoder_dense'
        )
        self.decoder_reshape = layers.Reshape(self.decoder_initial_shape, name='decoder_reshape')
        
        # Build decoder layers
        self.decoder_layers = []
        self.decoder_conv_layers = []  # Store Conv2DTranspose layers separately for tied weights
        
        reversed_filters = self.filters[:-1][::-1]  # Skip the last one as we'll use it for reshape
        
        for i, filters in enumerate(reversed_filters):
            # Create Conv2DTranspose layer with constraints
            conv_t_layer = layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                kernel_constraint=self.kernel_constraint,
                name=f'decoder_convt_{i}'
            )
            self.decoder_conv_layers.append(conv_t_layer)
            
            # Add all layers to the layer list
            self.decoder_layers.extend([
                conv_t_layer,
                layers.BatchNormalization(name=f'decoder_bn_{i}') if self.use_batch_norm else None,
                layers.Activation(self.activation, name=f'decoder_act_{i}'),
                layers.Dropout(self.dropout_rate, name=f'decoder_drop_{i}') if self.dropout_rate > 0 else None
            ])
        self.decoder_layers = [layer for layer in self.decoder_layers if layer is not None]
        
        # Final output layer
        self.final_layer = layers.Conv2DTranspose(
            filters=self.input_shape[-1],
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.final_activation,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            name='decoder_output'
        )
        
        # Build model with dummy input
        self.build((None,) + self.input_shape)
        
        # Set up tied weights if configured
        if self.use_tied_weights:
            self._setup_tied_weights()
    
    def _setup_tied_weights(self):
        """Set up tied weights between encoder and decoder layers."""
        # Check if we can tie weights (same number of layers)
        if len(self.encoder_conv_layers) != len(self.decoder_conv_layers) + 1:
            logger.warning("Cannot set up tied weights: encoder and decoder have different numbers of layers")
            return
            
        logger.info("Setting up tied weights between encoder and decoder")
        
        # Tie Conv2D weights with Conv2DTranspose weights (excluding the final layer)
        for i, encoder_layer in enumerate(reversed(self.encoder_conv_layers[:-1])):
            decoder_layer = self.decoder_conv_layers[i]
            
            # Create weight sharing relationship - only share kernels, not biases
            # The weight matrices need to be transposed when sharing Conv2D with Conv2DTranspose
            decoder_layer.kernel = tf.keras.backend.transpose(encoder_layer.kernel)
            
            # Add a callback to update tied weights after each batch
            # This is needed because constraints might be applied independently
            def update_kernel(encoder_layer=encoder_layer, decoder_layer=decoder_layer):
                decoder_layer.kernel.assign(tf.keras.backend.transpose(encoder_layer.kernel))
                
            # Add callback to the model
            self.add_update(update_kernel)
            
        logger.info(f"Tied weights set up for {len(self.decoder_conv_layers)} layer pairs")
    
    def call(self, inputs, training=False):
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            tensor: Reconstructed output
        """
        # Encode
        x = inputs
        skips = []
        
        for layer in self.encoder_layers:
            x = layer(x, training=training)
            if isinstance(layer, layers.Conv2D):
                skips.append(x)
        
        x = self.flatten(x)
        latent = self.encoder_dense(x)
        
        # Decode
        x = self.decoder_dense(latent)
        x = self.decoder_reshape(x)
        
        # Update tied weights if necessary before decoder pass
        if self.use_tied_weights and training:
            self._update_tied_weights()
        
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        
        return self.final_layer(x)
    
    def _update_tied_weights(self):
        """Update tied weights manually during training."""
        # This explicitly updates tied weights by calling the update functions
        for update in self.updates:
            update()

    def get_encoder_features(self, inputs):
        """Extract features from encoder layers for anomaly detection.
        
        Args:
            inputs: Input tensor
            
        Returns:
            List of feature tensors from different encoder layers
        """
        features = []
        
        # Encode through layers and collect intermediate features
        x = inputs
        
        # First encoder layer features
        if len(self.encoder_layers) > 0:
            x = self.encoder_layers[0](x, training=False)
            if isinstance(self.encoder_layers[0], layers.Conv2D):
                features.append(x)
        
        # Second encoder layer features
        if len(self.encoder_layers) > 3:  # Account for BN, activation, dropout
            x = self.encoder_layers[3](x, training=False)
            if isinstance(self.encoder_layers[3], layers.Conv2D):
                features.append(x)
        
        # Third encoder layer features
        if len(self.encoder_layers) > 6:  # Account for BN, activation, dropout
            x = self.encoder_layers[6](x, training=False)
            if isinstance(self.encoder_layers[6], layers.Conv2D):
                features.append(x)
        
        # Continue through remaining encoder layers
        for i in range(9, len(self.encoder_layers), 4):  # Every 4th layer (Conv2D)
            if i < len(self.encoder_layers):
                x = self.encoder_layers[i](x, training=False)
                if isinstance(self.encoder_layers[i], layers.Conv2D):
                    features.append(x)
        
        # Add latent/bottleneck features
        x = self.flatten(x)
        latent = self.encoder_dense(x)
        features.append(latent)
        
        return features

    def get_latent_features(self, inputs):
        """Get only the latent/bottleneck features.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Latent feature tensor
        """
        # Encode
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=False)
        
        x = self.flatten(x)
        latent = self.encoder_dense(x)
        
        return latent

    def compute_reconstruction_error(self, original, reconstructed):
        """Compute reconstruction error between original and reconstructed images.
        
        Args:
            original: Original input images
            reconstructed: Reconstructed images from autoencoder
            
        Returns:
            tf.Tensor: Reconstruction error per sample
        """
        # Ensure both inputs are float32 and normalized to [0, 1]
        original = tf.cast(original, tf.float32)
        reconstructed = tf.cast(reconstructed, tf.float32)
        
        # Normalize to [0, 1] if not already
        if tf.reduce_max(original) > 1.0:
            original = original / 255.0
        if tf.reduce_max(reconstructed) > 1.0:
            reconstructed = reconstructed / 255.0
        
        # Handle shape mismatches
        if len(original.shape) != len(reconstructed.shape):
            logger.warning(f"Dimension mismatch - original: {original.shape}, reconstructed: {reconstructed.shape}")
            if len(original.shape) == 1:  # If original is just labels
                return tf.zeros_like(original, dtype=tf.float32)  # Return zero error
        
        # Ensure both have 4 dimensions (batch, height, width, channels)
        if len(original.shape) == 3:
            original = tf.expand_dims(original, 0)
        if len(reconstructed.shape) == 3:
            reconstructed = tf.expand_dims(reconstructed, 0)
        
        # Ensure spatial dimensions match
        if original.shape[1:3] != reconstructed.shape[1:3]:
            logger.warning(f"Spatial dimension mismatch - original: {original.shape}, reconstructed: {reconstructed.shape}")
            reconstructed = tf.image.resize(reconstructed, [original.shape[1], original.shape[2]])
        
        # Handle batch dimension mismatch
        if original.shape[0] != reconstructed.shape[0]:
            logger.warning(f"Batch dimension mismatch - original: {original.shape[0]}, reconstructed: {reconstructed.shape[0]}")
            # If scalar vs batch issue, broadcast the scalar
            if original.shape[0] == 1:
                original = tf.repeat(original, repeats=reconstructed.shape[0], axis=0)
            elif reconstructed.shape[0] == 1:
                reconstructed = tf.repeat(reconstructed, repeats=original.shape[0], axis=0)
        
        try:
            # Get the loss type from model config (check both 'loss' and 'loss_function' keys)
            loss_type = self.model_config.get('loss') or self.model_config.get('loss_function', 'mse')
            if isinstance(loss_type, str):
                loss_type = loss_type.lower()
            else:
                loss_type = 'mse'
            
            if loss_type == 'ssim':
                # SSIM returns a value between -1 and 1, where 1 means identical images
                # We convert to an error by taking 1 - SSIM
                ssim_value = tf.image.ssim(
                    original, 
                    reconstructed, 
                    max_val=1.0,
                    filter_size=11,  # Larger filter size for better structural similarity
                    filter_sigma=1.5,  # Standard deviation for Gaussian filter
                    k1=0.01,  # Stability constant for luminance
                    k2=0.03   # Stability constant for contrast
                )
                return 1.0 - ssim_value
            elif loss_type in ['g_ssim', 'ssim_4', 'ssim_4_g', 'ms_ssim_4_g']:
                # Advanced SSIM variants using SSIM library
                if self.ssim_library is None:
                    logger.warning(f"SSIM library not available for {loss_type}, falling back to regular SSIM")
                    ssim_value = tf.image.ssim(original, reconstructed, max_val=1.0)
                    return 1.0 - ssim_value
                
                # Convert tensors to numpy for SSIM library
                original_np = original.numpy() if hasattr(original, 'numpy') else original
                reconstructed_np = reconstructed.numpy() if hasattr(reconstructed, 'numpy') else reconstructed
                
                errors = []
                batch_size = original_np.shape[0] if len(original_np.shape) > 3 else 1
                
                if len(original_np.shape) == 3:
                    original_np = np.expand_dims(original_np, 0)
                    reconstructed_np = np.expand_dims(reconstructed_np, 0)
                
                for i in range(batch_size):
                    true_img = original_np[i]
                    pred_img = reconstructed_np[i]
                    
                    # Convert to grayscale if needed
                    if len(true_img.shape) == 3 and true_img.shape[-1] > 1:
                        true_img = np.mean(true_img, axis=2)
                        pred_img = np.mean(pred_img, axis=2)
                    elif len(true_img.shape) == 3:
                        true_img = true_img[:,:,0]
                        pred_img = pred_img[:,:,0]
                    
                    # Calculate appropriate SSIM variant
                    if loss_type == 'g_ssim':
                        ssim_score = self.ssim_library.calculate_g_ssim(true_img, pred_img)
                    elif loss_type == 'ssim_4':
                        ssim_score = self.ssim_library.calculate_4_ssim(true_img, pred_img)
                    elif loss_type == 'ssim_4_g':
                        ssim_score = self.ssim_library.calculate_4_g_ssim(true_img, pred_img)
                    elif loss_type == 'ms_ssim_4_g':
                        ssim_score = self.ssim_library.calculate_4_ms_g_ssim(true_img, pred_img)
                    else:
                        ssim_score = self.ssim_library.calculate_ssim(true_img, pred_img)
                    
                    errors.append(1.0 - ssim_score)
                
                return tf.constant(errors, dtype=tf.float32)
            elif loss_type == 'mae':
                # Compute MAE (mean absolute error)
                mae = tf.abs(original - reconstructed)
                return tf.reduce_mean(mae, axis=list(range(1, len(mae.shape))))
            else:  # Default to MSE
                # Compute squared differences
                squared_diff = tf.square(original - reconstructed)
                
                # Reduce to per-sample error (mean across all dimensions except batch)
                if len(squared_diff.shape) > 1:
                    return tf.reduce_mean(squared_diff, axis=list(range(1, len(squared_diff.shape))))
                else:
                    return squared_diff  # Already reduced
        
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Error computing reconstruction error: {e}")
            logger.error(f"Shapes - original: {original.shape}, reconstructed: {reconstructed.shape}")
            # Return a default error value
            return tf.ones(tf.shape(original)[0], dtype=tf.float32)

    def fit_error_distribution(self, train_data=None, val_data=None):
        """Compute reconstruction errors and set threshold.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            
        Returns:
            float: The computed reconstruction threshold
        """
        if self.task != "classification":
            logger.info("Skipping threshold computation for non-classification task")
            return None

        if self.threshold_data not in ['train', 'val', 'both']:
            raise ValueError("threshold_data must be 'train', 'val', or 'both'")

        errors = []
        
        # Compute training errors if needed
        if self.threshold_data in ['train', 'both']:
            if train_data is None:
                raise ValueError("Training data required for threshold computation")
            
            train_errors = []
            for batch in train_data:
                if isinstance(batch, tuple):
                    batch = batch[0]  # Extract inputs if tuple (inputs, labels)
                reconstructed = self(batch, training=False)
                batch_errors = self.compute_reconstruction_error(batch, reconstructed)
                train_errors.extend(batch_errors.numpy())
            
            self.training_errors = train_errors
            errors.extend(train_errors)
            logger.info(f"Computed reconstruction errors for {len(train_errors)} training samples")

        # Compute validation errors if needed
        if self.threshold_data in ['val', 'both']:
            if val_data is None:
                raise ValueError("Validation data required for threshold computation")
            
            val_errors = []
            for batch in val_data:
                if isinstance(batch, tuple):
                    batch = batch[0]  # Extract inputs if tuple (inputs, labels)
                reconstructed = self(batch, training=False)
                batch_errors = self.compute_reconstruction_error(batch, reconstructed)
                val_errors.extend(batch_errors.numpy())
            
            self.validation_errors = val_errors
            errors.extend(val_errors)
            logger.info(f"Computed reconstruction errors for {len(val_errors)} validation samples")

        if not errors:
            raise ValueError("No data available for threshold computation")

        # Compute threshold
        if self.threshold_data in ['val', 'train']:
            self.reconstruction_threshold = np.percentile(errors, self.threshold_percentile)
        elif self.threshold_data == 'both':
            self.reconstruction_threshold = (min(val_errors) + max(train_errors)) / 2.0
        else:
            raise ValueError("Invalid threshold data")
        logger.info(f"Set reconstruction threshold at {self.reconstruction_threshold:.4f} "
                f"({self.threshold_percentile}th percentile) using {self.threshold_data} data")
        
        return self.reconstruction_threshold

    def predict_anomalies(self, data, return_scores=False):
        """Predict whether samples are anomalous based on reconstruction error.
        
        Args:
            data: Input data to evaluate
            return_scores: If True, return reconstruction errors along with predictions
            
        Returns:
            Tuple of (predictions, scores) if return_scores=True, else just predictions
            predictions: 1 for anomaly, 0 for normal
        """
        if self.reconstruction_threshold is None:
            raise ValueError("Must call fit_error_distribution before predicting")
        
        # Handle batch as tuple (inputs, labels)
        if isinstance(data, tuple):
            data = data[0]
        
        reconstructed = self(data, training=False)
        errors = self.compute_reconstruction_error(data, reconstructed)
        predictions = tf.cast(errors > self.reconstruction_threshold, tf.int32)
        
        if return_scores:
            return predictions, errors
        return predictions

    def get_anomaly_metrics(self, data, labels=None):
        """Get metrics for anomaly detection performance.
        
        Args:
            data: Input data to evaluate
            labels: Optional ground truth labels (1 for anomaly, 0 for normal)
            
        Returns:
            dict: Dictionary of metrics
        """
        predictions, scores = self.predict_anomalies(data, return_scores=True)
        metrics = {
            "mean_reconstruction_error": tf.reduce_mean(scores).numpy(),
            "std_reconstruction_error": tf.math.reduce_std(scores).numpy(),
            "anomaly_rate": tf.reduce_mean(tf.cast(predictions, tf.float32)).numpy()
        }
        
        if labels is not None:
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary')
            auc_score = roc_auc_score(labels, scores)
            
            metrics.update({
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_score": auc_score
            })
        
        return metrics

    def get_classification_metrics(self, data):
        """Get classification metrics for a dataset.
        
        Args:
            data: Dataset to evaluate
            
        Returns:
            dict: Dictionary containing classification metrics
        """
        predictions = []
        true_labels = []
        
        for batch in data:
            if isinstance(batch, tuple):
                x, y = batch
                pred = self.model(x, training=False)
                predictions.append(pred)
                true_labels.append(y)
        
        y_pred = np.concatenate(predictions, axis=0)
        y_true = np.concatenate(true_labels, axis=0)
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        precision, recall, f1, _ = precision_recall_fscore_support(
            np.argmax(y_true, axis=1), 
            np.argmax(y_pred, axis=1), 
            average='weighted'
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def get_summary(self, save_path=None) -> str:
        """Get a detailed summary of the model architecture.
        
        Args:
            save_path: Optional path to save the summary to a file
            
        Returns:
            Formatted summary string
        """
        summary_list = []
        self.summary(print_fn=lambda x: summary_list.append(x))
        summary_str = "\n".join(summary_list)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary_str)
        
        return summary_str

    def get_config(self):
        """Return model configuration.
        
        Returns:
            dict: Model configuration
        """
        config = super(ConvAutoencoder, self).get_config()
        config.update({
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "encoder_filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "final_activation": self.final_activation,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "use_tied_weights": self.use_tied_weights,
            "use_orthogonal_weights": self.use_orthogonal_weights,
            "use_unit_length_weights": self.use_unit_length_weights
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.
        
        Args:
            config (dict): Model configuration
            
        Returns:
            ConvAutoencoder: Model instance
        """
        # Convert the config to the format expected by __init__
        init_config = {
            "input_shape": config["input_shape"],
            "latent_dim": config["latent_dim"],
            "encoder_filters": config["encoder_filters"],
            "kernel_size": config["kernel_size"],
            "strides": config["strides"],
            "padding": config["padding"],
            "activation": config["activation"],
            "final_activation": config["final_activation"],
            "use_batch_norm": config.get("use_batch_norm", True),
            "dropout_rate": config.get("dropout_rate", 0.1),
            "use_tied_weights": config.get("use_tied_weights", False),
            "use_orthogonal_weights": config.get("use_orthogonal_weights", False),
            "use_unit_length_weights": config.get("use_unit_length_weights", False)
        }
        return cls(init_config)

    @property
    def anomaly_threshold(self):
        """Get the anomaly threshold (alias for reconstruction_threshold)."""
        return self.reconstruction_threshold
        
    @anomaly_threshold.setter
    def anomaly_threshold(self, value):
        """Set the anomaly threshold (and sync with reconstruction_threshold)."""
        self.reconstruction_threshold = value

# Example usage
if __name__ == '__main__':
    config = {
        "input_shape": [125, 750, 3],
        "latent_dim": 16,
        "filters": [32, 64],
        "kernel_size": 3,
        "strides": 2,
        "activation": "relu",
        "use_tied_weights": True,
        "use_orthogonal_weights": True,
        "use_unit_length_weights": False
    }
    model = ConvAutoencoder(config)
    model.build_graph().summary()
    
    print("\nTied weights:", model.use_tied_weights)
    print("Orthogonal weights:", model.use_orthogonal_weights)
    print("Unit length weights:", model.use_unit_length_weights)
