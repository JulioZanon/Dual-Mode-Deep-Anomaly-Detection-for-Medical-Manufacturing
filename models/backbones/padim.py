"""PaDiM-based anomaly detection classifier.

This implementation follows the original PaDiM paper closely:
1. Features are extracted from multiple layers of the backbone
2. Features are aligned and concatenated
3. Statistical modeling is done on CPU to avoid memory issues
4. Anomalies are detected using Mahalanobis distance
"""

import tensorflow as tf
import numpy as np
from loguru import logger
import json
import mlflow
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm

class AnomalyMapGeneratorTF:
    """TensorFlow implementation of PaDiM anomaly map generator."""
    
    def __init__(self, image_size, sigma=4, use_scipy_gaussian=False, normalize=False):
        self.image_size = tuple(image_size)
        self.sigma = sigma
        self.use_scipy_gaussian = use_scipy_gaussian
        self.normalize = normalize
        self.min_score = None
        self.max_score = None

    @staticmethod
    def embedding_concat(l1, l2):
        """Original patch-based feature concatenation from PaDiM paper.
        
        Args:
            l1: First feature tensor [B, H1, W1, C1]
            l2: Second feature tensor [B, H2, W2, C2]
            
        Returns:
            Concatenated features [B, H2, W2, C1+C2]
        """
        bs, h1, w1, c1 = l1.shape
        _, h2, w2, c2 = l2.shape
        
        # Calculate scale factor
        s = int(h1 / h2)
        
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

    @staticmethod
    def compute_distance(embedding, mean, inv_covariance):
        """Compute Mahalanobis distance for anomaly detection.
        
        Args:
            embedding: Feature embeddings [B, H, W, C]
            mean: Feature mean [C]
            inv_covariance: Inverse covariance matrix [C, C]
            
        Returns:
            Mahalanobis distance map [B, H, W]
        """
        B, H, W, C = embedding.shape
        embedding_flat = tf.reshape(embedding, [B, H * W, C])  # [B, HW, C]
        mean = tf.reshape(mean, [1, 1, C])  # [1, 1, C]
        delta = embedding_flat - mean  # [B, HW, C]
        left = tf.matmul(delta, inv_covariance)  # [B, HW, C]
        mahalanobis = tf.reduce_sum(left * delta, axis=-1)  # [B, HW]
        mahalanobis = tf.sqrt(tf.maximum(mahalanobis, 0.0))
        mahalanobis = tf.reshape(mahalanobis, [B, H, W])  # [B, H, W]
        return mahalanobis

    def up_sample(self, distance):
        """Upsample distance map to target size.
        
        Args:
            distance: Distance map [B, H, W]
            
        Returns:
            Upsampled distance map [B, H_out, W_out, 1]
        """
        # Add channel dimension
        distance = tf.expand_dims(distance, -1)  # [B, H, W, 1]
        
        # Simple bilinear upsampling
        upsampled = tf.image.resize(distance, self.image_size, method='bilinear')
        
        return upsampled

    def smooth_anomaly_map(self, anomaly_map):
        """Smooth anomaly map using Gaussian filtering."""
        if self.use_scipy_gaussian:
            # Use scipy's gaussian_filter for exact parity with original
            return tf.py_function(
                lambda x: gaussian_filter(x.numpy(), sigma=self.sigma),
                [anomaly_map],
                tf.float32
            )
        else:
            # TensorFlow implementation
            kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1
            x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
            gauss = tf.exp(-(x ** 2) / (2.0 * self.sigma ** 2))
            gauss = gauss / tf.reduce_sum(gauss)
            gauss_kernel = tf.tensordot(gauss, gauss, axes=0)
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            gauss_kernel = tf.tile(gauss_kernel, [1, 1, 1, 1])
            return tf.nn.depthwise_conv2d(anomaly_map, gauss_kernel, strides=[1, 1, 1, 1], padding='SAME')

    def normalize_scores(self, scores):
        """Normalize scores to [0, 1] range using min-max normalization."""
        if self.min_score is None or self.max_score is None:
            self.min_score = tf.reduce_min(scores)
            self.max_score = tf.reduce_max(scores)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        normalized = (scores - self.min_score) / (self.max_score - self.min_score + epsilon)
        return normalized

    def compute_anomaly_map(self, embedding, mean, inv_covariance):
        """Compute anomaly map with all processing steps."""
        # 1. Compute Mahalanobis distance
        score_map = self.compute_distance(embedding, mean, inv_covariance)
        
        # 2. Upsample to target size
        upsampled = self.up_sample(score_map)
        
        # 3. Apply Gaussian smoothing
        smoothed = self.smooth_anomaly_map(upsampled)
        
        # 4. Normalize scores to [0, 1] range
        if self.normalize:
            normalized = self.normalize_scores(smoothed)
            return normalized
        else:
            return smoothed

    def __call__(self, embedding, mean, inv_covariance):
        return self.compute_anomaly_map(embedding, mean, inv_covariance)


class PaDiMClassifier:
    """PaDiM (Patch Distribution Modeling) classifier for anomaly detection."""
    
    def __init__(self, input_shape, n_features=550, backbone='resnet18', use_attention=True, 
                 train_backbone=False, threshold=None, threshold_metadata=None,
                 use_cpu_for_covariance=True, feature_processing=None, use_scipy_gaussian=False,
                 per_position_stats=False, normalize_anomaly_map=False, model_path=None, random_seed=42, **kwargs):
        """Initialize PaDiM classifier.
        
        Args:
            input_shape: Input shape (height, width, channels)
            n_features: Number of features to select from embeddings
            backbone: Backbone model ('resnet18', 'resnet50', 'resattunet', 'attunet')
            use_attention: Whether to use attention mechanism
            train_backbone: Whether to train the backbone
            threshold: Anomaly threshold
            threshold_metadata: Metadata for threshold selection
            use_cpu_for_covariance: Whether to use CPU for covariance calculations
            feature_processing: Dictionary with feature processing options
            use_scipy_gaussian: Whether to use scipy's gaussian_filter for smoothing
            per_position_stats: If True, use per-spatial-position mean/covariance
            normalize_anomaly_map: If True, normalize anomaly map to [0, 1]
            model_path: Path to save/load model weights
            random_seed: Seed for random feature selection
        """
        # Store configuration
        self.feature_processing = feature_processing or {}
        self.use_cpu_for_covariance = use_cpu_for_covariance
        self.use_scipy_gaussian = use_scipy_gaussian
        self.per_position_stats = per_position_stats
        self.normalize_anomaly_map = normalize_anomaly_map
        self.model_path = model_path
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        self.input_shape = input_shape
        self.n_features = n_features
        self.backbone = backbone
        self.use_attention = use_attention
        self.train_backbone = train_backbone
        self.threshold = threshold
        self.threshold_metadata = threshold_metadata
        
        # Initialize feature vectors
        self.feature_vectors = None
        self.feature_means = None
        self.feature_covs = None
        
        # Initialize random indices for feature selection
        self.random_indices = None
        
        # Initialize feature extractor
        self._init_feature_extractor()
        
    def _init_feature_extractor(self):
        """Initialize feature extractor backbone."""
        if self.backbone == 'resnet18':
            # Use ResNet18V2 as backbone
            base_model = tf.keras.applications.ResNet18V2(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
            
            # Get intermediate layers for feature extraction
            layer1 = base_model.get_layer('conv2_block2_out').output  # (H/4, W/4, 64)
            layer2 = base_model.get_layer('conv3_block2_out').output  # (H/8, W/8, 128)
            layer3 = base_model.get_layer('conv4_block2_out').output  # (H/16, W/16, 256)
            
            # Create model with selected outputs
            self.feature_extractor = tf.keras.Model(
                inputs=base_model.input,
                outputs=[layer1, layer2, layer3],
                name='feature_extractor'
            )
            
            # Store feature layers
            self.feature_layers = [layer1, layer2, layer3]
            
        elif self.backbone in ['attunet', 'resattunet']:
            # Import the appropriate model
            if self.backbone == 'attunet':
                from models.backbones.att_unet import AttentionUNet
                self.backbone_model = AttentionUNet(
                    input_shape=self.input_shape
                )
            else:  # resattunet
                from models.backbones.res_att_unet import ResidualAttentionUNet
                self.backbone_model = ResidualAttentionUNet(
                    input_shape=self.input_shape,
                    use_attention=self.use_attention,
                    use_convae_structure=True,
                    save_convae_features=True,
                    latent_dim=32,
                    use_residual=False,
                    use_skip_connections=False
                )
            
            # Build the model with a dummy input
            dummy_input = tf.zeros((1,) + tuple(self.input_shape))
            _ = self.backbone_model(dummy_input)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Set trainable status
        if hasattr(self, 'feature_extractor'):
            self.feature_extractor.trainable = self.train_backbone
            for layer in self.feature_extractor.layers:
                layer.trainable = self.train_backbone
        elif hasattr(self, 'backbone_model'):
            self.backbone_model.trainable = self.train_backbone
            for layer in self.backbone_model.layers:
                layer.trainable = self.train_backbone
        
        logger.info(f"Initialized PaDiM with {self.backbone} backbone")
    
    def _extract_features(self, inputs, training=False):
        """Extract and align features from the backbone, then concatenate and select random channels."""
        # Extract features from backbone
        if self.backbone in ['resnet18', 'resnet50']:
            features = self.feature_extractor(inputs, training=training)
        elif self.backbone in ['attunet', 'resattunet']:
            if hasattr(self.backbone_model, 'get_convae_features') and hasattr(self.backbone_model, 'use_convae_structure') and self.backbone_model.use_convae_structure:
                features = self.backbone_model.get_convae_features(inputs)
                if not features:
                    logger.warning("ConvAE features not available, falling back to standard encoder features")
                    features = self.backbone_model.get_encoder_features(inputs)
            elif hasattr(self.backbone_model, 'get_encoder_features'):
                features = self.backbone_model.get_encoder_features(inputs)
            else:
                outputs = self.backbone_model(inputs)
                if isinstance(outputs, dict) and 'encoder_features' in outputs:
                    features = outputs['encoder_features']
                elif isinstance(outputs, list):
                    features = outputs
                else:
                    features = [outputs]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        # Use original patch-based concatenation
        concat_features = features[-1]
        
        # Concatenate with larger feature maps in reverse order
        for feat in reversed(features[:-1]):
            concat_features = AnomalyMapGeneratorTF.embedding_concat(feat, concat_features)

        # On first call, select random indices for channel selection
        if self.random_indices is None:
            total_channels = concat_features.shape[-1]
            if self.n_features > total_channels:
                logger.warning(f"Requested {self.n_features} features but only {total_channels} are available. Using all available features.")
                self.n_features = total_channels
                self.random_indices = tf.range(total_channels)
            else:
                indices = np.random.permutation(total_channels)[:self.n_features]
                self.random_indices = tf.convert_to_tensor(indices, dtype=tf.int32)
                logger.info(f"Selected {self.n_features} random features from {total_channels} total features")

        # Select the random channels
        selected_features = tf.gather(concat_features, self.random_indices, axis=-1)

        return selected_features
    
    def fit(self, dataset):
        """Fit the model to the training data.
        
        Args:
            dataset: tf.data.Dataset containing training images
        """
        logger.info("Starting PaDiM model fitting...")
        
        # Initialize lists to store features
        all_features = []
        
        # First pass: collect all features
        logger.info("First pass: collecting features...")
        for batch_idx, (x, _) in enumerate(dataset):
            features = self._extract_features(x, training=False)
            all_features.append(features)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx} batches")
        
        # Convert to numpy array
        all_features = np.concatenate(all_features, axis=0)
        logger.info(f"Collected features shape: {all_features.shape}")
        
        # Calculate statistics
        if self.per_position_stats:
            logger.info("Computing per-position statistics...")
            H, W, C = all_features.shape[1:]
            
            means = np.zeros((H, W, C), dtype=np.float32)
            inv_covs = np.zeros((H, W, C, C), dtype=np.float32)
            
            # Compute statistics for each position
            for i in range(H):
                for j in range(W):
                    patch_vecs = all_features[:, i, j, :]
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
            N, H, W, C = all_features.shape
            features_flat = all_features.reshape(N * H * W, -1)
            
            # Compute mean
            means = np.mean(features_flat, axis=0)
            
            # Compute covariance with regularization
            diff = features_flat - means
            cov = np.cov(diff, rowvar=False)
            reg = 1e-5 * np.eye(C)
            cov_reg = cov + reg
            
            # Compute inverse using Cholesky decomposition
            try:
                L = np.linalg.cholesky(cov_reg)
                L_inv = np.linalg.inv(L)
                inv_cov = np.dot(L_inv.T, L_inv)
            except np.linalg.LinAlgError:
                logger.warning("Cholesky failed, using regular inverse")
                try:
                    inv_cov = np.linalg.inv(cov_reg)
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.pinv(cov_reg)
            
            self.feature_means = tf.convert_to_tensor(means, dtype=tf.float32)
            self.feature_covs = tf.convert_to_tensor(inv_cov, dtype=tf.float32)
        
        # Second pass: compute distances and threshold
        logger.info("Second pass: computing distances and threshold...")
        all_distances = []
        
        for batch_idx, (x, _) in enumerate(dataset):
            features = self._extract_features(x, training=False)
            
            # Compute Mahalanobis distances
            if self.per_position_stats:
                features_np = features.numpy()
                means_np = self.feature_means.numpy()
                inv_covs_np = self.feature_covs.numpy()
                diff = features_np - means_np
                mahalanobis = np.einsum('bhwc,hwcd,bhwd->bhw', diff, inv_covs_np, diff)
                mahalanobis = np.sqrt(np.maximum(mahalanobis, 0.0))
            else:
                features_np = features.numpy()
                B, H, W, C = features_np.shape
                features_flat = features_np.reshape(B * H * W, -1)
                means = self.feature_means.numpy()
                inv_covs = self.feature_covs.numpy()
                
                diff = features_flat - means
                distances = np.sqrt(np.sum(np.dot(diff, inv_covs) * diff, axis=1))
                distances = distances.reshape(B, H, W)
                mahalanobis = np.maximum(distances, 0.0)
            
            # Get image-level scores (max of pixel-level distances)
            image_scores = np.max(mahalanobis, axis=(1, 2))
            all_distances.extend(image_scores)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx} batches")
        
        # Convert to numpy array
        all_distances = np.array(all_distances)
        
        # Calculate threshold using quantile
        self.threshold = np.percentile(all_distances, 95)
        
        # Log threshold information
        logger.info(f"Threshold statistics:")
        logger.info(f"Mean Mahalanobis distance: {np.mean(all_distances):.4f}")
        logger.info(f"Std Mahalanobis distance: {np.std(all_distances):.4f}")
        logger.info(f"Threshold (95th quantile): {self.threshold:.4f}")
        logger.info(f"Based on {len(all_distances)} samples")
        
        # Initialize anomaly map generator
        self.anomaly_map_generator = AnomalyMapGeneratorTF(
            image_size=self.input_shape[:2],
            use_scipy_gaussian=self.use_scipy_gaussian,
            normalize=self.normalize_anomaly_map
        )
        
        # Save model weights and statistics
        if self.model_path:
            self.save_weights(self.model_path)
            self.save_statistics()
        
        logger.success("PaDiM model fitting completed!")
    
    def call(self, inputs, training=False):
        """Forward pass of the model."""
        # Extract features with training=False to ensure backbone is frozen
        features = self._extract_features(inputs, training=False)
        
        # If not trained, return zero scores
        if self.feature_means is None or self.feature_covs is None:
            logger.warning("Model not trained, returning zero anomaly scores")
            return {
                'reconstruction': inputs,
                'anomaly_scores': tf.zeros([tf.shape(inputs)[0]], dtype=tf.float32)
            }
        
        if self.per_position_stats:
            # Per-position Mahalanobis
            B, H, W, C = features.shape
            anomaly_map = np.zeros((B, H, W), dtype=np.float32)
            means = self.feature_means.numpy()
            inv_covs = self.feature_covs.numpy()
            
            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        x = features[b, i, j, :].numpy()
                        mu = means[i, j, :]
                        inv_cov = inv_covs[i, j, :, :]
                        diff = x - mu
                        m = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
                        anomaly_map[b, i, j] = m
            
            anomaly_map = tf.convert_to_tensor(anomaly_map, dtype=tf.float32)
            upsampled = self.anomaly_map_generator.up_sample(anomaly_map)
            smoothed = self.anomaly_map_generator.smooth_anomaly_map(upsampled)
            
            if self.normalize_anomaly_map:
                normalized = self.anomaly_map_generator.normalize_scores(smoothed)
            else:
                normalized = smoothed
            
            if normalized.shape[-1] == 1:
                normalized = tf.squeeze(normalized, axis=-1)
            
            anomaly_scores = tf.reduce_max(normalized, axis=[1, 2])
            
            return {
                'reconstruction': inputs,
                'anomaly_scores': anomaly_scores,
                'anomaly_map': normalized
            }
        else:
            # Use the anomaly map generator for inference
            anomaly_map = self.anomaly_map_generator(
                embedding=features,
                mean=self.feature_means,
                inv_covariance=self.feature_covs
            )
            
            if anomaly_map.shape[-1] == 1:
                anomaly_map = tf.squeeze(anomaly_map, axis=-1)
            
            # Get scores as max over spatial map
            anomaly_scores = tf.reduce_max(anomaly_map, axis=[1, 2])
            
            return {
                'reconstruction': inputs,
                'anomaly_scores': anomaly_scores,
                'anomaly_map': anomaly_map
            }
    
    def predict_anomalies(self, data, return_scores=False):
        """Predict whether samples are anomalous."""
        # Extract inputs if data is a tuple
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        
        # Get anomaly scores
        if isinstance(x, tf.data.Dataset):
            all_scores = []
            for batch in x:
                if isinstance(batch, tuple):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                outputs = self(batch_x, training=False)
                all_scores.append(outputs['anomaly_scores'])
            scores = tf.concat(all_scores, axis=0)
        else:
            outputs = self(x, training=False)
            scores = outputs['anomaly_scores']
        
        # Get threshold
        threshold = self.threshold if self.threshold is not None else 0.5
        
        # Make predictions
        predictions = tf.cast(scores > threshold, tf.int32)
        
        if return_scores:
            return predictions, scores
        return predictions
    
    def save_weights(self, filepath, **kwargs):
        """Save model weights and additional attributes."""
        filepath_str = str(filepath)
        
        # Create directory if it doesn't exist
        save_dir = Path(filepath_str).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config_path = filepath_str + '.config.json'
        config = self.get_config()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            return obj
        
        config = convert_numpy(config)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save threshold information
        threshold_path = filepath_str + '.threshold.json'
        threshold_data = {
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'threshold_metadata': self.threshold_metadata
        }
        threshold_data = convert_numpy(threshold_data)
        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        # Save feature statistics
        stats_path = filepath_str + '.stats.npz'
        if self.feature_means is not None and self.feature_covs is not None:
            np.savez(
                stats_path,
                feature_means=self.feature_means.numpy(),
                feature_covs=self.feature_covs.numpy(),
                random_indices=self.random_indices.numpy() if self.random_indices is not None else None
            )
            logger.info(f"Saved feature statistics to {stats_path}")
        
        logger.info(f"Saved model weights to {filepath_str}")
        logger.info(f"Saved configuration to {config_path}")
        logger.info(f"Saved threshold data to {threshold_path}")
    
    def load_weights(self, filepath, **kwargs):
        """Load model weights and additional attributes."""
        filepath_str = str(filepath)
        
        # Load model configuration
        config_path = filepath_str + '.config.json'
        if tf.io.gfile.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                config = None
        else:
            logger.warning(f"No configuration file found at {config_path}")
            config = None
        
        # Load threshold information
        threshold_path = filepath_str + '.threshold.json'
        if tf.io.gfile.exists(threshold_path):
            try:
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                self.threshold = threshold_data['threshold']
                self.threshold_metadata = threshold_data['threshold_metadata']
                logger.info(f"Loaded threshold: {self.threshold}")
            except Exception as e:
                logger.error(f"Error loading threshold from {threshold_path}: {e}")
                self.threshold = 0.5
                self.threshold_metadata = None
        else:
            logger.warning(f"No threshold file found at {threshold_path}")
            self.threshold = 0.5
            self.threshold_metadata = None
        
        # Load feature statistics
        stats_path = filepath_str + '.stats.npz'
        if tf.io.gfile.exists(stats_path):
            try:
                stats = np.load(stats_path)
                self.feature_means = tf.convert_to_tensor(stats['feature_means'], dtype=tf.float32)
                self.feature_covs = tf.convert_to_tensor(stats['feature_covs'], dtype=tf.float32)
                if 'random_indices' in stats:
                    self.random_indices = tf.convert_to_tensor(stats['random_indices'], dtype=tf.int32)
                logger.info(f"Loaded feature statistics from {stats_path}")
                
                # Initialize anomaly map generator after loading statistics
                self.anomaly_map_generator = AnomalyMapGeneratorTF(
                    image_size=self.input_shape[:2],
                    use_scipy_gaussian=self.use_scipy_gaussian,
                    normalize=self.normalize_anomaly_map
                )
                logger.info("Initialized anomaly map generator")
            except Exception as e:
                logger.error(f"Error loading feature statistics from {stats_path}: {e}")
                self.feature_means = None
                self.feature_covs = None
                self.random_indices = None
        else:
            logger.warning(f"No feature statistics found at {stats_path}")
            self.feature_means = None
            self.feature_covs = None
            self.random_indices = None
    
    def get_config(self):
        """Get model configuration."""
        return {
            'input_shape': self.input_shape,
            'n_features': self.n_features,
            'backbone': self.backbone,
            'use_attention': self.use_attention,
            'train_backbone': self.train_backbone,
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'threshold_metadata': self.threshold_metadata,
            'use_cpu_for_covariance': self.use_cpu_for_covariance,
            'feature_processing': self.feature_processing,
            'use_scipy_gaussian': self.use_scipy_gaussian,
            'per_position_stats': self.per_position_stats,
            'normalize_anomaly_map': self.normalize_anomaly_map
        }
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
    
    def save_statistics(self):
        """Save feature statistics to file."""
        if self.model_path is None:
            logger.warning("No model path specified, skipping statistics save")
            return
            
        stats_path = str(self.model_path) + '.stats.npz'
        try:
            np.savez(
                stats_path,
                feature_means=self.feature_means.numpy(),
                feature_covs=self.feature_covs.numpy(),
                random_indices=self.random_indices.numpy() if self.random_indices is not None else None
            )
            logger.info(f"Saved feature statistics to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving feature statistics: {e}")
    
    def load_backbone_weights(self, weights_path):
        """Load weights for the backbone model."""
        if hasattr(self, 'backbone_model'):
            try:
                self.backbone_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                logger.info(f"Loaded backbone weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load backbone weights from {weights_path}: {e}")
        elif hasattr(self, 'feature_extractor'):
            try:
                self.feature_extractor.load_weights(weights_path, by_name=True, skip_mismatch=True)
                logger.info(f"Loaded feature extractor weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load feature extractor weights from {weights_path}: {e}")
        else:
            logger.warning("No backbone model found to load weights") 
