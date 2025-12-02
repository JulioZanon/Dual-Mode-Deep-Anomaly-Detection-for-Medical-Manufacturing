"""
Fixed pouches loader that avoids tf.py_function shape inference issues.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union
from loguru import logger

class FixedPouchLoader:
    """Fixed pouches loader that avoids shape inference issues."""
    
    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        train_split: float = 0.8,
        mode: str = 'reconstruction',
        use_single_channel: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.train_split = train_split
        self.mode = mode
        self.use_single_channel = use_single_channel
        
    def _load_image_direct(self, img_path: str) -> tf.Tensor:
        """Load image directly with robust error handling for intermittent failures."""
        try:
            # Read file
            img_data = tf.io.read_file(img_path)
            
            # Determine target channels based on configuration
            # If use_single_channel is True, load as grayscale (1 channel)
            # If use_single_channel is False, load as RGB (3 channels)
            if self.use_single_channel:
                # Load directly as grayscale
                target_channels = 1
            else:
                # Load as RGB (3 channels)
                target_channels = 3
            
            # Decode based on extension with robust error handling
            img = self._decode_image_safely(img_data, img_path, target_channels)
            
            # Ensure image has the expected shape
            if len(img.shape) != 3:
                logger.warning(f"Unexpected image shape {img.shape} for {img_path}")
                return None
            
            # Resize
            img = tf.image.resize(img, list(self.img_size))
            
            # Ensure proper shape after resize
            if len(img.shape) != 3 or img.shape[-1] != target_channels:
                logger.warning(f"Unexpected shape after resize {img.shape} for {img_path}")
                return None
            
            # Normalize to [0, 1]
            img = tf.cast(img, tf.float32) / 255.0
            
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return None to indicate this file should be skipped
            return None
    
    def _decode_image_safely(self, img_data: tf.Tensor, img_path: str, target_channels: int) -> tf.Tensor:
        """Safely decode image with robust error handling for intermittent failures."""
        
        # Check file extension to determine decoding method
        if isinstance(img_path, tf.Tensor):
            # For tensor inputs, we need to infer the type
            # Try PNG first, then fall back to other formats
            try:
                img = tf.image.decode_png(img_data, channels=target_channels)
                return img
            except tf.errors.InvalidArgumentError:
                # PNG failed, try other formats
                try:
                    img = tf.image.decode_jpeg(img_data, channels=target_channels)
                    return img
                except tf.errors.InvalidArgumentError:
                    try:
                        img = tf.image.decode_bmp(img_data, channels=target_channels)
                        return img
                    except tf.errors.InvalidArgumentError:
                        # All decoders failed, this is a real corruption
                        raise ValueError(f"All image decoders failed for {img_path}")
        else:
            # For string inputs, check extension
            img_path_lower = img_path.lower()
            
            if img_path_lower.endswith('.png'):
                # PNG files - try multiple times for intermittent failures
                max_retries = 5  # Increased retries for intermittent failures
                for attempt in range(max_retries):
                    try:
                        img = tf.image.decode_png(img_data, channels=target_channels)
                        return img
                    except tf.errors.InvalidArgumentError as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"PNG decode attempt {attempt + 1} failed for {img_path}, retrying... (attempt {attempt + 1}/{max_retries})")
                            # Progressive delay to avoid race conditions
                            import time
                            time.sleep(0.02 * (attempt + 1))  # Progressive delay
                        else:
                            logger.error(f"PNG decode failed after {max_retries} attempts for {img_path}: {e}")
                            # Don't fall back to other decoders for PNG files
                            raise ValueError(f"PNG decode failed after {max_retries} attempts: {e}")
                    
            elif img_path_lower.endswith('.jpg') or img_path_lower.endswith('.jpeg'):
                # JPEG files
                try:
                    img = tf.image.decode_jpeg(img_data, channels=target_channels)
                    return img
                except tf.errors.InvalidArgumentError as e:
                    logger.error(f"JPEG decode failed for {img_path}: {e}")
                    raise ValueError(f"JPEG decode failed: {e}")
                    
            elif img_path_lower.endswith('.bmp'):
                # BMP files
                try:
                    img = tf.image.decode_bmp(img_data, channels=target_channels)
                    return img
                except tf.errors.InvalidArgumentError as e:
                    logger.error(f"BMP decode failed for {img_path}: {e}")
                    raise ValueError(f"BMP decode failed: {e}")
                    
            else:
                # Unknown extension - try PNG first, then others
                try:
                    img = tf.image.decode_png(img_data, channels=target_channels)
                    return img
                except tf.errors.InvalidArgumentError:
                    try:
                        img = tf.image.decode_jpeg(img_data, channels=target_channels)
                        return img
                    except tf.errors.InvalidArgumentError:
                        try:
                            img = tf.image.decode_bmp(img_data, channels=target_channels)
                            return img
                        except tf.errors.InvalidArgumentError:
                            raise ValueError(f"All image decoders failed for {img_path}")
    
    def _get_file_lists(self) -> Tuple[List[str], List[str]]:
        """Get lists of good and bad image files."""
        good_dir = self.data_dir / "good"
        bad_dir = self.data_dir / "bad"
        
        good_files = []
        bad_files = []
        
        if good_dir.exists():
            good_files = [
                str(f) for f in good_dir.glob("*.png") 
            ] + [
                str(f) for f in good_dir.glob("*.bmp")
            ]
        
        if bad_dir.exists():
            bad_files = [
                str(f) for f in bad_dir.glob("*.png")
            ] + [
                str(f) for f in bad_dir.glob("*.bmp")
            ]
        
        logger.info(f"Found {len(good_files)} good files, {len(bad_files)} bad files")
        return good_files, bad_files
    
    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
        """Create training and validation datasets."""
        good_files, bad_files = self._get_file_lists()
        
        if not good_files:
            raise ValueError(f"No good files found in {self.data_dir}")
        
        # For reconstruction mode, we only use good files
        if self.mode == 'reconstruction':
            files = good_files
        else:
            # For other modes, use both good and bad
            files = good_files + bad_files
        
        # Shuffle files
        np.random.shuffle(files)
        
        # Split into train/val
        split_idx = int(len(files) * self.train_split)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        logger.info(f"Split: {len(train_files)} train, {len(val_files)} validation")
        
        # Create datasets
        train_dataset = self._create_dataset_from_files(train_files)
        val_dataset = self._create_dataset_from_files(val_files) if val_files else None
        
        return train_dataset, val_dataset, train_files
    
    def _create_dataset_from_files(self, file_paths: List[str]) -> tf.data.Dataset:
        """Create dataset from file paths with pre-filtering of problematic files."""
        if not file_paths:
            return None
        
        # Pre-filter files to remove problematic ones before dataset creation
        logger.info(f"Pre-filtering {len(file_paths)} files to identify problematic ones...")
        valid_files = []
        problematic_files = []
        
        for file_path in file_paths:
            try:
                # Test if the file can be loaded successfully
                test_img = self._load_image_direct(file_path)
                if test_img is not None:
                    # Additional validation: check if the image has the expected shape and values
                    expected_channels = 1 if self.use_single_channel else 3
                    expected_shape = (*self.img_size, expected_channels)
                    if (test_img.shape == expected_shape and 
                        tf.reduce_any(tf.not_equal(test_img, 0.0)) and
                        tf.reduce_all(tf.not_equal(test_img, -1.0))):
                        valid_files.append(file_path)
                    else:
                        problematic_files.append(file_path)
                        logger.warning(f"File {file_path} has invalid shape or values: {test_img.shape}, expected: {expected_shape}")
                else:
                    problematic_files.append(file_path)
                    logger.warning(f"File {file_path} failed validation and will be skipped")
            except Exception as e:
                problematic_files.append(file_path)
                logger.warning(f"File {file_path} caused error during validation: {e}")
        
        logger.info(f"Pre-filtering complete: {len(valid_files)} valid files, {len(problematic_files)} problematic files")
        
        if not valid_files:
            raise ValueError("No valid files found after pre-filtering!")
        
        # Create dataset only from valid files
        dataset = tf.data.Dataset.from_tensor_slices(valid_files)
        
        # Load images with runtime error handling for intermittent failures
        def safe_load_with_fallback(file_path):
            try:
                # Try to load the image normally
                img = self._load_image_direct(file_path)
                if img is not None:
                    return img
                else:
                    # If loading failed, return a black image (will be filtered out)
                    return tf.zeros((*self.img_size, 1), dtype=tf.float32)
            except Exception as e:
                # If any error occurs during TensorFlow execution, return black image
                return tf.zeros((*self.img_size, 1), dtype=tf.float32)
        
        # Use tf.py_function to handle Python exceptions during TensorFlow execution
        dataset = dataset.map(
            lambda file_path: tf.py_function(
                safe_load_with_fallback,
                [file_path],
                tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Restore shape information that tf.py_function loses
        dataset = dataset.map(
            lambda img: tf.ensure_shape(img, (*self.img_size, 1)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Filter out black images (all zeros) - these indicate failed loads
        # This will remove any images that failed to load and returned black
        dataset = dataset.filter(
            lambda img: tf.reduce_any(tf.not_equal(img, 0.0))
        )
        
        # Also filter out images with all -1 values (another failure indicator)
        dataset = dataset.filter(
            lambda img: tf.reduce_all(tf.not_equal(img, -1.0))
        )
        
        # Final shape validation to ensure all images have correct dimensions
        dataset = dataset.filter(
            lambda img: tf.equal(tf.shape(img)[0], self.img_size[0]) and 
                       tf.equal(tf.shape(img)[1], self.img_size[1]) and
                       tf.equal(tf.shape(img)[2], 1)
        )
        
        # Additional safety check: ensure images have reasonable values (not all identical)
        dataset = dataset.filter(
            lambda img: tf.math.reduce_std(img) > 0.001  # Images should have some variation
        )
        
        # Add error handling for the entire pipeline
        dataset = dataset.map(
            lambda img: tf.cond(
                tf.reduce_any(tf.math.is_nan(img)),
                lambda: tf.zeros((*self.img_size, 1), dtype=tf.float32),  # Return black image if NaN
                lambda: img
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # For autoencoder, input = output
        dataset = dataset.map(
            lambda img: (img, img),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset 
