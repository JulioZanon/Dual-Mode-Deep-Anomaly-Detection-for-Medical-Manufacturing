"""
Model Saver Utility

Provides a standardized way to save Keras models with automatic format handling.
Defaults to .keras format to avoid HDF5 group conflicts.
"""

import tensorflow as tf
from pathlib import Path
from typing import Optional, Union
from loguru import logger


class ModelSaver:
    """Standardized model saver that handles format conversion and cleanup."""
    
    def __init__(
        self,
        default_format: str = "keras",
        auto_convert_h5: bool = True,
        cleanup_existing: bool = True
    ):
        """
        Initialize the model saver.
        
        Args:
            default_format: Default format to use ("keras" or "h5")
            auto_convert_h5: If True, automatically convert .h5 paths to .keras
            cleanup_existing: If True, remove existing files before saving
        """
        self.default_format = default_format
        self.auto_convert_h5 = auto_convert_h5
        self.cleanup_existing = cleanup_existing
    
    def save(
        self,
        model: tf.keras.Model,
        save_path: Union[str, Path],
        overwrite: bool = True
    ) -> Path:
        """
        Save a Keras model to the specified path.
        
        Args:
            model: The Keras model to save
            save_path: Path where to save the model
            overwrite: If True, overwrite existing file
            
        Returns:
            Path object of the actual save location (may differ if format converted)
        """
        save_path = Path(save_path)
        
        # Handle format conversion if enabled
        if self.auto_convert_h5 and save_path.suffix == '.h5':
            save_path = save_path.with_suffix('.keras')
            logger.info(f"Auto-converting save path to .keras format: {save_path}")
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cleanup existing file if requested
        if self.cleanup_existing and save_path.exists():
            if overwrite:
                logger.info(f"Removing existing model file: {save_path}")
                save_path.unlink()
            else:
                raise FileExistsError(
                    f"Model file already exists: {save_path}. "
                    f"Set overwrite=True to replace it."
                )
        
        # Save the model
        logger.info(f"Saving model to: {save_path}")
        try:
            model.save(str(save_path))
            logger.info(f"Model saved successfully to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        return save_path
    
    @staticmethod
    def save_weights(
        model: tf.keras.Model,
        weights_path: Union[str, Path],
        overwrite: bool = True
    ) -> Path:
        """
        Save only model weights (lighter, faster, but requires architecture to reload).
        
        Args:
            model: The Keras model
            weights_path: Path where to save weights
            overwrite: If True, overwrite existing file
            
        Returns:
            Path object of the actual save location
        """
        weights_path = Path(weights_path)
        
        # Ensure directory exists
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cleanup existing file if requested
        if weights_path.exists():
            if overwrite:
                logger.info(f"Removing existing weights file: {weights_path}")
                weights_path.unlink()
            else:
                raise FileExistsError(
                    f"Weights file already exists: {weights_path}. "
                    f"Set overwrite=True to replace it."
                )
        
        # Save weights
        logger.info(f"Saving model weights to: {weights_path}")
        try:
            model.save_weights(str(weights_path))
            logger.info(f"Model weights saved successfully to: {weights_path}")
        except Exception as e:
            logger.error(f"Failed to save model weights: {e}")
            raise
        
        return weights_path


# Convenience function for easy usage
def save_model(
    model: tf.keras.Model,
    save_path: Union[str, Path],
    overwrite: bool = True,
    auto_convert_h5: bool = True
) -> Path:
    """
    Convenience function to save a model using default settings.
    
    Args:
        model: The Keras model to save
        save_path: Path where to save the model
        overwrite: If True, overwrite existing file
        auto_convert_h5: If True, automatically convert .h5 paths to .keras
        
    Returns:
        Path object of the actual save location
    """
    saver = ModelSaver(auto_convert_h5=auto_convert_h5)
    return saver.save(model, save_path, overwrite=overwrite)

