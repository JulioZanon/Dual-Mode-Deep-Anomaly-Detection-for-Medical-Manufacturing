
import tensorflow as tf
import json
from pathlib import Path
from models.backbones.unified_res_att_unet import UnifiedResidualAttentionUNet
from models.backbones.vanilla_convae import ConvAutoencoder
from models.backbones.skip_convae import SkipConvAutoencoder
from loguru import logger

# Register custom loss functions for Keras serialization
# These functions are registered to match the names used when models were saved
@tf.keras.saving.register_keras_serializable(name="ssim_loss")
def ssim_loss(y_true, y_pred):
    """SSIM-based loss function registered for Keras serialization."""
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(1.0 - ssim)

@tf.keras.saving.register_keras_serializable(name="ssim_mae_loss")
def ssim_mae_loss(y_true, y_pred):
    """Combined SSIM and MAE loss function registered for Keras serialization."""
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss_val = tf.reduce_mean(1.0 - ssim)
    mae_loss_val = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return 0.5 * ssim_loss_val + 0.5 * mae_loss_val

@tf.keras.saving.register_keras_serializable(name="g_ssim_loss")
def g_ssim_loss(y_true, y_pred):
    """Gradient SSIM loss function registered for Keras serialization.
    Note: This is a fallback version. Full g_ssim requires SSIM library at runtime."""
    # Fallback to regular SSIM if g_ssim is not available
    return ssim_loss(y_true, y_pred)

@tf.keras.saving.register_keras_serializable(name="ssim_4_loss")
def ssim_4_loss(y_true, y_pred):
    """4-neighborhood SSIM loss function registered for Keras serialization.
    Note: This is a fallback version. Full ssim_4 requires SSIM library at runtime."""
    # Fallback to regular SSIM if ssim_4 is not available
    return ssim_loss(y_true, y_pred)

@tf.keras.saving.register_keras_serializable(name="ssim_4_g_loss")
def ssim_4_g_loss(y_true, y_pred):
    """4-neighborhood gradient SSIM loss function registered for Keras serialization.
    Note: This is a fallback version. Full ssim_4_g requires SSIM library at runtime."""
    # Fallback to regular SSIM if ssim_4_g is not available
    return ssim_loss(y_true, y_pred)

@tf.keras.saving.register_keras_serializable(name="ms_ssim_4_g_loss")
def ms_ssim_4_g_loss(y_true, y_pred):
    """Multi-scale 4-neighborhood gradient SSIM loss function registered for Keras serialization.
    Note: This is a fallback version. Full ms_ssim_4_g requires SSIM library at runtime."""
    # Fallback to regular SSIM if ms_ssim_4_g is not available
    return ssim_loss(y_true, y_pred)

def load_model_proper(model_path: str, compile_model: bool = True) -> tf.keras.Model:
    """Load model - supports .keras (direct load) and .h5 (rebuild from JSON + weights)."""
    logger.info(f"Loading model from {model_path}...")
    
    model_path_obj = Path(model_path)
    
    # If .keras format, load directly (no HDF5 issues, full model saved)
    if model_path_obj.suffix == '.keras':
        logger.info("Detected .keras format - loading model directly...")
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=compile_model,
                custom_objects={
                    'UnifiedResidualAttentionUNet': UnifiedResidualAttentionUNet,
                    'ConvAutoencoder': ConvAutoencoder,
                    'SkipConvAutoencoder': SkipConvAutoencoder,
                    # Registered custom loss functions
                    'ssim_loss': ssim_loss,
                    'ssim_mae_loss': ssim_mae_loss,
                    'g_ssim_loss': g_ssim_loss,
                    'ssim_4_loss': ssim_4_loss,
                    'ssim_4_g_loss': ssim_4_g_loss,
                    'ms_ssim_4_g_loss': ms_ssim_4_g_loss,
                }
            )
            logger.info(f"Model loaded successfully from .keras file")
            
            # Print model structure
            logger.info("="*80)
            logger.info("MODEL STRUCTURE:")
            logger.info("="*80)
            model.summary(print_fn=lambda x: logger.info(x))
            logger.info("="*80)
            
            # Verify weights are loaded (check that they're not all zeros or random)
            total_weight_norm = 0.0
            weight_count = 0
            for layer in model.layers:
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    for w in layer.get_weights():
                        total_weight_norm += float(tf.norm(w).numpy())
                        weight_count += 1
            
            if weight_count > 0:
                avg_norm = total_weight_norm / weight_count
                logger.info(f"Weight verification: {weight_count} weight tensors, average norm: {avg_norm:.6f}")
                if avg_norm < 1e-6:
                    logger.warning(f"⚠️ WARNING: Average weight norm is very small ({avg_norm:.6e}). Model may have uninitialized weights!")
                elif avg_norm > 1000:
                    logger.warning(f"⚠️ WARNING: Average weight norm is very large ({avg_norm:.6e}). May indicate numerical issues.")
            
            # Quick inference test with dummy input to verify model works
            try:
                dummy_input = tf.zeros((1,) + tuple(model.input_shape[1:]))
                dummy_output = model(dummy_input, training=False)
                output_norm = float(tf.norm(dummy_output).numpy())
                logger.info(f"Model inference test: input shape {dummy_input.shape}, output shape {dummy_output.shape}, output norm: {output_norm:.6f}")
                if output_norm < 1e-6:
                    logger.warning(f"⚠️ WARNING: Model output norm is very small ({output_norm:.6e}). Model may not be functioning correctly!")
            except Exception as e:
                logger.warning(f"Could not run inference test: {e}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load .keras model: {e}")
            raise
    
    # For .h5 format, use JSON config to rebuild model and load weights
    logger.info("Detected .h5 format - will rebuild from JSON config and load weights...")
    
    # Load the JSON config
    json_path = model_path_obj.with_suffix('.json')
    if not json_path.exists():
        raise FileNotFoundError(f"JSON config not found: {json_path}")
    
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Determine model architecture from config
    architecture = config['model_config'].get('architecture', 'unified_res_att_unet')
    
    if architecture == 'convae':
        logger.info(f"Loading ConvAutoencoder model")
        logger.info(f"Using config: encoder_filters={config['model_config']['filters']}, latent_dim={config['model_config']['latent_dim']}, dropout={config['model_config']['dropout_rate']}")
        
        # Build config for ConvAutoencoder
        convae_config = {
            "input_shape": config['model_config']['input_shape'],
            "encoder_filters": config['model_config'].get('filters', [32, 64, 128]),
            "kernel_size": config['model_config'].get('kernel_size', 3),
            "strides": config['model_config'].get('strides', 2),
            "padding": config['model_config'].get('padding', 'same'),
            "activation": config['model_config'].get('activation', 'relu'),
            "final_activation": config['model_config'].get('final_activation', 'sigmoid'),
            "use_batch_norm": config['model_config'].get('use_batch_norm', True),
            "dropout_rate": config['model_config']['dropout_rate'],
            "latent_dim": config['model_config']['latent_dim'],
            "use_tied_weights": config['model_config'].get('use_tied_weights', False),
            "use_orthogonal_weights": config['model_config'].get('use_orthogonal_weights', False),
            "use_unit_length_weights": config['model_config'].get('use_unit_length_weights', False),
            "loss": config['model_config'].get('loss_function', 'mse')
        }
        
        # Create ConvAutoencoder model
        model = ConvAutoencoder(convae_config)
        
    elif architecture == 'skip_convae':
        logger.info(f"Loading SkipConvAutoencoder model")
        logger.info(f"Using config: encoder_filters={config['model_config']['filters']}, latent_dim={config['model_config']['latent_dim']}, dropout={config['model_config']['dropout_rate']}")
        
        # Build config for SkipConvAutoencoder
        skip_convae_config = {
            "input_shape": config['model_config']['input_shape'],
            "encoder_filters": config['model_config'].get('filters', [32, 64, 128]),
            "kernel_size": config['model_config'].get('kernel_size', 3),
            "strides": config['model_config'].get('strides', 2),
            "padding": config['model_config'].get('padding', 'same'),
            "activation": config['model_config'].get('activation', 'relu'),
            "final_activation": config['model_config'].get('final_activation', 'sigmoid'),
            "use_batch_norm": config['model_config'].get('use_batch_norm', True),
            "dropout_rate": config['model_config'].get('dropout_rate', 0.1),
            "latent_dim": config['model_config']['latent_dim'],
            "use_skip_connections": config['model_config'].get('use_skip_connections', True),
            "use_residual_blocks": config['model_config'].get('use_residual_blocks', True),
            "use_attention": config['model_config'].get('use_attention', False),
            "kernel_initializer": config['model_config'].get('kernel_initializer', 'he_normal')
        }
        
        # Create SkipConvAutoencoder model
        model = SkipConvAutoencoder(skip_convae_config)
        
    else:
        logger.info(f"Loading UnifiedResidualAttentionUNet model")
        logger.info(f"Using config: filters={config['model_config']['filters']}, latent_dim={config['model_config']['latent_dim']}, dropout={config['model_config']['dropout_rate']}")
        
        # Create model with exact config
        model = UnifiedResidualAttentionUNet(
            input_shape=tuple(config['model_config']['input_shape']),
            filters=config['model_config']['filters'],
            use_attention=config['model_config']['use_attention'],
            use_residual=config['model_config']['use_residual'],
            use_skip_connections=config['model_config']['use_skip_connections'],
            output_channels=config['model_config']['input_shape'][-1],
            loss_function=config['model_config']['loss_function'],
            dropout_rate=config['model_config']['dropout_rate'],
            latent_dim=config['model_config']['latent_dim']
        )
    
    # Build the model
    dummy_input = tf.zeros((1,) + tuple(config['model_config']['input_shape']))
    _ = model(dummy_input)
    
    logger.info(f"Model created with {len(model.layers)} layers, {model.count_params()} parameters")
    
    # Print model structure
    logger.info("="*80)
    logger.info("MODEL STRUCTURE (before loading weights):")
    logger.info("="*80)
    model.summary(print_fn=lambda x: logger.info(x))
    logger.info("="*80)
    
    # Load weights
    model.load_weights(model_path)
    logger.info("Weights loaded successfully")
    
    # Print model structure again after loading weights
    logger.info("="*80)
    logger.info("MODEL STRUCTURE (after loading weights):")
    logger.info("="*80)
    model.summary(print_fn=lambda x: logger.info(x))
    logger.info("="*80)
    
    if compile_model:
        try:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info("Model compiled")
        except Exception as e:
            logger.warning(f"Could not compile: {e}")
    
    return model

# Test the loader (only runs when script is executed directly, not on import)
if __name__ == "__main__":
    model = load_model_proper('data/models/vanilla_convae_pouches/split_10.h5')
    logger.info("Model loaded successfully!")
    logger.info(f"Type: {type(model).__name__}")
    logger.info(f"Parameters: {model.count_params():,}")
    logger.info(f"Input shape: {model.input_shape}")

    # Test inference
    test_input = tf.random.uniform((1, 256, 256, 1), 0, 1)
    output = model.predict(test_input, verbose=0)
    logger.info(f"Test output std: {tf.math.reduce_std(output):.4f}")
