"""Reusable attention mechanisms for neural networks.

This module provides various attention mechanisms that can be used in different
model architectures, including U-Net variants and autoencoders.

Classes:
    AttentionGate: Attention gate for focusing on relevant features
    SelfAttentionGate: Self-attention variant of the attention gate
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from loguru import logger


class AttentionGate(layers.Layer):
    """Attention gate implementation.
    
    This implements the attention mechanism shown in Attention U-Net and other papers:
    - Input feature map x and gating signal g go through 1x1 convolutions
    - The results are combined with element-wise addition
    - ReLU is applied, followed by 1x1 convolution and sigmoid
    - The output attention coefficients are multiplied with input x
    
    This can be used for:
    1. Standard attention between encoder and decoder features in U-Net
    2. Self-attention when the same feature map is used as both x and g
    """
    
    def __init__(self, inter_filters, use_bias=True, name=None, **kwargs):
        """Initialize attention gate layer.
        
        Args:
            inter_filters: Number of filters for intermediate representations
            use_bias: Whether to use bias in convolutions
            name: Name for the layer
            **kwargs: Additional arguments
        """
        super().__init__(name=name, **kwargs)
        self.inter_filters = inter_filters
        self.use_bias = use_bias
        
    def build(self, input_shape):
        """Build the layer with input shape information.
        
        Args:
            input_shape: Shape of the inputs, either:
                - List of shapes for the inputs [x_shape, g_shape] for standard attention
                - Single shape for self-attention
        """
        # Check if self-attention or standard attention
        self.is_self_attention = not isinstance(input_shape, list)
        
        # Input feature map convolution (W_x)
        self.Wx = layers.Conv2D(
            self.inter_filters, 
            1, 
            padding='same',
            use_bias=self.use_bias, 
            name='attention_wx'
        )
        
        # Gating signal convolution (W_g)
        self.Wg = layers.Conv2D(
            self.inter_filters, 
            1, 
            padding='same',
            use_bias=self.use_bias, 
            name='attention_wg'
        )
        
        # Intermediate activation
        self.relu = layers.Activation('relu', name='attention_relu')
        
        # Output projection (psi)
        self.psi = layers.Conv2D(1, 1, padding='same', name='attention_psi')
        
        # Sigmoid activation
        self.sigmoid = layers.Activation('sigmoid', name='attention_sigmoid')
        
        # Element-wise operations
        self.add = layers.Add(name='attention_add')
        self.multiply = layers.Multiply(name='attention_multiply')
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Either:
                - Tuple of (x, g) where:
                    x: Input feature map
                    g: Gating signal
                - Single input x for self-attention
            
        Returns:
            Tuple of (attention_map, attended_features) where:
                attention_map: The computed attention coefficients
                attended_features: Input features weighted by attention
        """
        if self.is_self_attention:
            # Self-attention: use the same input for both x and g
            x = inputs
            g = inputs
        else:
            # Standard attention: separate inputs for x and g
            x, g = inputs
        
        # Apply convolutions to input feature map and gating signal
        Wx_x = self.Wx(x)
        Wg_g = self.Wg(g)
        
        # Add the results
        combined = self.add([Wx_x, Wg_g])
        
        # Apply ReLU
        activated = self.relu(combined)
        
        # Apply final convolution
        psi_activated = self.psi(activated)
        
        # Apply sigmoid to get attention coefficients
        attention_map = self.sigmoid(psi_activated)
        
        # Apply attention to input features
        attended_features = self.multiply([x, attention_map])
        
        return attention_map, attended_features
    
    def compute_output_shape(self, input_shape):
        """Compute output shape.
        
        Args:
            input_shape: Input shape(s)
            
        Returns:
            Output shape(s)
        """
        if self.is_self_attention:
            # For self-attention, input_shape is a single shape
            x_shape = input_shape
        else:
            # For standard attention, input_shape is a list of shapes
            x_shape = input_shape[0]
        
        # Attention map has same spatial dimensions as input but 1 channel
        attention_shape = (x_shape[0], x_shape[1], x_shape[2], 1)
        
        # Attended features have same shape as input
        attended_shape = x_shape
        
        return attention_shape, attended_shape
    
    def get_config(self):
        """Get layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'inter_filters': self.inter_filters,
            'use_bias': self.use_bias
        })
        return config


class SelfAttentionGate(AttentionGate):
    """Self-attention gate variant.
    
    This is a convenience class that automatically sets up self-attention
    by using the same input for both the feature map and gating signal.
    """
    
    def call(self, inputs, training=None):
        """Forward pass for self-attention.
        
        Args:
            inputs: Input feature map
            
        Returns:
            Tuple of (attention_map, attended_features)
        """
        # For self-attention, use the same input for both x and g
        return super().call(inputs, training=training)


class ResidualBlock(layers.Layer):
    """Residual block implementation.
    
    This implements a standard residual block with:
    - Two convolutional layers with batch normalization and ReLU
    - Optional downsampling via stride
    - Skip connection that matches the output dimensions
    """
    
    def __init__(self, filters, kernel_size=3, strides=1, name=None, **kwargs):
        """Initialize residual block.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of convolutional kernels
            strides: Stride for convolutions (for downsampling)
            name: Name for the layer
            **kwargs: Additional arguments
        """
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
    def build(self, input_shape):
        """Build the layer with input shape information.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # First convolutional block
        self.conv1 = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            name='residual_conv1'
        )
        self.bn1 = layers.BatchNormalization(name='residual_bn1')
        self.relu1 = layers.Activation('relu', name='residual_relu1')
        
        # Second convolutional block
        self.conv2 = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=1,
            padding='same',
            use_bias=False,
            name='residual_conv2'
        )
        self.bn2 = layers.BatchNormalization(name='residual_bn2')
        
        # Skip connection (if needed)
        if self.strides > 1 or input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv2D(
                self.filters,
                1,
                strides=self.strides,
                padding='same',
                use_bias=False,
                name='residual_skip_conv'
            )
            self.skip_bn = layers.BatchNormalization(name='residual_skip_bn')
        else:
            self.skip_conv = None
            self.skip_bn = None
        
        # Final activation
        self.relu_out = layers.Activation('relu', name='residual_relu_out')
        
        # Add operation
        self.add = layers.Add(name='residual_add')
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip, training=training)
        else:
            skip = inputs
        
        # Add skip connection
        output = self.add([x, skip])
        
        # Final activation
        output = self.relu_out(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        """Compute output shape.
        
        Args:
            input_shape: Input shape
            
        Returns:
            Output shape
        """
        # Calculate spatial dimensions after convolutions
        h, w = input_shape[1], input_shape[2]
        
        # Apply stride
        h = h // self.strides if h is not None else None
        w = w // self.strides if w is not None else None
        
        return (input_shape[0], h, w, self.filters)
    
    def get_config(self):
        """Get layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides
        })
        return config 
