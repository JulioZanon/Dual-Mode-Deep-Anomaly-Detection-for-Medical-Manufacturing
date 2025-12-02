#!/usr/bin/env python3
"""
Complete SSIM Library - All variants from Li & Bovik (2010)

This module provides comprehensive implementations of all SSIM variants described in:
"Content-partitioned structural similarity index for image quality assessment"
by Li & Bovik (2010).

Implemented variants:
1. SSIM: Original Structural Similarity Index
2. G-SSIM: Gradient-based SSIM using Sobel operators
3. 4-SSIM: Four-component SSIM with region weighting
4. 4-G-SSIM: Four-component gradient-based SSIM
5. MS-SSIM: Multi-scale SSIM
6. 4-MS-SSIM: Four-component multi-scale SSIM
7. MS-G-SSIM: Multi-scale gradient-based SSIM
8. 4-MS-G-SSIM: Four-component multi-scale gradient-based SSIM
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from loguru import logger
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel_h, sobel_v
from skimage.transform import resize
from scipy import ndimage
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class SSIMLibrary:
    """
    Complete library for all SSIM variants from Li & Bovik (2010).
    
    This class provides implementations of all SSIM variants described in the paper:
    - SSIM: Original Structural Similarity Index
    - G-SSIM: Gradient-based SSIM using Sobel operators
    - 4-SSIM: Four-component SSIM with region weighting
    - 4-G-SSIM: Four-component gradient-based SSIM
    - MS-SSIM: Multi-scale SSIM
    - 4-MS-SSIM: Four-component multi-scale SSIM
    - MS-G-SSIM: Multi-scale gradient-based SSIM
    - 4-MS-G-SSIM: Four-component multi-scale gradient-based SSIM
    """
    
    def __init__(self, 
                 edge_threshold: float = 0.12, 
                 texture_threshold: float = 0.06,
                 scales: List[float] = None,
                 scale_weights: List[float] = None,
                 use_gaussian: bool = False,
                 gaussian_sigma: float = 1.5):
        """
        Initialize the complete SSIM library.
        
        Args:
            edge_threshold: Threshold to distinguish edges from non-edges
            texture_threshold: Threshold to distinguish texture from smooth regions
            scales: List of scales for multi-scale variants (default: [1.0, 0.5, 0.25, 0.125])
            scale_weights: Weights for each scale (default: [0.5, 0.25, 0.125, 0.125])
            use_gaussian: Whether to use Gaussian weighting in SSIM calculations (default: False)
            gaussian_sigma: Standard deviation of Gaussian filter when use_gaussian=True (default: 1.5)
        """
        self.edge_threshold = edge_threshold
        self.texture_threshold = texture_threshold
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        
        # Default scales and weights for multi-scale variants
        if scales is None:
            self.scales = [1.0, 0.5, 0.25, 0.125]
        else:
            self.scales = scales
            
        if scale_weights is None:
            self.scale_weights = [0.5, 0.25, 0.125, 0.125]
        else:
            self.scale_weights = scale_weights
            
        # Ensure weights sum to 1
        self.scale_weights = np.array(self.scale_weights)
        self.scale_weights = self.scale_weights / np.sum(self.scale_weights)
        
        # Four-component weights as per Li & Bovik (2010)
        self.region_weights = {
            'changed_edges': 0.5,
            'preserved_edges': 0.25,
            'texture_regions': 0.125,
            'smooth_regions': 0.125
        }
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is grayscale."""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert RGB to grayscale using standard weights
                return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            elif image.shape[2] == 1:
                return image[:, :, 0]
        return image
    
    def _get_ssim_window_size(self, image_shape: Tuple[int, int]) -> int:
        """Get appropriate window size for SSIM calculation."""
        min_dim = min(image_shape)
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        return max(3, win_size)
    
    def _get_data_range(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Get data range for SSIM calculation."""
        range1 = image1.max() - image1.min()
        range2 = image2.max() - image2.min()
        return max(range1, range2) if max(range1, range2) > 0 else 1.0
    
    def compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude using Sobel operators as described in Chen et al. (2006).
        This is used for G-SSIM calculation.
        
        Args:
            image: Input image (2D array)
            
        Returns:
            Gradient magnitude image
        """
        try:
            # Ensure grayscale
            image = self._ensure_grayscale(image)
            
            # Apply Sobel operators to get horizontal and vertical gradients
            grad_h = sobel_h(image)
            grad_v = sobel_v(image)
            
            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_h**2 + grad_v**2)
            
            return gradient_magnitude
        except Exception as e:
            logger.warning(f"Failed to compute gradient magnitude: {e}")
            return image  # Fallback to original image
    
    def compute_four_component_weights(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute four-component image model weights as described in Li & Bovik (2010).
        Classifies image regions into: Changed Edges, Preserved Edges, Texture, and Smooth regions.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            Dictionary containing weight maps for each component
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            # Compute gradient magnitudes for both images
            grad_orig = self.compute_gradient_magnitude(original)
            grad_recon = self.compute_gradient_magnitude(reconstructed)
            
            # Classify pixels based on gradient magnitudes
            # Edge regions: high gradient in at least one image
            edge_orig = grad_orig > self.edge_threshold
            edge_recon = grad_recon > self.edge_threshold
            
            # Changed edges: edge in one image but not the other
            changed_edges = np.logical_xor(edge_orig, edge_recon)
            
            # Preserved edges: edge in both images
            preserved_edges = np.logical_and(edge_orig, edge_recon)
            
            # Non-edge regions
            non_edge = np.logical_not(np.logical_or(edge_orig, edge_recon))
            
            # Among non-edge regions, distinguish texture from smooth
            # Texture: moderate gradient activity
            avg_grad = (grad_orig + grad_recon) / 2
            texture_regions = np.logical_and(non_edge, avg_grad > self.texture_threshold)
            
            # Smooth: low gradient activity
            smooth_regions = np.logical_and(non_edge, avg_grad <= self.texture_threshold)
            
            return {
                'changed_edges': changed_edges.astype(np.float32),
                'preserved_edges': preserved_edges.astype(np.float32),
                'texture_regions': texture_regions.astype(np.float32),
                'smooth_regions': smooth_regions.astype(np.float32)
            }
        except Exception as e:
            logger.warning(f"Failed to compute four-component weights: {e}")
            # Return uniform weights as fallback
            uniform_weight = np.ones_like(original, dtype=np.float32) * 0.25
            return {
                'changed_edges': uniform_weight,
                'preserved_edges': uniform_weight,
                'texture_regions': uniform_weight,
                'smooth_regions': uniform_weight
            }
    
    def calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate original SSIM (Structural Similarity Index).
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            win_size = self._get_ssim_window_size(original.shape)
            data_range = self._get_data_range(original, reconstructed)
            
            if self.use_gaussian:
                ssim_score = ssim(original, reconstructed, 
                                data_range=data_range, 
                                win_size=win_size,
                                gaussian_weights=True,
                                sigma=self.gaussian_sigma)
            else:
                ssim_score = ssim(original, reconstructed, 
                                data_range=data_range, 
                                win_size=win_size)
            
            return float(ssim_score)
        except Exception as e:
            logger.warning(f"Failed to calculate SSIM: {e}")
            return np.nan
    
    def calculate_g_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate G-SSIM (Gradient-based Structural Similarity Index) as described in Chen et al. (2006)
        and referenced in Li & Bovik (2010). This computes SSIM on gradient images instead of original images.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            G-SSIM value
        """
        try:
            # Compute gradient magnitude images
            grad_orig = self.compute_gradient_magnitude(original)
            grad_recon = self.compute_gradient_magnitude(reconstructed)
            
            # Calculate SSIM on gradient images
            win_size = self._get_ssim_window_size(grad_orig.shape)
            data_range = self._get_data_range(grad_orig, grad_recon)
            
            if self.use_gaussian:
                g_ssim_score = ssim(grad_orig, grad_recon, 
                                  data_range=data_range, 
                                  win_size=win_size,
                                  gaussian_weights=True,
                                  sigma=self.gaussian_sigma)
            else:
                g_ssim_score = ssim(grad_orig, grad_recon, 
                                  data_range=data_range, 
                                  win_size=win_size)
            
            return float(g_ssim_score)
        except Exception as e:
            logger.warning(f"Failed to calculate G-SSIM: {e}")
            return np.nan
    
    def calculate_4_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate 4-SSIM (Four-component SSIM) as described in Li & Bovik (2010).
        This applies four-component region weighting to SSIM scores.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            4-SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            # Get four-component weights
            weights = self.compute_four_component_weights(original, reconstructed)
            
            # Calculate local SSIM scores
            win_size = self._get_ssim_window_size(original.shape)
            data_range = self._get_data_range(original, reconstructed)
            
            # Calculate SSIM with full_output to get the SSIM map
            if self.use_gaussian:
                ssim_score, ssim_map = ssim(original, reconstructed, 
                                          data_range=data_range,
                                          win_size=win_size, 
                                          full=True,
                                          gaussian_weights=True,
                                          sigma=self.gaussian_sigma)
            else:
                ssim_score, ssim_map = ssim(original, reconstructed, 
                                          data_range=data_range,
                                          win_size=win_size, full=True)
            
            # Apply four-component weighting as per Li & Bovik (2010)
            weighted_score = 0.0
            total_weight = 0.0
            
            for region_type, region_mask in weights.items():
                if region_type in self.region_weights:
                    region_weight = self.region_weights[region_type]
                    # Apply region mask to SSIM map
                    masked_scores = ssim_map * region_mask
                    region_contribution = np.sum(masked_scores) * region_weight
                    region_pixels = np.sum(region_mask)
                    
                    if region_pixels > 0:
                        weighted_score += region_contribution
                        total_weight += region_pixels * region_weight
            
            # Normalize by total weight
            if total_weight > 0:
                return float(weighted_score / total_weight)
            else:
                return float(ssim_score)  # Fallback to regular SSIM
        except Exception as e:
            logger.warning(f"Failed to calculate 4-SSIM: {e}")
            return np.nan
    
    def calculate_4_g_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate 4-G-SSIM (Four-component Gradient-based SSIM) as described in Li & Bovik (2010).
        This applies four-component region weighting to G-SSIM scores.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            4-G-SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            # Compute gradient magnitude images
            grad_orig = self.compute_gradient_magnitude(original)
            grad_recon = self.compute_gradient_magnitude(reconstructed)
            
            # Get four-component weights
            weights = self.compute_four_component_weights(original, reconstructed)
            
            # Calculate local G-SSIM scores
            win_size = self._get_ssim_window_size(grad_orig.shape)
            data_range = self._get_data_range(grad_orig, grad_recon)
            
            # Calculate G-SSIM with full_output to get the SSIM map
            if self.use_gaussian:
                g_ssim_score, g_ssim_map = ssim(grad_orig, grad_recon, 
                                               data_range=data_range,
                                               win_size=win_size, 
                                               full=True,
                                               gaussian_weights=True,
                                               sigma=self.gaussian_sigma)
            else:
                g_ssim_score, g_ssim_map = ssim(grad_orig, grad_recon, 
                                               data_range=data_range,
                                               win_size=win_size, full=True)
            
            # Apply four-component weighting as per Li & Bovik (2010)
            weighted_score = 0.0
            total_weight = 0.0
            
            for region_type, region_mask in weights.items():
                if region_type in self.region_weights:
                    region_weight = self.region_weights[region_type]
                    # Apply region mask to G-SSIM map
                    masked_scores = g_ssim_map * region_mask
                    region_contribution = np.sum(masked_scores) * region_weight
                    region_pixels = np.sum(region_mask)
                    
                    if region_pixels > 0:
                        weighted_score += region_contribution
                        total_weight += region_pixels * region_weight
            
            # Normalize by total weight
            if total_weight > 0:
                return float(weighted_score / total_weight)
            else:
                return float(g_ssim_score)  # Fallback to regular G-SSIM
        except Exception as e:
            logger.warning(f"Failed to calculate 4-G-SSIM: {e}")
            return np.nan
    
    def calculate_ms_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate MS-SSIM (Multi-scale SSIM) as described in Wang et al. (2003).
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            MS-SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            ms_ssim_scores = []
            valid_weights = []
            
            for i, scale in enumerate(self.scales):
                if scale == 1.0:
                    # Use original resolution
                    orig_scaled = original
                    recon_scaled = reconstructed
                else:
                    # Resize images to the scale
                    h, w = original.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Ensure minimum size for SSIM calculation
                    min_size_for_ssim = 7
                    
                    if new_h < min_size_for_ssim or new_w < min_size_for_ssim:
                        continue
                    
                    orig_scaled = resize(original, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                    recon_scaled = resize(reconstructed, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                
                # Calculate SSIM at this scale
                try:
                    win_size = self._get_ssim_window_size(orig_scaled.shape)
                    data_range = self._get_data_range(orig_scaled, recon_scaled)
                    
                    if self.use_gaussian:
                        scale_ssim = ssim(orig_scaled, recon_scaled, 
                                        data_range=data_range, 
                                        win_size=win_size,
                                        gaussian_weights=True,
                                        sigma=self.gaussian_sigma)
                    else:
                        scale_ssim = ssim(orig_scaled, recon_scaled, 
                                        data_range=data_range, 
                                        win_size=win_size)
                    
                    if not np.isnan(scale_ssim):
                        ms_ssim_scores.append(scale_ssim)
                        valid_weights.append(self.scale_weights[i])
                except Exception as e:
                    logger.warning(f"Failed to calculate SSIM at scale {scale}: {e}")
                    continue
            
            if not ms_ssim_scores:
                logger.warning("No valid SSIM scores calculated for MS-SSIM")
                return np.nan
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)
            
            # Return weighted average of SSIM scores across scales
            return float(np.average(ms_ssim_scores, weights=valid_weights))
        except Exception as e:
            logger.warning(f"Failed to calculate MS-SSIM: {e}")
            return np.nan
    
    def calculate_4_ms_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate 4-MS-SSIM (Four-component Multi-scale SSIM) as described in Li & Bovik (2010).
        This applies four-component region weighting to multi-scale SSIM scores.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            4-MS-SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            ms_ssim_scores = []
            valid_weights = []
            
            for i, scale in enumerate(self.scales):
                if scale == 1.0:
                    # Use original resolution
                    orig_scaled = original
                    recon_scaled = reconstructed
                else:
                    # Resize images to the scale
                    h, w = original.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Ensure minimum size for SSIM calculation
                    min_size_for_ssim = 7
                    
                    if new_h < min_size_for_ssim or new_w < min_size_for_ssim:
                        continue
                    
                    orig_scaled = resize(original, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                    recon_scaled = resize(reconstructed, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                
                # Calculate 4-SSIM at this scale
                try:
                    scale_4ssim = self.calculate_4_ssim(orig_scaled, recon_scaled)
                    if not np.isnan(scale_4ssim):
                        ms_ssim_scores.append(scale_4ssim)
                        valid_weights.append(self.scale_weights[i])
                except Exception as e:
                    logger.warning(f"Failed to calculate 4-SSIM at scale {scale}: {e}")
                    continue
            
            if not ms_ssim_scores:
                logger.warning("No valid 4-SSIM scores calculated for 4-MS-SSIM")
                return np.nan
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)
            
            # Return weighted average of 4-SSIM scores across scales
            return float(np.average(ms_ssim_scores, weights=valid_weights))
        except Exception as e:
            logger.warning(f"Failed to calculate 4-MS-SSIM: {e}")
            return np.nan
    
    def calculate_ms_g_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate MS-G-SSIM (Multi-scale Gradient-based SSIM) as described in Li & Bovik (2010).
        This applies multi-scale approach to G-SSIM scores.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            MS-G-SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            ms_g_ssim_scores = []
            valid_weights = []
            
            for i, scale in enumerate(self.scales):
                if scale == 1.0:
                    # Use original resolution
                    orig_scaled = original
                    recon_scaled = reconstructed
                else:
                    # Resize images to the scale
                    h, w = original.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Ensure minimum size for SSIM calculation
                    min_size_for_ssim = 7
                    
                    if new_h < min_size_for_ssim or new_w < min_size_for_ssim:
                        continue
                    
                    orig_scaled = resize(original, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                    recon_scaled = resize(reconstructed, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                
                # Calculate G-SSIM at this scale
                try:
                    scale_g_ssim = self.calculate_g_ssim(orig_scaled, recon_scaled)
                    if not np.isnan(scale_g_ssim):
                        ms_g_ssim_scores.append(scale_g_ssim)
                        valid_weights.append(self.scale_weights[i])
                except Exception as e:
                    logger.warning(f"Failed to calculate G-SSIM at scale {scale}: {e}")
                    continue
            
            if not ms_g_ssim_scores:
                logger.warning("No valid G-SSIM scores calculated for MS-G-SSIM")
                return np.nan
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)
            
            # Return weighted average of G-SSIM scores across scales
            return float(np.average(ms_g_ssim_scores, weights=valid_weights))
        except Exception as e:
            logger.warning(f"Failed to calculate MS-G-SSIM: {e}")
            return np.nan
    
    def calculate_4_ms_g_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate 4-MS-G-SSIM (Four-component Multi-scale Gradient-based SSIM) as described in Li & Bovik (2010).
        This applies four-component region weighting to multi-scale G-SSIM scores.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            4-MS-G-SSIM value
        """
        try:
            # Ensure grayscale
            original = self._ensure_grayscale(original)
            reconstructed = self._ensure_grayscale(reconstructed)
            
            ms_g_ssim_scores = []
            valid_weights = []
            
            for i, scale in enumerate(self.scales):
                if scale == 1.0:
                    # Use original resolution
                    orig_scaled = original
                    recon_scaled = reconstructed
                else:
                    # Resize images to the scale
                    h, w = original.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Ensure minimum size for SSIM calculation
                    min_size_for_ssim = 7
                    
                    if new_h < min_size_for_ssim or new_w < min_size_for_ssim:
                        continue
                    
                    orig_scaled = resize(original, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                    recon_scaled = resize(reconstructed, (new_h, new_w), anti_aliasing=True, preserve_range=True)
                
                # Calculate 4-G-SSIM at this scale
                try:
                    scale_4g_ssim = self.calculate_4_g_ssim(orig_scaled, recon_scaled)
                    if not np.isnan(scale_4g_ssim):
                        ms_g_ssim_scores.append(scale_4g_ssim)
                        valid_weights.append(self.scale_weights[i])
                except Exception as e:
                    logger.warning(f"Failed to calculate 4-G-SSIM at scale {scale}: {e}")
                    continue
            
            if not ms_g_ssim_scores:
                logger.warning("No valid 4-G-SSIM scores calculated for 4-MS-G-SSIM")
                return np.nan
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)
            
            # Return weighted average of 4-G-SSIM scores across scales
            return float(np.average(ms_g_ssim_scores, weights=valid_weights))
        except Exception as e:
            logger.warning(f"Failed to calculate 4-MS-G-SSIM: {e}")
            return np.nan
    
    def calculate_all_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Calculate all SSIM variants for comparison.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            Dictionary containing all SSIM metric values
        """
        results = {}
        
        # Original SSIM
        results['ssim'] = self.calculate_ssim(original, reconstructed)
        
        # G-SSIM
        results['g_ssim'] = self.calculate_g_ssim(original, reconstructed)
        
        # 4-SSIM
        results['4_ssim'] = self.calculate_4_ssim(original, reconstructed)
        
        # 4-G-SSIM
        results['4_g_ssim'] = self.calculate_4_g_ssim(original, reconstructed)
        
        # MS-SSIM
        results['ms_ssim'] = self.calculate_ms_ssim(original, reconstructed)
        
        # 4-MS-SSIM
        results['4_ms_ssim'] = self.calculate_4_ms_ssim(original, reconstructed)
        
        # MS-G-SSIM
        results['ms_g_ssim'] = self.calculate_ms_g_ssim(original, reconstructed)
        
        # 4-MS-G-SSIM
        results['4_ms_g_ssim'] = self.calculate_4_ms_g_ssim(original, reconstructed)
        
        return results


# Convenience functions for direct usage
def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray, 
                  edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                  use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_ssim(original, reconstructed)


def calculate_g_ssim(original: np.ndarray, reconstructed: np.ndarray, 
                    edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                    use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate G-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_g_ssim(original, reconstructed)


def calculate_4_ssim(original: np.ndarray, reconstructed: np.ndarray,
                    edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                    use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate 4-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_4_ssim(original, reconstructed)


def calculate_4_g_ssim(original: np.ndarray, reconstructed: np.ndarray,
                      edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                      use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate 4-G-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_4_g_ssim(original, reconstructed)


def calculate_ms_ssim(original: np.ndarray, reconstructed: np.ndarray,
                     edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                     use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate MS-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_ms_ssim(original, reconstructed)


def calculate_4_ms_ssim(original: np.ndarray, reconstructed: np.ndarray,
                       edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                       use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate 4-MS-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_4_ms_ssim(original, reconstructed)


def calculate_ms_g_ssim(original: np.ndarray, reconstructed: np.ndarray,
                       edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                       use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate MS-G-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_ms_g_ssim(original, reconstructed)


def calculate_4_ms_g_ssim(original: np.ndarray, reconstructed: np.ndarray,
                         edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                         use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> float:
    """Convenience function to calculate 4-MS-G-SSIM."""
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_4_ms_g_ssim(original, reconstructed)


def calculate_all_ssim_metrics(original: np.ndarray, reconstructed: np.ndarray,
                              edge_threshold: float = 0.12, texture_threshold: float = 0.06,
                              use_gaussian: bool = False, gaussian_sigma: float = 1.5) -> Dict[str, float]:
    """
    Convenience function to calculate all SSIM metrics.
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        edge_threshold: Threshold to distinguish edges from non-edges
        texture_threshold: Threshold to distinguish texture from smooth regions
        use_gaussian: Whether to use Gaussian weighting in SSIM calculations
        gaussian_sigma: Standard deviation of Gaussian filter when use_gaussian=True
        
    Returns:
        Dictionary containing all SSIM metric values
    """
    ssim_lib = SSIMLibrary(edge_threshold=edge_threshold, texture_threshold=texture_threshold,
                          use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma)
    return ssim_lib.calculate_all_metrics(original, reconstructed)
