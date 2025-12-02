#!/usr/bin/env python3
"""
Generic Data Loader Module

This module provides a unified interface for loading different types of datasets
including pouches and MVTec anomaly detection datasets.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from loguru import logger


class GenericDataLoader:
    """
    Generic data loader that supports different dataset types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.config = config
        self.dataset_type = config.get('dataset_type', 'pouches').lower()
        self.image_size = config.get('image_size', [64, 64])
        self.normal_class = config.get('normal_class', 'good')
        self.anomaly_class = config.get('anomaly_class', 'bad')
        
        # MVTec-specific configuration
        self.mvtec_normal_classes = config.get('mvtec_normal_classes', ['good'])
        self.mvtec_anomaly_classes = config.get('mvtec_anomaly_classes', [])
        
        logger.info(f"Initialized GenericDataLoader for {self.dataset_type} dataset")
        logger.info(f"Normal class: {self.normal_class}, Anomaly class: {self.anomaly_class}")
    
    def load_dataset(self, data_dir: str, split_name: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load dataset from directory structure.
        
        Args:
            data_dir: Path to the dataset directory
            split_name: Name of the split (e.g., 'train', 'test')
            
        Returns:
            Tuple of (images, filenames, labels)
        """
        logger.info(f"Loading {split_name} dataset from {data_dir}...")
        logger.info(f"Dataset type: {self.dataset_type}")
        
        data_path = Path(data_dir)
        images = []
        filenames = []
        labels = []
        
        if self.dataset_type == 'mvtec':
            images, filenames, labels = self._load_mvtec_dataset(data_path, split_name)
        else:  # pouches or default
            images, filenames, labels = self._load_pouches_dataset(data_path, split_name)
        
        images = np.array(images)
        logger.info(f"Loaded {len(images)} {split_name} images")
        logger.info(f"Shape: {images.shape}")
        logger.info(f"Classes: {set(labels)}")
        
        return images, filenames, labels
    
    def _load_pouches_dataset(self, data_path: Path, split_name: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """
        Load pouches dataset from directory structure.
        
        Args:
            data_path: Path to the dataset directory
            split_name: Name of the split
            
        Returns:
            Tuple of (images, filenames, labels)
        """
        images = []
        filenames = []
        labels = []
        
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                logger.info(f"Loading {class_name} images...")
                
                for img_file in class_dir.glob('*.png'):
                    try:
                        img = self._load_and_preprocess_image(img_file)
                        if img is None:
                            continue
                        
                        images.append(img)
                        filenames.append(img_file.name)
                        labels.append(class_name)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {e}")
                        continue
        
        return images, filenames, labels
    
    def _load_mvtec_dataset(self, data_path: Path, split_name: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """
        Load MVTec dataset from directory structure.
        
        MVTec structure:
        - train/good/ (normal samples)
        - test/good/ (normal samples)
        - test/{anomaly_type}/ (anomaly samples)
        
        Args:
            data_path: Path to the dataset directory
            split_name: Name of the split
            
        Returns:
            Tuple of (images, filenames, labels)
        """
        images = []
        filenames = []
        labels = []
        
        if split_name == 'train':
            # Training data only contains normal samples
            normal_dir = data_path / 'good'
            if normal_dir.exists():
                logger.info("Loading normal training samples...")
                for img_file in normal_dir.glob('*.png'):
                    try:
                        img = self._load_and_preprocess_image(img_file)
                        if img is None:
                            continue
                        
                        images.append(img)
                        filenames.append(img_file.name)
                        labels.append(self.normal_class)  # Convert to standard normal class
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {e}")
                        continue
            else:
                logger.warning(f"Normal training directory not found: {normal_dir}")
        
        elif split_name == 'test':
            # Test data contains both normal and anomaly samples
            # Load normal samples
            normal_dir = data_path / 'good'
            if normal_dir.exists():
                logger.info("Loading normal test samples...")
                for img_file in normal_dir.glob('*.png'):
                    try:
                        img = self._load_and_preprocess_image(img_file)
                        if img is None:
                            continue
                        
                        images.append(img)
                        filenames.append(img_file.name)
                        labels.append(self.normal_class)  # Convert to standard normal class
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {e}")
                        continue
            
            # Load anomaly samples from all anomaly subdirectories
            for subdir in data_path.iterdir():
                if subdir.is_dir() and subdir.name != 'good':
                    # This is an anomaly type directory
                    anomaly_type = subdir.name
                    logger.info(f"Loading anomaly samples from {anomaly_type}...")
                    
                    for img_file in subdir.glob('*.png'):
                        try:
                            img = self._load_and_preprocess_image(img_file)
                            if img is None:
                                continue
                            
                            images.append(img)
                            filenames.append(f"{anomaly_type}_{img_file.name}")  # Include anomaly type in filename
                            labels.append(self.anomaly_class)  # Convert all anomaly types to standard anomaly class
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {img_file}: {e}")
                            continue
        
        else:
            logger.warning(f"Unknown split name: {split_name}")
        
        return images, filenames, labels
    
    def _load_and_preprocess_image(self, img_file: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            img_file: Path to the image file
            
        Returns:
            Preprocessed image array or None if loading failed
        """
        try:
            # Load image in grayscale
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize to target size
            img = cv2.resize(img, tuple(self.image_size))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add channel dimension if needed
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            
            return img
            
        except Exception as e:
            logger.warning(f"Error preprocessing image {img_file}: {e}")
            return None
    
    def load_existing_image_pairs(self, images_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], 
                                                                  np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load existing image pairs from the output directory.
        
        Args:
            images_dir: Path to the images directory containing train/test subdirectories
            
        Returns:
            Tuple of (train_images, train_reconstructions, train_filenames, train_labels,
                     test_images, test_reconstructions, test_filenames, test_labels)
        """
        logger.info("Loading existing image pairs...")
        
        # Load train images
        train_images, train_reconstructions, train_filenames, train_labels = self._load_images_from_pairs(
            images_dir / 'train'
        )
        
        # Load test images
        test_images, test_reconstructions, test_filenames, test_labels = self._load_images_from_pairs(
            images_dir / 'test'
        )
        
        logger.info(f"Loaded existing image pairs: Train samples: {len(train_images)} Test samples: {len(test_images)}")
        
        return train_images, train_reconstructions, train_filenames, train_labels, test_images, test_reconstructions, test_filenames, test_labels
    
    def _load_images_from_pairs(self, split_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load images from a specific split directory.
        
        Args:
            split_dir: Path to the split directory (train or test)
            
        Returns:
            Tuple of (images, reconstructions, filenames, labels)
        """
        logger.info(f"Loading {split_dir.name} images from pairs...")
        
        input_dir = split_dir / 'input'
        output_dir = split_dir / 'output'
        
        if not input_dir.exists() or not output_dir.exists():
            raise FileNotFoundError(f"Required directories not found: {split_dir}")
        
        # Get all image files
        input_files = sorted([f for f in input_dir.glob('*.png')])
        output_files = sorted([f for f in output_dir.glob('*.png')])
        
        if len(input_files) != len(output_files):
            raise ValueError(f"Mismatch in number of input ({len(input_files)}) and output ({len(output_files)}) files")
        
        images = []
        reconstructions = []
        filenames = []
        labels = []
        
        for input_file, output_file in zip(input_files, output_files):
            # Load images
            img = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)
            recon = cv2.imread(str(output_file), cv2.IMREAD_GRAYSCALE)
            
            if img is None or recon is None:
                logger.warning(f"Could not load {input_file} or {output_file}")
                continue
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            recon = recon.astype(np.float32) / 255.0
            
            # Add channel dimension if needed
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if len(recon.shape) == 2:
                recon = np.expand_dims(recon, axis=-1)
            
            images.append(img)
            reconstructions.append(recon)
            
            # Extract label from filename (format: label_filename.png)
            filename_parts = input_file.stem.split('_', 1)
            if len(filename_parts) >= 2:
                label = filename_parts[0]
                original_filename = filename_parts[1]
            else:
                label = 'unknown'
                original_filename = input_file.stem
            
            filenames.append(original_filename)
            labels.append(label)
        
        images = np.array(images)
        reconstructions = np.array(reconstructions)
        
        logger.info(f"Loaded {len(images)} {split_dir.name} image pairs")
        logger.info(f"Classes: {set(labels)}")
        
        return images, reconstructions, filenames, labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset configuration.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'dataset_type': self.dataset_type,
            'image_size': self.image_size,
            'normal_class': self.normal_class,
            'anomaly_class': self.anomaly_class,
            'mvtec_normal_classes': self.mvtec_normal_classes,
            'mvtec_anomaly_classes': self.mvtec_anomaly_classes
        }

