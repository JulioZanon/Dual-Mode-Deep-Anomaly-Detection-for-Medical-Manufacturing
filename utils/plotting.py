#!/usr/bin/env python3
"""
Plotting Module

This module handles the creation of various plots for anomaly detection
evaluation including ROC curves, confusion matrices, threshold analysis, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")

class AnomalyDetectionPlotter:
    """
    Handles creation of plots for anomaly detection evaluation.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different plot types
        self.histograms_dir = self.plots_dir / 'histograms'
        self.roc_curves_dir = self.plots_dir / 'roc_curves'
        self.confusion_matrices_dir = self.plots_dir / 'confusion_matrices'
        self.other_dir = self.plots_dir / 'other'
        
        for subdir in [self.histograms_dir, self.roc_curves_dir, self.confusion_matrices_dir, self.other_dir]:
            subdir.mkdir(exist_ok=True)
    
    def _convert_to_binary(self, labels: np.ndarray) -> np.ndarray:
        """Convert string labels to binary format (0 for good/normal, 1 for bad/anomaly)."""
        # Check if labels contain string values by trying to convert first element
        try:
            # Try to convert to int - if it fails, we have strings
            int(labels[0])
            # If successful, labels are numeric
            return labels.astype(int)
        except (ValueError, TypeError):
            # Labels are strings, convert to binary
            binary_labels = np.array([1 if str(label).lower() in ['bad', 'defective', '1', 'true'] else 0 
                                    for label in labels])
            return binary_labels
        
    def create_all_plots(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                        thresholds: Dict[str, Dict[str, float]], 
                        results: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Create all evaluation plots.
        
        Args:
            train_df: Training data DataFrame
            test_df: Test data DataFrame
            thresholds: Calculated thresholds
            results: Evaluation results
        """
        print("📊 Creating evaluation plots...")
        
        # Create different types of plots
        self._plot_metric_distributions(train_df, test_df)
        self._plot_individual_histograms(train_df, test_df, thresholds)  # Pass thresholds
        self._plot_roc_curves(test_df, results)
        self._plot_confusion_matrices(test_df, results)
        self._plot_threshold_analysis(train_df, test_df, thresholds)
        self._plot_performance_comparison(results)
        self._plot_feature_analysis(train_df, test_df)
        
        print(f"✅ All plots saved to {self.plots_dir}")
    
    def _plot_metric_distributions(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Plot distributions of metrics for good vs bad samples."""
        print("   Creating metric distribution plots...")
        
        # Get metric columns
        metric_columns = [col for col in train_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        # Create subplots
        n_metrics = len(metric_columns)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle different subplot configurations
        if n_metrics == 1:
            # Single metric - axes is a single Axes object
            axes_array = [axes]
        elif n_rows == 1:
            # Single row - axes is a 1D array
            axes_array = axes.reshape(1, -1)
        else:
            # Multiple rows and columns - axes is a 2D array
            axes_array = axes
        
        for i, metric in enumerate(metric_columns):
            row = i // n_cols
            col = i % n_cols
            
            # Get the correct axes object
            if n_metrics == 1:
                ax = axes_array[0]
            elif n_rows == 1:
                ax = axes_array[0, col]
            else:
                ax = axes_array[row, col]
            
            # Plot distributions (handle both string and binary labels)
            train_good = train_df[train_df['label'].isin([0, 'good', 'Good'])][metric].values
            train_bad = train_df[train_df['label'].isin([1, 'bad', 'Bad'])][metric].values
            test_good = test_df[test_df['label'].isin([0, 'good', 'Good'])][metric].values
            test_bad = test_df[test_df['label'].isin([1, 'bad', 'Bad'])][metric].values
            
            # Remove NaN values
            train_good = train_good[~np.isnan(train_good)]
            train_bad = train_bad[~np.isnan(train_bad)]
            test_good = test_good[~np.isnan(test_good)]
            test_bad = test_bad[~np.isnan(test_bad)]
            
            # Plot histograms only if we have data
            if len(train_good) > 0:
                ax.hist(train_good, bins=30, alpha=0.7, label='Train Good', density=True, color='blue')
            if len(train_bad) > 0:
                ax.hist(train_bad, bins=30, alpha=0.7, label='Train Bad', density=True, color='red')
            if len(test_good) > 0:
                ax.hist(test_good, bins=30, alpha=0.5, label='Test Good', density=True, color='lightblue')
            if len(test_bad) > 0:
                ax.hist(test_bad, bins=30, alpha=0.5, label='Test Bad', density=True, color='lightcoral')
            
            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Metric distribution plots saved")
    
    def _plot_individual_histograms(self, train_df: pd.DataFrame, test_df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]] = None):
        """Create individual histogram plots for each metric."""
        print("   Creating individual histograms...")
        
        # Get metric columns
        metric_columns = [col for col in train_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        for metric in metric_columns:
            try:
                # Plot distributions (handle both string and binary labels)
                train_good = train_df[train_df['label'].isin([0, 'good', 'Good'])][metric].values
                train_bad = train_df[train_df['label'].isin([1, 'bad', 'Bad'])][metric].values
                test_good = test_df[test_df['label'].isin([0, 'good', 'Good'])][metric].values
                test_bad = test_df[test_df['label'].isin([1, 'bad', 'Bad'])][metric].values
                
                # Debug: Check data availability
                print(f"   {metric}: train_good={len(train_good)}, train_bad={len(train_bad)}, test_good={len(test_good)}, test_bad={len(test_bad)}")
                
                # Special debugging for M-distance features
                if 'mdistance' in metric.lower():
                    print(f"   M-distance debug for {metric}:")
                    print(f"     Train data shape: {train_df.shape}")
                    print(f"     Test data shape: {test_df.shape}")
                    print(f"     Train columns: {list(train_df.columns)}")
                    print(f"     Train label values: {train_df['label'].value_counts().to_dict()}")
                    print(f"     Train metric values range: [{np.nanmin(train_df[metric]):.6f}, {np.nanmax(train_df[metric]):.6f}]")
                    print(f"     Test metric values range: [{np.nanmin(test_df[metric]):.6f}, {np.nanmax(test_df[metric]):.6f}]")
                    print(f"     Train metric NaN count: {train_df[metric].isna().sum()}")
                    print(f"     Test metric NaN count: {test_df[metric].isna().sum()}")
                
                # Check if we have any data
                if len(train_good) == 0 and len(train_bad) == 0 and len(test_good) == 0 and len(test_bad) == 0:
                    print(f"   Warning: No data found for {metric}, skipping histogram")
                    continue
                
                # Remove NaN values
                train_good = train_good[~np.isnan(train_good)]
                train_bad = train_bad[~np.isnan(train_bad)]
                test_good = test_good[~np.isnan(test_good)]
                test_bad = test_bad[~np.isnan(test_bad)]
                
                # Create separate plots for training and testing
                self._create_histogram_plot(metric, train_good, train_bad, 'Training', thresholds)
                self._create_histogram_plot(metric, test_good, test_bad, 'Testing', thresholds)
                
            except Exception as e:
                print(f"   Warning: Could not create histogram for {metric}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"   Individual histograms saved to {self.histograms_dir}")
    
    def _create_histogram_plot(self, metric: str, good_values: np.ndarray, bad_values: np.ndarray, 
                              split_name: str, thresholds: Dict[str, Dict[str, float]] = None):
        """Create a single histogram plot with thresholds."""
        plt.figure(figsize=(12, 8))
        
        # Calculate bins for consistent scaling
        all_values = np.concatenate([good_values, bad_values]) if len(good_values) > 0 and len(bad_values) > 0 else (good_values if len(good_values) > 0 else bad_values)
        if len(all_values) == 0:
            print(f"   Warning: No data for {metric} {split_name}")
            plt.close()
            return
        
        bins = np.linspace(np.min(all_values), np.max(all_values), 30)
        
        # Plot histograms with counts (not density)
        if len(good_values) > 0:
            plt.hist(good_values, bins=bins, alpha=0.7, label=f'Normal (n={len(good_values)})', 
                    color='green', edgecolor='darkgreen', linewidth=0.5)
        if len(bad_values) > 0:
            plt.hist(bad_values, bins=bins, alpha=0.7, label=f'Defect (n={len(bad_values)})', 
                    color='red', edgecolor='darkred', linewidth=0.5)
        
        # Add threshold lines if available
        if thresholds and metric in thresholds:
            threshold_colors = ['blue', 'black', 'orange', 'purple']
            threshold_methods = ['2sigma', '95th_quantile', 'max_acc', 'max_ba']
            
            for i, method in enumerate(threshold_methods):
                if method in thresholds[metric]:
                    threshold_value = thresholds[metric][method]
                    color = threshold_colors[i % len(threshold_colors)]
                    plt.axvline(threshold_value, color=color, linestyle='--', linewidth=2, 
                              label=f'{method}: {threshold_value:.4f}')
        
        if 'mdistance' in metric.lower() or 'mse' in metric.lower():
            plt.title(f'{metric} Distribution - {split_name} Set')
        else:
            plt.title(f'1 - {metric} Distribution - {split_name} Set')
        plt.xlabel('Anomaly Score (Higher = More Anomalous)')
        plt.ylabel('Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.histograms_dir / f'{metric}_{split_name.lower()}_histogram.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, test_df: pd.DataFrame, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Plot ROC curves for different methods."""
        print("   Creating ROC curves...")
        
        # Get metric columns
        metric_columns = [col for col in test_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        # Create subplots
        n_metrics = len(metric_columns)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle different subplot configurations
        if n_metrics == 1:
            # Single metric - axes is a single Axes object
            axes_array = [axes]
        elif n_rows == 1:
            # Single row - axes is a 1D array
            axes_array = axes.reshape(1, -1)
        else:
            # Multiple rows and columns - axes is a 2D array
            axes_array = axes
        
        for i, metric in enumerate(metric_columns):
            row = i // n_cols
            col = i % n_cols
            
            # Get the correct axes object
            if n_metrics == 1:
                ax = axes_array[0]
            elif n_rows == 1:
                ax = axes_array[0, col]
            else:
                ax = axes_array[row, col]
            
            # Get metric values and labels
            metric_values = test_df[metric].values
            labels = test_df['label'].values
            
            # Convert labels to binary format for sklearn functions
            labels_binary = self._convert_to_binary(labels)
            
            # All metrics are now standardized anomaly scores (higher = more anomalous)
            # Use metric values directly as scores for ROC curve
            scores = metric_values
            
            fpr, tpr, _ = roc_curve(labels_binary, scores)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {metric}')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ROC curves saved")
        
        # Create individual ROC curve plots
        self._plot_individual_roc_curves(test_df)
    
    def _plot_individual_roc_curves(self, test_df: pd.DataFrame):
        """Create individual ROC curve plots for each metric."""
        print("   Creating individual ROC curves...")
        
        # Get metric columns
        metric_columns = [col for col in test_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        for metric in metric_columns:
            try:
                # Get metric values and labels
                metric_values = test_df[metric].values
                labels = test_df['label'].values
                
                # Convert labels to binary format for sklearn functions
                labels_binary = self._convert_to_binary(labels)
                
                # All metrics are now standardized anomaly scores (higher = more anomalous)
                # Use metric values directly as scores for ROC curve
                scores = metric_values
                
                fpr, tpr, _ = roc_curve(labels_binary, scores)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random classifier')
                
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {metric}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.roc_curves_dir / f'{metric}_roc_curves.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"   Warning: Could not create ROC curve for {metric}: {e}")
                continue
        
        print(f"   Individual ROC curves saved to {self.roc_curves_dir}")
    
    def _plot_confusion_matrices(self, test_df: pd.DataFrame, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Plot confusion matrices for different methods."""
        print("   Creating confusion matrices...")
        
        # Get metric columns
        metric_columns = [col for col in test_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        # Create subplots for each metric
        for metric in metric_columns:
            if metric not in results:
                continue
            
            # Get the best method for this metric (by balanced accuracy)
            best_method = None
            best_ba = 0
            
            for method, result in results[metric].items():
                if method == 'max_ba_test':
                    continue
                # Check if balanced_accuracy exists, otherwise skip
                if 'balanced_accuracy' not in result:
                    continue
                if result['balanced_accuracy'] > best_ba:
                    best_ba = result['balanced_accuracy']
                    best_method = method
            
            if best_method is None:
                continue
            
            # Create confusion matrix
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            
            # Get confusion matrix values
            tp = results[metric][best_method]['tp']
            fp = results[metric][best_method]['fp']
            tn = results[metric][best_method]['tn']
            fn = results[metric][best_method]['fn']
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                       yticklabels=['Actual Normal', 'Actual Anomaly'])
            
            ax.set_title(f'Confusion Matrix - {metric} ({best_method})\n'
                        f'Balanced Accuracy: {best_ba:.3f}')
            
            plt.tight_layout()
            plt.savefig(self.confusion_matrices_dir / f'confusion_matrix_{metric}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   Confusion matrices saved")
    
    def _plot_threshold_analysis(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                thresholds: Dict[str, Dict[str, float]]):
        """Plot threshold analysis showing good vs bad distributions and thresholds."""
        print("   Creating threshold analysis plots...")
        
        # Get metric columns
        metric_columns = [col for col in train_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        # Create subplots
        n_metrics = len(metric_columns)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
        
        # Handle different subplot configurations
        if n_metrics == 1:
            # Single metric - axes is a single Axes object
            axes_array = [axes]
        elif n_rows == 1:
            # Single row - axes is a 1D array
            axes_array = axes.reshape(1, -1)
        else:
            # Multiple rows and columns - axes is a 2D array
            axes_array = axes
        
        for i, metric in enumerate(metric_columns):
            row = i // n_cols
            col = i % n_cols
            
            # Get the correct axes object
            if n_metrics == 1:
                ax = axes_array[0]
            elif n_rows == 1:
                ax = axes_array[0, col]
            else:
                ax = axes_array[row, col]
            
            # Get data (handle both string and binary labels)
            train_good = train_df[train_df['label'].isin([0, 'good', 'Good'])][metric].values
            train_bad = train_df[train_df['label'].isin([1, 'bad', 'Bad'])][metric].values
            
            # Plot distributions
            ax.hist(train_good, bins=30, alpha=0.7, label='Good', density=True, color='green')
            ax.hist(train_bad, bins=30, alpha=0.7, label='Bad', density=True, color='red')
            
            # Plot thresholds with different colors
            threshold_colors = [
                'blue', 'orange', 'purple', 'black', 'magenta', 'cyan', 'olive', 'navy', 'teal', 'gold'
            ]
            if metric in thresholds:
                for idx, (method, threshold) in enumerate(thresholds[metric].items()):
                    color = threshold_colors[idx % len(threshold_colors)]
                    ax.axvline(threshold, color=color, linestyle='--', 
                               label=f'{method}: {threshold:.3f}')
            
            if 'mdistance' in metric.lower() or 'mse' in metric.lower():
                ax.set_title(f'{metric} - Threshold Analysis')
            else:
                ax.set_title(f'1 - {metric} - Threshold Analysis')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Threshold analysis plots saved")
    
    def _plot_performance_comparison(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Plot performance comparison across methods."""
        print("   Creating performance comparison plots...")
        
        # Collect data
        all_data = []
        
        for metric, metric_results in results.items():
            for method, result in metric_results.items():
                if method == 'max_ba_test':
                    continue
                
                # Skip if required keys are missing
                required_keys = ['balanced_accuracy', 'f1', 'precision', 'recall']
                if not all(key in result for key in required_keys):
                    continue
                
                all_data.append({
                    'Metric': metric,
                    'Method': method,
                    'Balanced_Accuracy': result.get('balanced_accuracy', 0.0),
                    'AUC_ROC': result.get('auc_roc', 0.0),
                    'F1_Score': result.get('f1', 0.0),
                    'Precision': result.get('precision', 0.0),
                    'Recall': result.get('recall', 0.0)
                })
        
        df = pd.DataFrame(all_data)
        
        # Create performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Balanced Accuracy comparison
        sns.boxplot(data=df, x='Method', y='Balanced_Accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Balanced Accuracy by Method')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # AUC-ROC comparison
        sns.boxplot(data=df, x='Method', y='AUC_ROC', ax=axes[0, 1])
        axes[0, 1].set_title('AUC-ROC by Method')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        sns.boxplot(data=df, x='Method', y='F1_Score', ax=axes[1, 0])
        axes[1, 0].set_title('F1 Score by Method')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall scatter
        sns.scatterplot(data=df, x='Precision', y='Recall', hue='Method', ax=axes[1, 1])
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create heatmap of performance metrics
        self._plot_performance_heatmap(df)
        
        print(f"   Performance comparison plots saved")
    
    def _plot_performance_heatmap(self, df: pd.DataFrame):
        """Plot heatmap of performance metrics."""
        # Pivot data for heatmap
        metrics = ['Balanced_Accuracy', 'AUC_ROC', 'F1_Score', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 6))
        
        # Handle different subplot configurations
        if len(metrics) == 1:
            # Single metric - axes is a single Axes object
            axes_array = [axes]
        else:
            # Multiple metrics - axes is a 1D array
            axes_array = axes
        
        for i, metric in enumerate(metrics):
            pivot_data = df.pivot_table(values=metric, index='Metric', columns='Method', aggfunc='mean')
            
            # Get the correct axes object
            ax = axes_array[i]
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': metric})
            ax.set_title(f'{metric} Heatmap')
            ax.set_xlabel('Method')
            ax.set_ylabel('Metric')
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Plot feature analysis including correlation matrices."""
        print("   Creating feature analysis plots...")
        
        # Get metric columns
        metric_columns = [col for col in train_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        if len(metric_columns) < 2:
            print("   Not enough metrics for correlation analysis")
            return
        
        # Create correlation matrix
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training data correlation
        train_corr = train_df[metric_columns].corr()
        sns.heatmap(train_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   ax=axes[0], center=0)
        axes[0].set_title('Training Data - Metric Correlations')
        
        # Test data correlation
        test_corr = test_df[metric_columns].corr()
        sns.heatmap(test_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   ax=axes[1], center=0)
        axes[1].set_title('Test Data - Metric Correlations')
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot matrix for top metrics
        top_metrics = metric_columns[:6]  # Limit to 6 metrics for readability
        
        if len(top_metrics) >= 2:
            fig = sns.pairplot(train_df[top_metrics + ['label']], 
                              hue='label', diag_kind='hist')
            fig.fig.suptitle('Metric Scatter Plot Matrix (Training Data)', y=1.02)
            plt.savefig(self.other_dir / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   Feature analysis plots saved")
    
    def _is_inverse_metric(self, metric_name: str) -> bool:
        """Check if a metric is inverse (lower is better)."""
        # All metrics are now standardized to anomaly scores (higher = more anomalous)
        # This method is kept for backward compatibility but always returns False
        return False
    
    def create_summary_plot(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Create a summary plot showing the best performing methods."""
        print("   Creating summary plot...")
        
        # Collect best methods for each metric
        best_methods = []
        
        for metric, metric_results in results.items():
            best_method = None
            best_ba = 0
            
            for method, result in metric_results.items():
                if method == 'max_ba_test':
                    continue
                # Check if balanced_accuracy exists
                if 'balanced_accuracy' not in result:
                    continue
                if result['balanced_accuracy'] > best_ba:
                    best_ba = result['balanced_accuracy']
                    best_method = method
            
            if best_method:
                best_methods.append({
                    'Metric': metric,
                    'Method': best_method,
                    'Balanced_Accuracy': best_ba,
                    'AUC_ROC': metric_results[best_method].get('auc_roc', 0.0),
                    'F1_Score': metric_results[best_method].get('f1', 0.0)
                })
        
        if not best_methods:
            print("   No results available for summary plot")
            return
        
        df = pd.DataFrame(best_methods)
        df = df.sort_values('Balanced_Accuracy', ascending=False)
        
        # Create summary plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar(range(len(df)), df['Balanced_Accuracy'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(df))))
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df['Balanced_Accuracy'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Metric-Method Combination')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Best Performing Methods by Metric')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"{row['Metric']}\n({row['Method']})" for _, row in df.iterrows()], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.other_dir / 'summary_best_methods.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Summary plot saved")
