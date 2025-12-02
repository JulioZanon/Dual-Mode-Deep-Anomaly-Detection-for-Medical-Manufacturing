#!/usr/bin/env python3
"""
Performance Metrics Calculation Module

This module handles the calculation of various performance metrics
for anomaly detection evaluation including accuracy, precision, recall, F1, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import warnings

class PerformanceMetrics:
    """
    Handles calculation of performance metrics for anomaly detection.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the performance metrics calculator."""
        self.output_dir = output_dir
    
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
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            y_true: True labels (0 for normal, 1 for anomaly, or 'good'/'bad')
            y_pred: Predicted labels (0 for normal, 1 for anomaly, or 'good'/'bad')
            y_scores: Prediction scores (optional, for AUC calculation)
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Convert string labels to binary if needed
        y_true_binary = self._convert_to_binary(y_true)
        y_pred_binary = self._convert_to_binary(y_pred)
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['f2'] = fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)
        
        # Sensitivity and Specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Balanced Accuracy
        metrics['balanced_accuracy'] = 0.5 * metrics['sensitivity'] + 0.5 * metrics['specificity']
        
        # AUC-ROC and AUC-PR (if scores provided)
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true_binary, y_scores)
            except ValueError:
                metrics['auc_roc'] = 0.5  # Random classifier
            
            try:
                metrics['auc_pr'] = average_precision_score(y_true_binary, y_scores)
            except ValueError:
                metrics['auc_pr'] = 0.0  # No positives or all same class
        else:
            metrics['auc_roc'] = 0.5
            metrics['auc_pr'] = 0.0
        
        return metrics
    
    def evaluate_with_thresholds(self, test_df: pd.DataFrame, 
                                thresholds: Dict[str, Dict[str, float]], 
                                train_df: pd.DataFrame = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate test data using calculated thresholds.
        
        Args:
            test_df: Test data DataFrame with metrics and labels
            thresholds: Dictionary of thresholds for each metric and method
            
        Returns:
            Dictionary of evaluation results
        """
        print("📊 Evaluating test data with calculated thresholds...")
        
        results = {}
        
        # Get metric columns (exclude non-metric columns)
        metric_columns = [col for col in test_df.columns 
                         if col not in ['filename', 'original_filename', 'label']]
        
        # Create detailed results log
        detailed_log = []
        
        for metric in metric_columns:
            print(f"   Evaluating {metric}...")
            
            metric_values = test_df[metric].values
            labels = test_df['label'].values
            filenames = test_df['filename'].values if 'filename' in test_df.columns else [f"image_{i}" for i in range(len(labels))]
            
            # All metrics are now standardized anomaly scores (higher = more anomalous)
            metric_results = {}
            balanced_accuracies = []  # Track for max_balanced_accuracy calculation
            
            # Evaluate each threshold method
            for method, threshold in thresholds.get(metric, {}).items():
                if method == 'max_ba_test':
                    continue
                
                print(f"     Method: {method}, Threshold: {threshold:.6f}")
                
                # Make predictions based on threshold (higher values = more anomalous)
                predictions = metric_values >= threshold
                
                # Convert boolean predictions to string labels to match true labels
                predictions = np.where(predictions, 'bad', 'good')
                
                # Log detailed results for first few samples
                for i in range(min(10, len(labels))):
                    detailed_log.append({
                        'metric': metric,
                        'method': method,
                        'filename': filenames[i],
                        'original_label': labels[i],
                        'metric_value': metric_values[i],
                        'threshold': threshold,
                        'prediction': predictions[i],
                        'correct': labels[i] == predictions[i]
                    })
                
                # Calculate metrics
                metrics = self.calculate_all_metrics(labels, predictions, metric_values)
                metrics['threshold'] = threshold
                
                # Track balanced accuracy for max calculation
                balanced_accuracies.append(metrics['balanced_accuracy'])
                
                # Log summary statistics
                correct_predictions = np.sum(labels == predictions)
                total_predictions = len(labels)
                accuracy = correct_predictions / total_predictions
                print(f"       Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
                
                metric_results[method] = metrics
            
            # Calculate max_ba_test threshold (using test data)
            if metric in thresholds and 'max_ba' in thresholds[metric]:
                test_threshold = self._calculate_test_max_ba_threshold(
                    metric_values, labels
                )
                # For anomaly scores: higher values = more anomalous
                test_predictions = metric_values >= test_threshold
                # Convert boolean predictions to string labels to match true labels
                test_predictions = np.where(test_predictions, 'bad', 'good')
                test_metrics = self.calculate_all_metrics(labels, test_predictions, metric_values)
                test_metrics['threshold'] = test_threshold
                balanced_accuracies.append(test_metrics['balanced_accuracy'])
                metric_results['max_ba_test'] = test_metrics
            
            # Calculate max_balanced_accuracy across all threshold methods for this metric
            max_balanced_accuracy = max(balanced_accuracies) if balanced_accuracies else 0.0
            
            # Add max_balanced_accuracy to all threshold results for this metric
            for method in metric_results:
                metric_results[method]['max_balanced_accuracy'] = max_balanced_accuracy
            
            results[metric] = metric_results
        
        # Save detailed log
        self._save_detailed_log(detailed_log, train_df, thresholds)
        
        print("✅ Test evaluation completed")
        return results
    
    def _save_detailed_log(self, detailed_log: List[Dict], train_df: pd.DataFrame = None, 
                          thresholds: Dict[str, Dict[str, float]] = None):
        """Save detailed evaluation log to file with training data and thresholds."""
        if not detailed_log:
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        log_df = pd.DataFrame(detailed_log)
        
        # Print summary
        print(f"📋 Detailed evaluation log:")
        print(f"   Total entries: {len(log_df)}")
        print(f"   Metrics: {log_df['metric'].nunique()}")
        print(f"   Methods: {log_df['method'].nunique()}")
        
        # Print sample entries
        print("\n📊 Sample evaluation entries:")
        print(log_df.head(10).to_string(index=False))
        
        # Print accuracy summary by metric and method
        accuracy_summary = log_df.groupby(['metric', 'method'])['correct'].agg(['count', 'sum', 'mean']).round(4)
        accuracy_summary.columns = ['total', 'correct', 'accuracy']
        print(f"\n📈 Accuracy summary:")
        print(accuracy_summary.to_string())
        
        # Save to CSV if output_dir is available
        if self.output_dir is not None:
            # Save test evaluation log
            test_log_file = self.output_dir / 'metrics' / 'detailed_evaluation_log_test.csv'
            test_log_file.parent.mkdir(exist_ok=True)
            log_df.to_csv(test_log_file, index=False)
            print(f"📁 Test evaluation log saved to {test_log_file}")
            
            # Save training data log if available
            if train_df is not None:
                train_log = []
                for metric in train_df.columns:
                    if metric not in ['filename', 'original_filename', 'label']:
                        for method in ['2sigma', '95th_quantile', 'max_acc', 'max_ba']:
                            if metric in thresholds and method in thresholds[metric]:
                                threshold = thresholds[metric][method]
                                
                                for idx, row in train_df.iterrows():
                                    metric_value = row[metric]
                                    original_label = row['label']
                                    
                                    # Determine prediction based on threshold (higher values = more anomalous)
                                    prediction = 'bad' if metric_value >= threshold else 'good'
                                    
                                    # Determine if prediction is correct
                                    correct = (prediction == 'bad' and original_label == 1) or (prediction == 'good' and original_label == 0)
                                    
                                    train_log.append({
                                        'metric': metric,
                                        'method': method,
                                        'filename': row.get('filename', f'train_{idx}'),
                                        'original_label': 'bad' if original_label == 1 else 'good',
                                        'metric_value': metric_value,
                                        'threshold': threshold,
                                        'prediction': prediction,
                                        'correct': correct
                                    })
                
                if train_log:
                    train_log_df = pd.DataFrame(train_log)
                    train_log_file = self.output_dir / 'metrics' / 'detailed_evaluation_log_train.csv'
                    train_log_df.to_csv(train_log_file, index=False)
                    print(f"📁 Training evaluation log saved to {train_log_file}")
            
            # Save thresholds summary
            if thresholds is not None:
                thresholds_data = []
                for metric, methods in thresholds.items():
                    for method, threshold in methods.items():
                        thresholds_data.append({
                            'metric': metric,
                            'method': method,
                            'threshold': threshold
                        })
                
                thresholds_df = pd.DataFrame(thresholds_data)
                thresholds_file = self.output_dir / 'metrics' / 'thresholds_summary.csv'
                thresholds_df.to_csv(thresholds_file, index=False)
                print(f"📁 Thresholds summary saved to {thresholds_file}")
        else:
            print("📁 Detailed evaluation log not saved (no output directory specified)")
    
    def _is_inverse_metric(self, metric_name: str) -> bool:
        """Check if a metric is inverse (lower is better)."""
        # All metrics are now standardized to anomaly scores (higher = more anomalous)
        # This method is kept for backward compatibility but always returns False
        return False
    
    def _calculate_test_max_ba_threshold(self, metric_values: np.ndarray, 
                                       labels: np.ndarray) -> float:
        """Calculate max balanced accuracy threshold on test data."""
        # Convert labels to binary if needed
        labels_binary = self._convert_to_binary(labels)
        good_values = metric_values[labels_binary == 0]
        bad_values = metric_values[labels_binary == 1]
        
        if len(good_values) == 0 or len(bad_values) == 0:
            return np.median(metric_values)
        
        # Check if distributions are non-overlapping
        good_max = np.max(good_values)
        bad_min = np.min(bad_values)
        if good_max < bad_min:
            # For anomaly scores: higher values = more anomalous
            return (good_max + bad_min) / 2
        
        # Overlapping distributions: use optimization
        unique_values = np.unique(metric_values)
        best_ba = 0
        best_threshold = unique_values[0]
        
        for threshold in unique_values:
            # For anomaly scores: higher values = more anomalous
            predictions = metric_values >= threshold
            
            tn, fp, fn, tp = confusion_matrix(labels_binary, predictions).ravel()
            
            if tp + fn > 0 and tn + fp > 0:
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                balanced_acc = 0.5 * sensitivity + 0.5 * specificity
                
                if balanced_acc > best_ba:
                    best_ba = balanced_acc
                    best_threshold = threshold
        
        return best_threshold
    
    def create_results_tables(self, thresholds: Dict[str, Dict[str, float]], 
                            results: Dict[str, Dict[str, Dict[str, float]]], 
                            tables_dir: Path):
        """Create comprehensive results tables."""
        print(f"📊 Creating results tables...")
        
        # Create tables directory
        tables_dir.mkdir(exist_ok=True)
        
        # Create different types of tables
        self._create_overall_performance_table(thresholds, results, tables_dir)
        self._create_detailed_methods_table(thresholds, results, tables_dir)
        self._create_thresholds_table(thresholds, tables_dir)
        self._create_best_methods_table(results, tables_dir)
        self._create_master_results_file(thresholds, results, tables_dir)
        
        print(f"✅ Results tables created in {tables_dir}")
    
    def _create_overall_performance_table(self, thresholds: Dict[str, Dict[str, float]], 
                                        results: Dict[str, Dict[str, Dict[str, float]]], 
                                        tables_dir: Path):
        """Create overall performance summary table."""
        print(f"   Creating overall performance table...")
        
        all_data = []
        
        for metric, metric_results in results.items():
            for method, result in metric_results.items():
                if method == 'max_ba_test':
                    continue
                
                all_data.append({
                    'Metric': metric,
                    'Method': method,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1'],
                    'F2_Score': result['f2'],
                    'Balanced_Accuracy': result['balanced_accuracy'],
                    'Max_Balanced_Accuracy': result.get('max_balanced_accuracy', 0.0),
                    'AUC_ROC': result['auc_roc'],
                    'AUC_PR': result.get('auc_pr', 0.0),
                    'Specificity': result['specificity'],
                    'Sensitivity': result['sensitivity']
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df = df.sort_values(['Metric', 'Balanced_Accuracy'], ascending=[True, False])
        
        # Save CSV
        df.to_csv(tables_dir / 'overall_performance.csv', index=False)
        
        # Save TXT summary
        with open(tables_dir / 'overall_performance_summary.txt', 'w') as f:
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Best performing methods
            best_methods = df.nlargest(10, 'Balanced_Accuracy')
            f.write("TOP 10 METHODS (by Balanced Accuracy):\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Rank':<4} {'Metric':<25} {'Method':<15} {'Bal_Acc':<10} {'AUC_ROC':<9} {'AUC_PR':<9} {'F1':<8}\n")
            f.write("-" * 90 + "\n")
            
            for i, (_, row) in enumerate(best_methods.iterrows(), 1):
                f.write(f"{i:<4} {row['Metric']:<25} {row['Method']:<15} "
                       f"{row['Balanced_Accuracy']:<10.4f} {row['AUC_ROC']:<9.4f} {row.get('AUC_PR', 0.0):<9.4f} {row['F1_Score']:<8.4f}\n")
        
        print(f"   Overall performance table created")
    
    def _create_detailed_methods_table(self, thresholds: Dict[str, Dict[str, float]], 
                                     results: Dict[str, Dict[str, Dict[str, float]]], 
                                     tables_dir: Path):
        """Create detailed methods table with confusion matrix components."""
        print(f"   Creating detailed methods table...")
        
        all_data = []
        
        for metric, metric_results in results.items():
            for method, result in metric_results.items():
                if method == 'max_ba_test':
                    continue
                
                # Get training threshold
                train_threshold = thresholds.get(metric, {}).get(method, 'N/A')
                
                all_data.append({
                    'Metric': metric,
                    'Method': method,
                    'Training_Threshold': train_threshold,
                    'Test_Threshold': result['threshold'],
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1'],
                    'F2_Score': result['f2'],
                    'Balanced_Accuracy': result['balanced_accuracy'],
                    'Max_Balanced_Accuracy': result.get('max_balanced_accuracy', 0.0),
                    'AUC_ROC': result['auc_roc'],
                    'AUC_PR': result.get('auc_pr', 0.0),
                    'Specificity': result['specificity'],
                    'Sensitivity': result['sensitivity'],
                    'True_Positives': result['tp'],
                    'False_Positives': result['fp'],
                    'True_Negatives': result['tn'],
                    'False_Negatives': result['fn']
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df = df.sort_values(['Metric', 'Balanced_Accuracy'], ascending=[True, False])
        
        # Save CSV
        df.to_csv(tables_dir / 'detailed_methods.csv', index=False)
        
        # Save TXT summary
        with open(tables_dir / 'detailed_methods_summary.txt', 'w') as f:
            f.write("DETAILED METHODS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Method performance summary
            method_stats = df.groupby('Method').agg({
                'Balanced_Accuracy': ['mean', 'max', 'min', 'std'],
                'AUC_ROC': ['mean', 'max', 'min', 'std'],
                'AUC_PR': ['mean', 'max', 'min', 'std'],
                'F1_Score': ['mean', 'max', 'min', 'std']
            }).round(4)
            
            f.write("METHOD PERFORMANCE SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Method':<15} {'Bal_Acc_Avg':<12} {'AUC_ROC_Avg':<12} {'AUC_PR_Avg':<12}\n")
            f.write("-" * 70 + "\n")
            
            for method in method_stats.index:
                bal_acc_avg = method_stats.loc[method, ('Balanced_Accuracy', 'mean')]
                auc_roc_avg = method_stats.loc[method, ('AUC_ROC', 'mean')]
                auc_pr_avg = method_stats.loc[method, ('AUC_PR', 'mean')]
                
                f.write(f"{method:<15} {bal_acc_avg:<12.4f} {auc_roc_avg:<12.4f} {auc_pr_avg:<12.4f}\n")
        
        print(f"   Detailed methods table created")
    
    def _create_thresholds_table(self, thresholds: Dict[str, Dict[str, float]], 
                               tables_dir: Path):
        """Create thresholds summary table."""
        print(f"   Creating thresholds table...")
        
        all_data = []
        
        for metric, metric_thresholds in thresholds.items():
            for method, threshold in metric_thresholds.items():
                all_data.append({
                    'Metric': metric,
                    'Method': method,
                    'Threshold': threshold
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df = df.sort_values(['Metric', 'Method'])
        
        # Save CSV
        df.to_csv(tables_dir / 'thresholds_summary.csv', index=False)
        
        # Save TXT summary
        with open(tables_dir / 'thresholds_summary.txt', 'w') as f:
            f.write("THRESHOLDS SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            for metric in df['Metric'].unique():
                f.write(f"{metric}:\n")
                f.write("-" * 30 + "\n")
                metric_data = df[df['Metric'] == metric]
                for _, row in metric_data.iterrows():
                    f.write(f"  {row['Method']:<15}: {row['Threshold']:.6f}\n")
                f.write("\n")
        
        print(f"   Thresholds table created")
    
    def _create_best_methods_table(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                                 tables_dir: Path):
        """Create best methods summary table."""
        print(f"   Creating best methods table...")
        
        all_data = []
        
        for metric, metric_results in results.items():
            # Find best method for this metric (by balanced accuracy)
            best_method = None
            best_ba = 0
            
            for method, result in metric_results.items():
                if method == 'max_ba_test':
                    continue
                if result['balanced_accuracy'] > best_ba:
                    best_ba = result['balanced_accuracy']
                    best_method = method
            
            if best_method:
                best_result = metric_results[best_method]
                all_data.append({
                    'Metric': metric,
                    'Best_Method': best_method,
                    'Balanced_Accuracy': best_result['balanced_accuracy'],
                    'Max_Balanced_Accuracy': best_result.get('max_balanced_accuracy', 0.0),
                    'AUC_ROC': best_result['auc_roc'],
                    'AUC_PR': best_result.get('auc_pr', 0.0),
                    'F1_Score': best_result['f1'],
                    'Precision': best_result['precision'],
                    'Recall': best_result['recall'],
                    'Specificity': best_result['specificity'],
                    'Sensitivity': best_result['sensitivity']
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df = df.sort_values('Balanced_Accuracy', ascending=False)
        
        # Save CSV
        df.to_csv(tables_dir / 'best_methods.csv', index=False)
        
        # Save TXT summary
        with open(tables_dir / 'best_methods_summary.txt', 'w') as f:
            f.write("BEST METHODS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"{'Metric':<30} {'Best_Method':<15} {'Bal_Acc':<10} {'AUC_ROC':<9} {'AUC_PR':<9}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"{row['Metric']:<30} {row['Best_Method']:<15} "
                       f"{row['Balanced_Accuracy']:<10.4f} {row['AUC_ROC']:<9.4f} {row.get('AUC_PR', 0.0):<9.4f}\n")
        
        print(f"   Best methods table created")
    
    def _create_master_results_file(self, thresholds: Dict[str, Dict[str, float]], 
                                   results: Dict[str, Dict[str, Dict[str, float]]], 
                                   tables_dir: Path):
        """Create master results file with all methods and metrics."""
        print(f"   Creating master results file...")
        
        # Collect all data
        all_data = []
        
        for metric, metric_results in results.items():
            for method, result in metric_results.items():
                if method == 'max_ba_test':
                    continue
                
                # Get training threshold
                train_threshold = thresholds.get(metric, {}).get(method, 'N/A')
                
                all_data.append({
                    'Metric': metric,
                    'Method': method,
                    'Training_Threshold': train_threshold,
                    'Test_Threshold': result['threshold'],
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1'],
                    'F2_Score': result['f2'],
                    'Balanced_Accuracy': result['balanced_accuracy'],
                    'Max_Balanced_Accuracy': result.get('max_balanced_accuracy', 0.0),
                    'AUC_ROC': result['auc_roc'],
                    'AUC_PR': result.get('auc_pr', 0.0),
                    'Specificity': result['specificity'],
                    'Sensitivity': result['sensitivity'],
                    'True_Positives': result['tp'],
                    'False_Positives': result['fp'],
                    'True_Negatives': result['tn'],
                    'False_Negatives': result['fn'],
                    'Total_Samples': result['tp'] + result['fp'] + result['tn'] + result['fn']
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df = df.sort_values(['Metric', 'Balanced_Accuracy'], ascending=[True, False])
        
        # Save master CSV file
        df.to_csv(tables_dir / 'master_results_all_methods.csv', index=False)
        
        # Create comprehensive summary
        with open(tables_dir / 'master_results_summary.txt', 'w') as f:
            f.write("MASTER RESULTS SUMMARY - ALL METHODS TESTED\n")
            f.write("=" * 120 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Metrics Tested: {df['Metric'].nunique()}\n")
            f.write(f"Total Methods per Metric: {df.groupby('Metric')['Method'].nunique().iloc[0]}\n")
            f.write(f"Total Method-Metric Combinations: {len(df)}\n")
            f.write(f"Best Overall Balanced Accuracy: {df['Balanced_Accuracy'].max():.4f}\n")
            f.write(f"Best Overall AUC-ROC: {df['AUC_ROC'].max():.4f}\n")
            f.write(f"Best Overall AUC-PR: {df['AUC_PR'].max():.4f}\n")
            f.write(f"Average Balanced Accuracy: {df['Balanced_Accuracy'].mean():.4f}\n")
            f.write(f"Average AUC-ROC: {df['AUC_ROC'].mean():.4f}\n")
            f.write(f"Average AUC-PR: {df['AUC_PR'].mean():.4f}\n\n")
            
            # Top 10 performing methods
            f.write("TOP 10 PERFORMING METHODS (by Balanced Accuracy):\n")
            f.write("-" * 90 + "\n")
            top_10 = df.nlargest(10, 'Balanced_Accuracy')
            f.write(f"{'Rank':<4} {'Metric':<25} {'Method':<15} {'Bal_Acc':<10} {'AUC_ROC':<9} {'AUC_PR':<9} {'F1':<8}\n")
            f.write("-" * 90 + "\n")
            
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:<4} {row['Metric']:<25} {row['Method']:<15} "
                       f"{row['Balanced_Accuracy']:<10.4f} {row['AUC_ROC']:<9.4f} {row.get('AUC_PR', 0.0):<9.4f} {row['F1_Score']:<8.4f}\n")
            
            f.write("\n")
            
            # Best method per metric
            f.write("BEST METHOD PER METRIC:\n")
            f.write("-" * 70 + "\n")
            best_per_metric = df.loc[df.groupby('Metric')['Balanced_Accuracy'].idxmax()]
            f.write(f"{'Metric':<30} {'Best_Method':<15} {'Bal_Acc':<10} {'AUC_ROC':<9} {'AUC_PR':<9}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in best_per_metric.iterrows():
                f.write(f"{row['Metric']:<30} {row['Method']:<15} "
                       f"{row['Balanced_Accuracy']:<10.4f} {row['AUC_ROC']:<9.4f} {row.get('AUC_PR', 0.0):<9.4f}\n")
            
            f.write("\n")
            
            # Method performance summary
            f.write("METHOD PERFORMANCE SUMMARY:\n")
            f.write("-" * 80 + "\n")
            method_stats = df.groupby('Method').agg({
                'Balanced_Accuracy': ['mean', 'max', 'min', 'std'],
                'AUC_ROC': ['mean', 'max', 'min', 'std'],
                'AUC_PR': ['mean', 'max', 'min', 'std'],
                'F1_Score': ['mean', 'max', 'min', 'std']
            }).round(4)
            
            f.write(f"{'Method':<15} {'Bal_Acc_Avg':<12} {'AUC_ROC_Avg':<12} {'AUC_PR_Avg':<12}\n")
            f.write("-" * 80 + "\n")
            
            for method in method_stats.index:
                bal_acc_avg = method_stats.loc[method, ('Balanced_Accuracy', 'mean')]
                auc_roc_avg = method_stats.loc[method, ('AUC_ROC', 'mean')]
                auc_pr_avg = method_stats.loc[method, ('AUC_PR', 'mean')]
                
                f.write(f"{method:<15} {bal_acc_avg:<12.4f} {auc_roc_avg:<12.4f} {auc_pr_avg:<12.4f}\n")
        
        print(f"   Master results file created")
