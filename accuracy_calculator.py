"""
Accuracy Calculator Module
Calculates accuracy metrics by comparing predicted scores with ground truth.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import math


class AccuracyCalculator:
    """Calculates accuracy metrics for code evaluation system."""

    def __init__(self, classification_threshold: float = 0.7):
        """
        Initialize Accuracy Calculator.

        Args:
            classification_threshold: Threshold for binary classification (pass/fail)
        """
        self.threshold = classification_threshold

    def calculate_regression_metrics(
        self,
        ground_truth: List[float],
        predictions: List[float]
    ) -> Dict[str, float]:
        """
        Calculate regression metrics (MAE, RMSE, R², etc.).

        Args:
            ground_truth: List of ground truth scores (0-1 or 0-100)
            predictions: List of predicted scores (0-1 or 0-100)

        Returns:
            Dictionary with regression metrics
        """
        # Convert to numpy arrays
        y_true = np.array(ground_truth)
        y_pred = np.array(predictions)

        # Ensure same length
        if len(y_true) != len(y_pred):
            raise ValueError("Ground truth and predictions must have same length")

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Calculate correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = 0

        # Calculate prediction bounds
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'correlation': float(correlation),
            'mape': float(mape),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors))
        }

    def calculate_classification_metrics(
        self,
        ground_truth: List[float],
        predictions: List[float],
        threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Calculate classification metrics (accuracy, precision, recall, F1).

        Args:
            ground_truth: List of ground truth scores
            predictions: List of predicted scores
            threshold: Classification threshold (if None, uses default)

        Returns:
            Dictionary with classification metrics
        """
        if threshold is None:
            threshold = self.threshold

        # Convert scores to binary labels (pass/fail)
        y_true = np.array([1 if score >= threshold else 0 for score in ground_truth])
        y_pred = np.array([1 if score >= threshold else 0 for score in predictions])

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Handle edge cases where not all classes are present
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class present
            if y_true[0] == 1:
                tp = cm[0, 0]
                tn, fp, fn = 0, 0, 0
            else:
                tn = cm[0, 0]
                tp, fp, fn = 0, 0, 0
        else:
            tp, tn, fp, fn = 0, 0, 0, 0

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'specificity': float(specificity),
            'confusion_matrix': {
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            },
            'threshold': float(threshold)
        }

    def compare_methods(
        self,
        ground_truth: List[float],
        method1_predictions: List[float],
        method2_predictions: List[float],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2"
    ) -> Dict[str, any]:
        """
        Compare two evaluation methods against ground truth.

        Args:
            ground_truth: List of ground truth scores
            method1_predictions: Predictions from first method
            method2_predictions: Predictions from second method
            method1_name: Name of first method
            method2_name: Name of second method

        Returns:
            Comparison results
        """
        # Calculate metrics for both methods
        method1_regression = self.calculate_regression_metrics(ground_truth, method1_predictions)
        method2_regression = self.calculate_regression_metrics(ground_truth, method2_predictions)

        method1_classification = self.calculate_classification_metrics(ground_truth, method1_predictions)
        method2_classification = self.calculate_classification_metrics(ground_truth, method2_predictions)

        # Determine which method is better
        better_mae = method1_name if method1_regression['mae'] < method2_regression['mae'] else method2_name
        better_rmse = method1_name if method1_regression['rmse'] < method2_regression['rmse'] else method2_name
        better_r2 = method1_name if method1_regression['r2_score'] > method2_regression['r2_score'] else method2_name
        better_accuracy = method1_name if method1_classification['accuracy'] > method2_classification['accuracy'] else method2_name

        return {
            method1_name: {
                'regression_metrics': method1_regression,
                'classification_metrics': method1_classification
            },
            method2_name: {
                'regression_metrics': method2_regression,
                'classification_metrics': method2_classification
            },
            'comparison': {
                'better_mae': better_mae,
                'better_rmse': better_rmse,
                'better_r2': better_r2,
                'better_accuracy': better_accuracy,
                'mae_difference': abs(method1_regression['mae'] - method2_regression['mae']),
                'rmse_difference': abs(method1_regression['rmse'] - method2_regression['rmse']),
                'r2_difference': abs(method1_regression['r2_score'] - method2_regression['r2_score']),
                'accuracy_difference': abs(method1_classification['accuracy'] - method2_classification['accuracy'])
            }
        }

    def analyze_errors(
        self,
        ground_truth: List[float],
        predictions: List[float],
        identifiers: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Analyze prediction errors in detail.

        Args:
            ground_truth: List of ground truth scores
            predictions: List of predicted scores
            identifiers: Optional identifiers for each prediction (e.g., student IDs)

        Returns:
            Error analysis results
        """
        y_true = np.array(ground_truth)
        y_pred = np.array(predictions)
        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        # Find worst predictions
        worst_indices = np.argsort(abs_errors)[::-1][:5]  # Top 5 worst
        best_indices = np.argsort(abs_errors)[:5]  # Top 5 best

        worst_predictions = []
        for idx in worst_indices:
            item = {
                'index': int(idx),
                'ground_truth': float(y_true[idx]),
                'prediction': float(y_pred[idx]),
                'error': float(errors[idx]),
                'absolute_error': float(abs_errors[idx])
            }
            if identifiers and idx < len(identifiers):
                item['identifier'] = identifiers[idx]
            worst_predictions.append(item)

        best_predictions = []
        for idx in best_indices:
            item = {
                'index': int(idx),
                'ground_truth': float(y_true[idx]),
                'prediction': float(y_pred[idx]),
                'error': float(errors[idx]),
                'absolute_error': float(abs_errors[idx])
            }
            if identifiers and idx < len(identifiers):
                item['identifier'] = identifiers[idx]
            best_predictions.append(item)

        # Error distribution
        overestimations = np.sum(errors > 0)
        underestimations = np.sum(errors < 0)
        perfect_predictions = np.sum(errors == 0)

        # Error ranges
        small_errors = np.sum(abs_errors < 0.1)  # Within 10%
        medium_errors = np.sum((abs_errors >= 0.1) & (abs_errors < 0.2))  # 10-20%
        large_errors = np.sum(abs_errors >= 0.2)  # Over 20%

        return {
            'worst_predictions': worst_predictions,
            'best_predictions': best_predictions,
            'error_distribution': {
                'overestimations': int(overestimations),
                'underestimations': int(underestimations),
                'perfect_predictions': int(perfect_predictions)
            },
            'error_ranges': {
                'small_errors_under_10_percent': int(small_errors),
                'medium_errors_10_20_percent': int(medium_errors),
                'large_errors_over_20_percent': int(large_errors)
            },
            'total_samples': len(ground_truth)
        }

    def comprehensive_accuracy_report(
        self,
        ground_truth: List[float],
        codebert_predictions: List[float],
        token_predictions: List[float],
        combined_predictions: List[float],
        identifiers: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive accuracy report for all methods.

        Args:
            ground_truth: List of ground truth scores
            codebert_predictions: CodeBERT predictions
            token_predictions: Token similarity predictions
            combined_predictions: Combined method predictions
            identifiers: Optional identifiers for each sample

        Returns:
            Comprehensive accuracy report
        """
        # Calculate metrics for each method
        codebert_metrics = {
            'regression': self.calculate_regression_metrics(ground_truth, codebert_predictions),
            'classification': self.calculate_classification_metrics(ground_truth, codebert_predictions),
            'error_analysis': self.analyze_errors(ground_truth, codebert_predictions, identifiers)
        }

        token_metrics = {
            'regression': self.calculate_regression_metrics(ground_truth, token_predictions),
            'classification': self.calculate_classification_metrics(ground_truth, token_predictions),
            'error_analysis': self.analyze_errors(ground_truth, token_predictions, identifiers)
        }

        combined_metrics = {
            'regression': self.calculate_regression_metrics(ground_truth, combined_predictions),
            'classification': self.calculate_classification_metrics(ground_truth, combined_predictions),
            'error_analysis': self.analyze_errors(ground_truth, combined_predictions, identifiers)
        }

        # Compare methods
        codebert_vs_token = self.compare_methods(
            ground_truth, codebert_predictions, token_predictions,
            "CodeBERT", "Token Similarity"
        )

        codebert_vs_combined = self.compare_methods(
            ground_truth, codebert_predictions, combined_predictions,
            "CodeBERT", "Combined"
        )

        token_vs_combined = self.compare_methods(
            ground_truth, token_predictions, combined_predictions,
            "Token Similarity", "Combined"
        )

        return {
            'codebert_method': codebert_metrics,
            'token_method': token_metrics,
            'combined_method': combined_metrics,
            'method_comparisons': {
                'codebert_vs_token': codebert_vs_token,
                'codebert_vs_combined': codebert_vs_combined,
                'token_vs_combined': token_vs_combined
            },
            'dataset_info': {
                'total_samples': len(ground_truth),
                'ground_truth_mean': float(np.mean(ground_truth)),
                'ground_truth_std': float(np.std(ground_truth)),
                'ground_truth_min': float(np.min(ground_truth)),
                'ground_truth_max': float(np.max(ground_truth))
            }
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    ground_truth = [0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.75, 0.65, 0.55, 0.45]
    codebert_pred = [0.88, 0.82, 0.68, 0.58, 0.52, 0.83, 0.78, 0.63, 0.57, 0.48]
    token_pred = [0.85, 0.75, 0.72, 0.62, 0.48, 0.80, 0.70, 0.68, 0.52, 0.42]
    combined_pred = [0.87, 0.79, 0.70, 0.60, 0.50, 0.82, 0.74, 0.65, 0.55, 0.45]

    print("Initializing Accuracy Calculator...")
    calculator = AccuracyCalculator(classification_threshold=0.7)

    print("\n" + "="*60)
    print("CodeBERT Method Metrics:")
    print("="*60)
    codebert_reg = calculator.calculate_regression_metrics(ground_truth, codebert_pred)
    print(f"MAE: {codebert_reg['mae']:.4f}")
    print(f"RMSE: {codebert_reg['rmse']:.4f}")
    print(f"R² Score: {codebert_reg['r2_score']:.4f}")
    print(f"Correlation: {codebert_reg['correlation']:.4f}")

    print("\n" + "="*60)
    print("Method Comparison:")
    print("="*60)
    comparison = calculator.compare_methods(
        ground_truth, codebert_pred, token_pred,
        "CodeBERT", "Token Similarity"
    )
    print(f"Better MAE: {comparison['comparison']['better_mae']}")
    print(f"Better RMSE: {comparison['comparison']['better_rmse']}")
    print(f"Better R²: {comparison['comparison']['better_r2']}")
    print(f"Better Accuracy: {comparison['comparison']['better_accuracy']}")

    print("\n" + "="*60)
    print("Error Analysis:")
    print("="*60)
    error_analysis = calculator.analyze_errors(ground_truth, combined_pred)
    print(f"Overestimations: {error_analysis['error_distribution']['overestimations']}")
    print(f"Underestimations: {error_analysis['error_distribution']['underestimations']}")
    print(f"Small errors (<10%): {error_analysis['error_ranges']['small_errors_under_10_percent']}")
