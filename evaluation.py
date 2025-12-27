#!/usr/bin/env python3
"""
Evaluation Module for Rainfall Prediction

This module provides comprehensive evaluation metrics and analysis for the rainfall prediction model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, median_absolute_error
)
from typing import Dict, List, Tuple, Optional
import json
import os


class ModelEvaluator:
    """
    A comprehensive evaluator for regression models with rainfall prediction focus.
    """

    def __init__(self, model_name: str = "rainfall_predictor"):
        """
        Initialize the model evaluator.

        Args:
            model_name (str): Name identifier for the model
        """
        self.model_name = model_name
        self.metrics = {}
        self.predictions = None
        self.actual_values = None

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression evaluation metrics.

        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        self.actual_values = y_true
        self.predictions = y_pred

        metrics = {
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred),
            'explained_variance_score': explained_variance_score(y_true, y_pred),
            'median_absolute_error': median_absolute_error(y_true, y_pred)
        }

        # Additional rainfall-specific metrics
        metrics.update(self._calculate_rainfall_specific_metrics(y_true, y_pred))

        self.metrics = metrics
        return metrics

    def _calculate_rainfall_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate rainfall-specific evaluation metrics.

        Args:
            y_true (np.ndarray): True rainfall values
            y_pred (np.ndarray): Predicted rainfall values

        Returns:
            Dict[str, float]: Rainfall-specific metrics
        """
        # Mean Absolute Percentage Error (MAPE) - handle zero values
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan

        # Accuracy within certain thresholds (common in rainfall prediction)
        thresholds = [1, 5, 10, 25]  # mm thresholds
        threshold_accuracies = {}

        for threshold in thresholds:
            # Percentage of predictions within threshold mm of actual
            within_threshold = np.abs(y_true - y_pred) <= threshold
            threshold_accuracies[f'accuracy_within_{threshold}mm'] = np.mean(within_threshold) * 100

        # Rainfall event detection metrics
        rainfall_threshold = 0.1  # mm - consider this as "rainfall event"
        actual_rainfall_events = y_true > rainfall_threshold
        predicted_rainfall_events = y_pred > rainfall_threshold

        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((actual_rainfall_events) & (predicted_rainfall_events))
        fp = np.sum((~actual_rainfall_events) & (predicted_rainfall_events))
        tn = np.sum((~actual_rainfall_events) & (~predicted_rainfall_events))
        fn = np.sum((actual_rainfall_events) & (~predicted_rainfall_events))

        # Rainfall detection metrics
        rainfall_metrics = {}
        if tp + fp > 0:  # Precision
            rainfall_metrics['rainfall_precision'] = tp / (tp + fp)
        else:
            rainfall_metrics['rainfall_precision'] = 0.0

        if tp + fn > 0:  # Recall
            rainfall_metrics['rainfall_recall'] = tp / (tp + fn)
        else:
            rainfall_metrics['rainfall_recall'] = 0.0

        if tp + fp + tn + fn > 0:  # Accuracy
            rainfall_metrics['rainfall_accuracy'] = (tp + tn) / (tp + fp + tn + fn)
        else:
            rainfall_metrics['rainfall_accuracy'] = 0.0

        # F1 Score for rainfall detection
        precision = rainfall_metrics['rainfall_precision']
        recall = rainfall_metrics['rainfall_recall']
        if precision + recall > 0:
            rainfall_metrics['rainfall_f1_score'] = 2 * (precision * recall) / (precision + recall)
        else:
            rainfall_metrics['rainfall_f1_score'] = 0.0

        return {
            'mean_absolute_percentage_error': mape,
            **threshold_accuracies,
            **rainfall_metrics
        }

    def calculate_residuals_analysis(self) -> Dict[str, np.ndarray]:
        """
        Perform residual analysis for model diagnostics.

        Returns:
            Dict[str, np.ndarray]: Residual analysis results
        """
        if self.predictions is None or self.actual_values is None:
            raise ValueError("Must calculate metrics first using calculate_regression_metrics()")

        residuals = self.actual_values - self.predictions

        analysis = {
            'residuals': residuals,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'residuals_min': np.min(residuals),
            'residuals_max': np.max(residuals),
            'residuals_skewness': self._calculate_skewness(residuals),
            'residuals_kurtosis': self._calculate_kurtosis(residuals)
        }

        return analysis

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def get_performance_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive performance summary.

        Returns:
            Dict[str, any]: Performance summary
        """
        if not self.metrics:
            raise ValueError("Must calculate metrics first using calculate_regression_metrics()")

        summary = {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'performance_rating': self._get_performance_rating(),
            'residuals_analysis': self.calculate_residuals_analysis(),
            'data_info': {
                'n_samples': len(self.actual_values) if self.actual_values is not None else 0,
                'target_range': {
                    'min': float(np.min(self.actual_values)) if self.actual_values is not None else None,
                    'max': float(np.max(self.actual_values)) if self.actual_values is not None else None,
                    'mean': float(np.mean(self.actual_values)) if self.actual_values is not None else None,
                    'std': float(np.std(self.actual_values)) if self.actual_values is not None else None
                }
            }
        }

        return summary

    def _get_performance_rating(self) -> str:
        """
        Get a qualitative performance rating based on R² score.

        Returns:
            str: Performance rating
        """
        r2 = self.metrics.get('r2_score', 0)

        if r2 >= 0.9:
            return "Excellent"
        elif r2 >= 0.8:
            return "Very Good"
        elif r2 >= 0.7:
            return "Good"
        elif r2 >= 0.6:
            return "Fair"
        elif r2 >= 0.5:
            return "Poor"
        else:
            return "Very Poor"

    def save_evaluation_results(self, output_path: str = "results/model_metrics.json") -> None:
        """
        Save evaluation results to JSON file.

        Args:
            output_path (str): Path to save the results
        """
        if not self.metrics:
            raise ValueError("Must calculate metrics first using calculate_regression_metrics()")

        results = self.get_performance_summary()

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        results = convert_numpy_types(results)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Evaluation results saved to {output_path}")

    def print_evaluation_report(self) -> None:
        """
        Print a formatted evaluation report.
        """
        if not self.metrics:
            raise ValueError("Must calculate metrics first using calculate_regression_metrics()")

        print(f"\n{'='*60}")
        print(f"Model Evaluation Report - {self.model_name}")
        print(f"{'='*60}")

        print(f"\n{'Basic Regression Metrics':<30}")
        print(f"{'-'*30}")
        print(f"Mean Absolute Error (MAE):     {self.metrics['mean_absolute_error']:.4f} mm")
        print(f"Mean Squared Error (MSE):      {self.metrics['mean_squared_error']:.4f} mm²")
        print(f"Root Mean Squared Error (RMSE): {self.metrics['root_mean_squared_error']:.4f} mm")
        print(f"R² Score:                      {self.metrics['r2_score']:.4f}")
        print(f"Explained Variance Score:      {self.metrics['explained_variance_score']:.4f}")

        print(f"\n{'Rainfall-Specific Metrics':<30}")
        print(f"{'-'*30}")
        if not np.isnan(self.metrics['mean_absolute_percentage_error']):
            print(f"Mean Absolute Percentage Error: {self.metrics['mean_absolute_percentage_error']:.2f}%")

        # Print threshold accuracies
        threshold_keys = [k for k in self.metrics.keys() if k.startswith('accuracy_within_')]
        for key in sorted(threshold_keys):
            threshold = key.split('_')[2]
            print(f"Accuracy within {threshold}:        {self.metrics[key]:.2f}%")

        print(f"\n{'Rainfall Detection Metrics':<30}")
        print(f"{'-'*30}")
        print(f"Rainfall Detection Precision:  {self.metrics['rainfall_precision']:.4f}")
        print(f"Rainfall Detection Recall:     {self.metrics['rainfall_recall']:.4f}")
        print(f"Rainfall Detection Accuracy:   {self.metrics['rainfall_accuracy']:.4f}")
        print(f"Rainfall Detection F1 Score:   {self.metrics['rainfall_f1_score']:.4f}")

        print(f"\n{'Performance Rating':<30}")
        print(f"{'-'*30}")
        print(f"Overall Rating: {self._get_performance_rating()}")

        if self.actual_values is not None:
            print(f"\n{'Dataset Information':<30}")
            print(f"{'-'*30}")
            print(f"Number of samples: {len(self.actual_values)}")
            print(f"Rainfall range: {np.min(self.actual_values):.2f} - {np.max(self.actual_values):.2f} mm")
            print(f"Rainfall mean: {np.mean(self.actual_values):.2f} mm")
            print(f"Rainfall std: {np.std(self.actual_values):.2f} mm")

        print(f"{'='*60}\n")


def compare_models(*evaluators: ModelEvaluator) -> pd.DataFrame:
    """
    Compare multiple model evaluators.

    Args:
        *evaluators: ModelEvaluator instances to compare

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = {}

    for evaluator in evaluators:
        if not evaluator.metrics:
            continue

        model_data = {
            'MAE': evaluator.metrics['mean_absolute_error'],
            'RMSE': evaluator.metrics['root_mean_squared_error'],
            'R²': evaluator.metrics['r2_score'],
            'Rainfall_Accuracy_5mm': evaluator.metrics.get('accuracy_within_5mm', 0),
            'Rainfall_F1': evaluator.metrics['rainfall_f1_score'],
            'Rating': evaluator._get_performance_rating()
        }
        comparison_data[evaluator.model_name] = model_data

    return pd.DataFrame(comparison_data).T


def main():
    """
    Main function to demonstrate evaluation capabilities.
    """
    print("=== Model Evaluation Demo ===")

    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    # Generate sample true and predicted values
    y_true = np.random.exponential(5, n_samples)  # Actual rainfall
    y_pred = y_true + np.random.normal(0, 2, n_samples)  # Predictions with some error

    # Initialize evaluator
    evaluator = ModelEvaluator("demo_rainfall_model")

    # Calculate metrics
    metrics = evaluator.calculate_regression_metrics(y_true, y_pred)

    # Print evaluation report
    evaluator.print_evaluation_report()

    # Save results
    evaluator.save_evaluation_results()

    print("Evaluation demo completed!")


if __name__ == "__main__":
    main()
