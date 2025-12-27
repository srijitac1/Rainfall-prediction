#!/usr/bin/env python3
"""
Prediction Module for Rainfall Prediction

This module provides functionality to make rainfall predictions using trained models
with both command-line interface and programmatic access.
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import os
import sys


class RainfallPredictor:
    """
    A predictor class for rainfall prediction using trained models.
    """

    def __init__(self, model_path: str = "results/trained_model.pkl",
                 scaler_path: str = "results/scaler.pkl"):
        """
        Initialize the predictor.

        Args:
            model_path (str): Path to the trained model file
            scaler_path (str): Path to the feature scaler file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            'temperature', 'humidity', 'pressure', 'wind_speed',
            'cloud_cover', 'visibility'
        ]

        # Load the model and scaler
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the trained model and scaler from disk.
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file not found at {self.model_path}")
            print("Please train the model first using: python train.py")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        try:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {self.scaler_path}")
        except FileNotFoundError:
            print(f"Warning: Scaler file not found at {self.scaler_path}")
            print("Feature scaling will be skipped.")
            self.scaler = None
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None

    def predict_single(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Make a prediction for a single set of weather features.

        Args:
            features (Dict[str, float]): Weather features dictionary

        Returns:
            Dict[str, float]: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")

        # Validate input features
        self._validate_features(features)

        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([features])

        # Make prediction
        prediction = self._predict(df)

        # Prepare result
        result = {
            'predicted_rainfall': float(prediction[0]),
            'input_features': features,
            'prediction_confidence': self._estimate_confidence(features, prediction[0])
        }

        return result

    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Make predictions for multiple sets of weather features.

        Args:
            features_list (List[Dict[str, float]]): List of weather features dictionaries

        Returns:
            List[Dict[str, float]]: List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")

        results = []
        for features in features_list:
            try:
                result = self.predict_single(features)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input_features': features
                })

        return results

    def _predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Internal prediction method.

        Args:
            features_df (pd.DataFrame): Features DataFrame

        Returns:
            np.ndarray: Predictions
        """
        # Ensure correct feature order
        X = features_df[self.feature_names].values

        # Apply scaling if available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Make prediction
        predictions = self.model.predict(X)

        return predictions

    def _validate_features(self, features: Dict[str, float]) -> None:
        """
        Validate input features.

        Args:
            features (Dict[str, float]): Features to validate

        Raises:
            ValueError: If features are invalid
        """
        missing_features = []
        for feature in self.feature_names:
            if feature not in features:
                missing_features.append(feature)

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Validate feature ranges (reasonable weather values)
        validations = {
            'temperature': (-50, 60),    # Celsius
            'humidity': (0, 100),        # Percentage
            'pressure': (900, 1100),     # hPa
            'wind_speed': (0, 200),      # km/h
            'cloud_cover': (0, 100),     # Percentage
            'visibility': (0, 50)        # km
        }

        for feature, (min_val, max_val) in validations.items():
            value = features[feature]
            if not (min_val <= value <= max_val):
                print(f"Warning: {feature} value {value} is outside typical range [{min_val}, {max_val}]")

    def _estimate_confidence(self, features: Dict[str, float], prediction: float) -> str:
        """
        Estimate prediction confidence based on feature values and prediction.

        Args:
            features (Dict[str, float]): Input features
            prediction (float): Predicted rainfall

        Returns:
            str: Confidence level description
        """
        # Simple confidence estimation based on feature reliability
        # High confidence when features are in typical ranges and prediction is reasonable

        confidence_score = 0

        # Temperature in comfortable range
        if 0 <= features['temperature'] <= 35:
            confidence_score += 1

        # Humidity in reasonable range
        if 20 <= features['humidity'] <= 90:
            confidence_score += 1

        # Pressure in normal range
        if 980 <= features['pressure'] <= 1030:
            confidence_score += 1

        # Reasonable prediction value
        if 0 <= prediction <= 100:
            confidence_score += 1

        # Visibility affects confidence
        if features['visibility'] > 5:
            confidence_score += 1

        # Map score to confidence level
        if confidence_score >= 4:
            return "High"
        elif confidence_score >= 3:
            return "Medium"
        elif confidence_score >= 2:
            return "Low"
        else:
            return "Very Low"

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available from the model.

        Returns:
            Optional[Dict[str, float]]: Feature importance dictionary or None
        """
        if self.model is None:
            return None

        # Try to get feature importance (works for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return importance_dict
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_)))
            return importance_dict
        else:
            return None


def create_sample_predictions() -> List[Dict[str, float]]:
    """
    Create sample weather conditions for demonstration.

    Returns:
        List[Dict[str, float]]: Sample weather conditions
    """
    samples = [
        # Sunny day
        {
            'temperature': 25.0,
            'humidity': 45.0,
            'pressure': 1015.0,
            'wind_speed': 10.0,
            'cloud_cover': 20.0,
            'visibility': 15.0
        },
        # Rainy day
        {
            'temperature': 18.0,
            'humidity': 85.0,
            'pressure': 995.0,
            'wind_speed': 25.0,
            'cloud_cover': 90.0,
            'visibility': 3.0
        },
        # Storm conditions
        {
            'temperature': 12.0,
            'humidity': 95.0,
            'pressure': 980.0,
            'wind_speed': 45.0,
            'cloud_cover': 100.0,
            'visibility': 1.0
        },
        # Dry desert-like
        {
            'temperature': 35.0,
            'humidity': 15.0,
            'pressure': 1020.0,
            'wind_speed': 20.0,
            'cloud_cover': 5.0,
            'visibility': 25.0
        }
    ]

    return samples


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Rainfall Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python predict.py --temp 25 --humidity 70 --pressure 1013 --wind 15 --clouds 50 --visibility 10

  # Batch prediction with JSON file
  python predict.py --input-file weather_data.json --output-file predictions.json

  # Interactive mode
  python predict.py --interactive
        """
    )

    # Single prediction arguments
    parser.add_argument('--temp', type=float, help='Temperature in Celsius')
    parser.add_argument('--humidity', type=float, help='Humidity in percentage')
    parser.add_argument('--pressure', type=float, help='Atmospheric pressure in hPa')
    parser.add_argument('--wind', type=float, help='Wind speed in km/h')
    parser.add_argument('--clouds', type=float, help='Cloud cover in percentage')
    parser.add_argument('--visibility', type=float, help='Visibility in km')

    # Batch prediction arguments
    parser.add_argument('--input-file', type=str, help='JSON file with weather data for batch prediction')
    parser.add_argument('--output-file', type=str, help='Output file for batch predictions')

    # Other options
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true', help='Run demonstration with sample data')
    parser.add_argument('--model-path', type=str, default='results/trained_model.pkl',
                       help='Path to trained model file')
    parser.add_argument('--scaler-path', type=str, default='results/scaler.pkl',
                       help='Path to scaler file')

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = RainfallPredictor(args.model_path, args.scaler_path)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return

    # Handle different modes
    if args.demo:
        # Run demonstration
        print("=== Rainfall Prediction Demo ===")
        samples = create_sample_predictions()

        print("Sample Weather Conditions:")
        conditions = ["Sunny Day", "Rainy Day", "Storm Conditions", "Dry Desert"]
        for i, (sample, condition) in enumerate(zip(samples, conditions)):
            print(f"\n{i+1}. {condition}:")
            for key, value in sample.items():
                print(f"   {key}: {value}")

        print("\nPredictions:")
        for i, (sample, condition) in enumerate(zip(samples, conditions)):
            try:
                result = predictor.predict_single(sample)
                print(f"\n{i+1}. {condition}:")
                print(f"   Predicted rainfall: {result['predicted_rainfall']:.2f} mm")
                print(f"   Confidence: {result['prediction_confidence']}")
            except Exception as e:
                print(f"\n{i+1}. {condition}: Error - {e}")

    elif args.interactive:
        # Interactive mode
        print("=== Interactive Rainfall Prediction ===")
        print("Enter weather conditions (press Enter for default values):")

        features = {}
        defaults = {
            'temperature': 20.0,
            'humidity': 60.0,
            'pressure': 1013.0,
            'wind_speed': 10.0,
            'cloud_cover': 50.0,
            'visibility': 10.0
        }

        for feature in predictor.feature_names:
            while True:
                try:
                    value = input(f"{feature} ({defaults[feature]}): ").strip()
                    if value == "":
                        features[feature] = defaults[feature]
                    else:
                        features[feature] = float(value)
                    break
                except ValueError:
                    print("Please enter a valid number.")

        try:
            result = predictor.predict_single(features)
            print("
Prediction Results:")
            print(f"Predicted rainfall: {result['predicted_rainfall']:.2f} mm")
            print(f"Confidence level: {result['prediction_confidence']}")
        except Exception as e:
            print(f"Error making prediction: {e}")

    elif args.input_file:
        # Batch prediction from file
        try:
            with open(args.input_file, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                results = predictor.predict_batch(data)
            else:
                results = [predictor.predict_single(data)]

            # Save results
            output_file = args.output_file or 'predictions_output.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Batch predictions completed. Results saved to {output_file}")

        except Exception as e:
            print(f"Error in batch prediction: {e}")

    elif any([args.temp, args.humidity, args.pressure, args.wind, args.clouds, args.visibility]):
        # Single prediction from command line
        # Check if all required features are provided
        provided_features = {
            'temperature': args.temp,
            'humidity': args.humidity,
            'pressure': args.pressure,
            'wind_speed': args.wind,
            'cloud_cover': args.clouds,
            'visibility': args.visibility
        }

        missing = [k for k, v in provided_features.items() if v is None]
        if missing:
            print(f"Error: Missing required features: {missing}")
            print("Use --help for usage information.")
            return

        try:
            result = predictor.predict_single(provided_features)
            print("Rainfall Prediction Results:")
            print(f"Predicted rainfall: {result['predicted_rainfall']:.2f} mm")
            print(f"Confidence level: {result['prediction_confidence']}")

            # Show feature importance if available
            importance = predictor.get_feature_importance()
            if importance:
                print("\nFeature Importance:")
                for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {imp:.4f}")

        except Exception as e:
            print(f"Error making prediction: {e}")

    else:
        # No arguments provided, show help
        parser.print_help()


if __name__ == "__main__":
    main()
