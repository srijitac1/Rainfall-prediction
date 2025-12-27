import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=1000, save_path='data/sample_data.csv'):
    """
    Generates a synthetic dataset for rainfall prediction.
    """
    np.random.seed(42)
    
    # Generate features
    # Temperature in Celsius (mean 25, std 5)
    temperature = np.random.normal(25, 5, n_samples)
    # Humidity in % (mean 60, std 10)
    humidity = np.random.normal(60, 10, n_samples)
    # Pressure in hPa (mean 1013, std 5)
    pressure = np.random.normal(1013, 5, n_samples)
    # Wind Speed in km/h (mean 10, std 5)
    wind_speed = np.abs(np.random.normal(10, 5, n_samples))

    # Generate target (Rainfall in mm)
    # Formula: Base + coeff*Hum - coeff*Press + coeff*Temp + Noise
    # We assume high humidity and low pressure correlate with rainfall.
    rainfall = 200 + (0.8 * humidity) - (0.2 * pressure) + (0.1 * temperature) + np.random.normal(0, 2, n_samples)
    rainfall = np.maximum(rainfall, 0) # Ensure no negative rainfall

    df = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'Pressure': pressure,
        'Wind_Speed': wind_speed,
        'Rainfall': rainfall
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Synthetic data generated and saved to {save_path}")

    return df

def load_data(filepath):
    return pd.read_csv(filepath)