import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import generate_synthetic_data, load_data
from model import RainfallModel

def main():
    # 1. Data Preparation
    data_path = 'data/sample_data.csv'
    
    if not os.path.exists(data_path):
        print("Data file not found. Generating synthetic data...")
        df = generate_synthetic_data(save_path=data_path)
    else:
        print(f"Loading data from {data_path}...")
        df = load_data(data_path)

    # Select features and target
    features = ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']
    target = 'Rainfall'
    
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Model Training
    model = RainfallModel()
    print("Training Linear Regression model...")
    model.train(X_train, y_train)

    # 3. Evaluation
    metrics = model.evaluate(X_test, y_test)
    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 4. Save Model
    os.makedirs('results', exist_ok=True)
    model_path = 'results/rainfall_model.pkl'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()