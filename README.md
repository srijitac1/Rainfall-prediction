# Rainfall Prediction Using Linear Regression

This project implements a machine learning system to predict rainfall based on weather parameters (Temperature, Humidity, Pressure, Wind Speed) using Linear Regression.

## Project Structure

- `data_loader.py`: Handles data generation and loading.
- `model.py`: Contains the Linear Regression model class.
- `train.py`: Main script to train and evaluate the model.
- `requirements.txt`: List of dependencies.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the training script to generate synthetic data and train the model:

```bash
python train.py
```