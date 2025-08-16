# train.py

import os
import warnings
import sys
import argparse  # For handling command-line arguments

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

# Set up a logger
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Function to calculate evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # --- 1. DATA PREPARATION ---
    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        sys.exit(1)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    # --- 2. TRAINING & MLFLOW TRACKING ---
    # Get parameters from command line arguments (if provided)
    # This allows you to run experiments like: python train.py 0.7 0.7
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # This is the core of MLflow: starting a "run"
    # Everything logged inside this 'with' block will be associated with this single experiment run.
    with mlflow.start_run():
        print(f"--- Starting a new MLflow run with alpha={alpha} and l1_ratio={l1_ratio} ---")
        
        # Create and train the ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Make predictions on the test set
        predicted_qualities = lr.predict(test_x)

        # Evaluate the model
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print the metrics to the console
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # --- 3. LOGGING TO MLFLOW ---
        # Log the parameters you used for this run
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log the metrics you calculated
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log the trained model itself
        # This is super powerful! It saves the model, its version, dependencies, etc.
        mlflow.sklearn.log_model(lr, "model")

        print("--- Run complete. Check the MLflow UI. ---")