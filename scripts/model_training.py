import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from preprosessing_data import preprocess_data


# Split data into training and testing sets
def split_data(stock_data):
    # Drop rows with NaN values created during feature engineering
    stock_data.dropna(inplace=True)

    # Define input features (X) and target variable (y)
    X = stock_data.drop(columns=["Close"])
    y = stock_data["Close"]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print("Data splitting completed:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# Train and evaluate the model
def train_model(X_train, y_train, X_test, y_test):
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    plot_predictions(y_test, y_pred)


# Improved visualization
def plot_predictions(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.values, label="Actual", color="blue", linewidth=1.5)
    plt.plot(
        predicted, label="Predicted", color="red", linestyle="dashed", linewidth=1.5
    )
    plt.title("Stock Price Prediction - Actual vs Predicted", fontsize=14)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Normalized Stock Price", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=12)
    plt.show()


if __name__ == "__main__":
    # File paths
    input_file = "../data/raw/AAPL_stock_data.csv"
    output_file = "../data/processed/AAPL_preprocessed_data.csv"

    # Preprocess data
    stock_data = preprocess_data(input_file, output_file)

    # Split data
    X_train, X_test, y_train, y_test = split_data(stock_data)

    # Train and evaluate the model
    train_model(X_train, y_train, X_test, y_test)
