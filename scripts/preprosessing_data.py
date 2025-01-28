import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Define the RSI function
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Preprocess and Feature Engineer the Data
def preprocess_data(input_file, output_file):
    # Load data
    stock_data = pd.read_csv(input_file, index_col="Date", parse_dates=True)

    # Handle missing values
    stock_data.fillna(method="ffill", inplace=True)

    # Keep only the columns we need
    stock_data = stock_data[["Open", "High", "Low", "Close", "Volume"]]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    stock_data = pd.DataFrame(
        scaled_data, columns=stock_data.columns, index=stock_data.index
    )

    # Feature Engineering
   
    stock_data["SMA_30"] = stock_data["Close"].rolling(window=30).mean()
    stock_data["RSI"] = compute_rsi(stock_data["Close"])
    stock_data["Daily_Return"] = stock_data["Close"].pct_change()
    # Fix Daily_Return: Replace Inf/-Inf with NaN, then fill NaN with 0
    stock_data["Daily_Return"].replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_data["Daily_Return"].fillna(0, inplace=True)
    stock_data["EMA_12"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
    stock_data["EMA_26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()
    stock_data["MACD"] = stock_data["EMA_12"] - stock_data["EMA_26"]
    stock_data["Signal_Line"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()
    stock_data["Histogram"] = stock_data["MACD"] - stock_data["Signal_Line"]

    # Now check for feature redundancy using correlation matrix
    correlation_matrix = stock_data.corr()

    # Set a threshold for high correlation (e.g., 0.8 or -0.8)
    threshold = 0.8

    # Find pairs of highly correlated features
    highly_correlated_pairs = []
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if abs(correlation_matrix.loc[row, col]) > threshold and row != col:
                highly_correlated_pairs.append((row, col, correlation_matrix.loc[row, col]))

    # Print the highly correlated feature pairs
    print("Highly Correlated Feature Pairs (|correlation| > 0.8):")
    for pair in highly_correlated_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]}")

    # Save preprocessed data
    stock_data.to_csv(output_file)
    print(
        f"Data preprocessing and feature engineering completed! Preprocessed data saved to {output_file}"
    )

    return stock_data


# Example usage
if __name__ == "__main__":
    input_file = "../data/raw/AAPL_stock_data.csv"
    output_file = "../data/processed/AAPL_preprocessed_data.csv"
    preprocess_data(input_file, output_file)
