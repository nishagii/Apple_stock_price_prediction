import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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
    stock_data = pd.read_csv(input_file,index_col="Date",parse_dates=True)

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
    stock_data["SMA_7"] = stock_data["Close"].rolling(window=7).mean()
    stock_data["SMA_30"] = stock_data["Close"].rolling(window=30).mean()
    stock_data["EMA_7"] = stock_data["Close"].ewm(span=7, adjust=False).mean()
    stock_data["Volatility_7"] = stock_data["Close"].rolling(window=7).std()
    stock_data["RSI"] = compute_rsi(stock_data["Close"])
    stock_data["Close_Lag_1"] = stock_data["Close"].shift(1)
    stock_data["Close_Lag_2"] = stock_data["Close"].shift(2)
    stock_data["Daily_Return"] = stock_data["Close"].pct_change()
    stock_data["EMA_12"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
    stock_data["EMA_26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()
    stock_data["MACD"] = stock_data["EMA_12"] - stock_data["EMA_26"]
    stock_data["Signal_Line"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()
    stock_data["Histogram"] = stock_data["MACD"] - stock_data["Signal_Line"]

    # Save preprocessed data
    stock_data.to_csv(output_file)
    print(
        f"Data preprocessing and feature engineering completed! Preprocessed data saved to {output_file}"
    )

    return stock_data


# Plotting Function
def plot_data(stock_data):
    plt.figure(figsize=(14, 7))

    # Price chart
    plt.subplot(2, 1, 1)
    plt.plot(stock_data.index, stock_data["Close"], label="Close Price", color="blue")
    plt.title("Stock Price")
    plt.legend()

    # MACD chart
    plt.subplot(2, 1, 2)
    plt.plot(stock_data.index, stock_data["MACD"], label="MACD Line", color="green")
    plt.plot(
        stock_data.index, stock_data["Signal_Line"], label="Signal Line", color="red"
    )
    plt.bar(
        stock_data.index,
        stock_data["Histogram"],
        label="Histogram",
        color="gray",
        alpha=0.5,
    )
    plt.title("MACD")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Run the script
if __name__ == "__main__":
    # Input and output file paths
    input_file = "../data/raw/AAPL_stock_data.csv"
    output_file = "../data/processed/AAPL_preprocessed_data.csv"

    # Preprocess data
    stock_data = preprocess_data(input_file, output_file)

    # Plot data
    plot_data(stock_data)

    # Print first few rows for verification
    print(stock_data.head(100))
