import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


# Function to fetch and save stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    print(f"Fetching data for {stock_symbol}...")
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Specify the preferred save location
    preferred_save_location = (
        "../data/raw"
    )

    # Ensure the directory exists
    if not os.path.exists(preferred_save_location):
        os.makedirs(preferred_save_location)

    # Save data to CSV
    csv_filename = os.path.join(
        preferred_save_location, f"{stock_symbol}_stock_data.csv"
    )
    stock_data.to_csv(csv_filename)
    print(f"Data saved as {csv_filename}")

    return stock_data


# Function to plot stock data
def plot_stock_data(stock_data, stock_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data["Close"], label=f"{stock_symbol} Closing Price", color="blue")
    plt.title(f"{stock_symbol} Stock Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Inputs
    stock_symbol = "AAPL"
    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch and save data for the main stock
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

    # Plot the data for the main stock
    plot_stock_data(stock_data, stock_symbol)

    # Fetch and process multiple stocks
    stock_symbols = ["AAPL", "MSFT", "GOOG"]
    for symbol in stock_symbols:
        print(f"Processing {symbol}...")
        data = fetch_stock_data(symbol, start_date, end_date)
        plot_stock_data(data, symbol)
