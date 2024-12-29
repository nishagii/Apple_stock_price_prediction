# Apple Stock Price Prediction Project

## Project Overview
This project aims to analyze and predict stock prices for multiple companies (e.g., AAPL, GOOG, MSFT). The pipeline includes data collection, preprocessing, feature engineering, and model training for stock price prediction. The repository is organized for scalability and ease of use.

## Folder Structure
```
|-- data
|   |-- raw
|   |   |-- AAPL_stock_data.csv
|   |   |-- GOOG_stock_data.csv
|   |   |-- MSFT_stock_data.csv
|   |-- processed
|       |-- AAPL_preprocessed_data.csv
|
|-- models
|
|-- results
|
|-- scripts
|   |-- stock_data_collector.py
|   |-- preprocessing_data.py
|   |-- model_training.py
|
|-- venv
|
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
```

## Steps to Use the Project

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/nishagii/Apple_stock_price_prediction.git
cd <repository-folder>
```

### 2. Set Up the Environment
Create and activate a virtual environment, then install the dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

### 3. Data Collection
To collect raw stock data, run the `stock_data_collector.py` script:
```bash
python scripts/stock_data_collector.py
```
This will fetch stock data for companies like AAPL, GOOG, and MSFT and store it in the `data/raw` folder.

### 4. Preprocessing Data
Clean and preprocess the raw data by running:
```bash
python scripts/preprocessing_data.py
```
This will generate preprocessed data for the given company and file will be stored in the `data/processed` folder.

### 5. Model Training
Train the prediction model using the processed data:
```bash
python scripts/model_training.py
```
This will save the trained model in the `models` folder and produce results stored in the `results` folder.

### 6. Analyze Results
The results folder contains visualizations and metrics to evaluate the model's performance.

## Dependencies
The required libraries and versions are specified in `requirements.txt`. Key dependencies include:
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- yfinance
- xgboost

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Notes
- Ensure you have an active internet connection for fetching data using the Yahoo Finance API.
- Modify the `stock_data_collector.py` script to include additional stocks or adjust date ranges.
- Feel free to customize preprocessing and feature engineering steps in `preprocessing_data.py` to suit your project needs.

## License
This project is licensed under the [MIT License](LICENSE).

