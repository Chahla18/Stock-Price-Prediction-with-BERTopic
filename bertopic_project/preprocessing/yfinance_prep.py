import numpy as np
import os
import re
import pandas as pd
from datetime import datetime


def create_directory_structure():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    data_raw_dir = os.path.join(data_dir, "raw")
    data_processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)
    print(f"Directory structure created/verified in: {data_processed_dir}")
    return data_raw_dir, data_processed_dir


def add_technical_features(df):
    """
    Add technical indicators exactly as specified in the paper:
    - MA7, MA20 (Moving Averages)
    - MACD (Moving Average Convergence Divergence)
    - 20SD (20-day Standard Deviation)
    - Bollinger Bands
    - EMA (Exponential Moving Average)
    - Log Momentum
    """
    df = df.copy()

    # Moving Averages
    df["MA7"] = df["Close"].rolling(window=7, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()

    # MACD (12-day EMA - 26-day EMA)
    df["MACD"] = (
        df["Close"].ewm(span=12, adjust=False).mean()
        - df["Close"].ewm(span=26, adjust=False).mean()
    )

    # 20-day Standard Deviation
    df["20SD"] = df["Close"].rolling(window=20, min_periods=1).std()

    # Bollinger Bands
    df["upper_band"] = df["MA20"] + (df["20SD"] * 2)
    df["lower_band"] = df["MA20"] - (df["20SD"] * 2)

    # EMA
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Log Momentum
    df["log_momentum"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


def load_and_clean_data(data_raw_dir):
    stock_path = os.path.join(data_raw_dir, "all_stocks_history.csv")

    if not os.path.exists(stock_path):
        print(f"Error: File {stock_path} not found.")
        return None

    stock_df = pd.read_csv(stock_path)
    print(f"Data loaded successfully from:\n{stock_path}")

    # Convert dates
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")

    # Set date range: Jan 1, 2024 to Jan 31, 2025
    start_date, end_date = pd.to_datetime("2024-01-01"), pd.to_datetime("2025-01-31")
    stock_df = stock_df[
        (stock_df["Date"] >= start_date) & (stock_df["Date"] <= end_date)
    ]

    def clean_stock_data(df):
        df = df.copy()
        # Clean Volume data
        df["Volume"] = df["Volume"].replace({",": ""}, regex=True).astype(float)

        # Handle missing and infinite values
        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Sort by date
        df.sort_values("Date", inplace=True)

        # Add technical features
        return add_technical_features(df)

    return clean_stock_data(stock_df)


def save_cleaned_data(stock_df, data_processed_dir):
    try:
        stock_output = os.path.join(data_processed_dir, "cleaned_stock_data.csv")
        stock_df.to_csv(stock_output, index=False)
        print(f"Data saved successfully to:\n{stock_output}")
    except Exception as e:
        print(f"Error during save: {e}")


if __name__ == "__main__":
    data_raw_dir, data_processed_dir = create_directory_structure()
    clean_stock_df = load_and_clean_data(data_raw_dir)
    if clean_stock_df is not None:
        save_cleaned_data(clean_stock_df, data_processed_dir)
