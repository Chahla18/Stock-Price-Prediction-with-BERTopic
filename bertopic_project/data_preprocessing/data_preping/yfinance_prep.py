import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict

class StockDataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with directory paths"""
        # Get path to current script
        current_file = os.path.abspath(__file__)
        
        # Navigate up to bertopic_project directory (3 levels up from the script)
        # From: bertopic_project/data_preprocessing/data_preping/yfinance_prep.py
        # To: bertopic_project/
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        # Setup input/output paths
        self.data_extraction_dir = os.path.join(project_dir, "data_extraction")
        self.data_preprocessing_dir = os.path.join(project_dir, "data_preprocessing")
        
        # Complete paths
        self.raw_dir = os.path.join(self.data_extraction_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_preprocessing_dir, "processed_data")
        
        # Ensure output directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # File paths
        self.stock_file = os.path.join(self.raw_dir, "tesla_stock_history.csv")
        self.processed_file = os.path.join(self.processed_data_dir, "processed_stock_data.csv")
        
        print(f"Reading from: {self.stock_file}")
        print(f"Writing to: {self.processed_file}")

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the stock data
        
        Parameters:
        - df: DataFrame with stock price data
        
        Returns:
        - DataFrame with added technical indicators
        """
        df = df.copy()

        # Moving Averages
        df["MA7"] = df["Close"].rolling(window=7, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()

        # MACD
        df["MACD"] = (
            df["Close"].ewm(span=12, adjust=False).mean()
            - df["Close"].ewm(span=26, adjust=False).mean()
        )

        # 20-day Standard Deviation
        df["20SD"] = df["Close"].rolling(window=20, min_periods=1).std()

        # Bollinger Bands
        df["Upper_Band"] = df["MA20"] + (df["20SD"] * 2)
        df["Lower_Band"] = df["MA20"] - (df["20SD"] * 2)

        # EMA
        df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()

        # Log Momentum
        df["Log_Momentum"] = np.log(df["Close"] / df["Close"].shift(1))

        return df

    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess stock data
        
        Parameters:
        - df: Raw stock DataFrame
        
        Returns:
        - Cleaned DataFrame with technical indicators
        """
        df = df.copy()
        
        # Convert Date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # Clean numeric columns
        numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        # Handle missing and infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Sort by date
        df.sort_values("Date", inplace=True)

        # Add technical features
        df = self.add_technical_features(df)

        return df

    def process_data(self) -> Dict:
        """
        Main method to process stock data
        
        Returns:
        - Dictionary with processing results and metadata
        """
        try:
            # Load data
            if not os.path.exists(self.stock_file):
                return {"success": False, "error": f"File not found: {self.stock_file}"}

            df = pd.read_csv(self.stock_file)
            
            # Clean and process data
            df = self.clean_stock_data(df)

            # Save processed data
            df.to_csv(self.processed_file, index=False)

            # Return processing results
            return {
                "success": True,
                "rows_processed": len(df),
                "date_range": {
                    "start": df["Date"].min().strftime("%Y-%m-%d"),
                    "end": df["Date"].max().strftime("%Y-%m-%d")
                },
                "columns": list(df.columns),
                "file_path": self.processed_file
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Main function to run the preprocessor"""
    preprocessor = StockDataPreprocessor()
    results = preprocessor.process_data()
    print("\nProcessing Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()