import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
from typing import Dict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self):
        self.stocks = {
            'NVDA': 'NVIDIA',
            'TSLA': 'Tesla',
            'AAPL': 'Apple',
            'MC.PA': 'LVMH'
        }
    
    def fetch_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock data and return a clean DataFrame.
        """
        all_data = []
        
        for ticker, company_name in self.stocks.items():
            try:
                logger.info(f"Fetching data for {ticker}...")
                
                # Fetch data for single stock
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # Clean up the DataFrame
                df = df.reset_index()  # Make Date a column
                df = df.droplevel(1, axis=1)  # Remove the ticker level from columns
                
                # Add identifier columns
                df['Ticker'] = ticker
                df['Stock Name'] = company_name
                
                all_data.append(df)
                logger.info(f"Successfully fetched data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data was fetched for any stock")
            
        # Combine all stock data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns to desired format
        column_order = ['Date', 'Ticker', 'Stock Name', 'Open', 'High', 'Low', 'Close', 'Volume']
        combined_df = combined_df[column_order]
        
        # Sort by Date and Ticker
        combined_df = combined_df.sort_values(['Date', 'Ticker'])
        
        return combined_df

def main():
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # Set date range
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    try:
        # Fetch data
        df = fetcher.fetch_stock_data(start_date, end_date)
        
        # Save to CSV
        df.to_csv('data/yfinance_data.csv', index=False)
        
        # Display first few rows
        print("\nFirst few rows of the data:")
        print(df.head())
        
        # Display basic information
        print("\nDataFrame Info:")
        print(df.info())
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()