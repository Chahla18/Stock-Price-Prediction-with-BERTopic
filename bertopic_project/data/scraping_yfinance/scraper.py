import os
import pandas as pd
from .trending_tickers import get_trending_tickers
from .historical_data import get_stock_history

def setup_directories():
    """Create necessary directories if they don't exist"""
    # Get the absolute path of the current script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the data directory
    data_dir = os.path.dirname(base_dir)
    
    dirs = ['raw', 'processed']
    created_dirs = {}
    for dir_name in dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        created_dirs[dir_name] = dir_path
    return created_dirs['raw']

def main():
    # Setup directories
    raw_dir = setup_directories()
    
    # Get trending stocks
    print("Fetching trending stocks...")
    trending_stocks = get_trending_tickers()
    
    # Display available stocks
    print("\nTrending Stocks:")
    for ticker, name in trending_stocks.items():
        print(f"{ticker}: {name}")
    
    # Get user input
    print("\nEnter the tickers you want to analyze (separated by spaces):")
    selected_tickers = input().upper().split()
    
    # Validate user input
    selected_stocks = {}
    for ticker in selected_tickers:
        if ticker in trending_stocks:
            selected_stocks[ticker] = trending_stocks[ticker]
        else:
            print(f"Warning: {ticker} is not in trending list but will be included")
            selected_stocks[ticker] = "Unknown"
    
    # Get historical data
    print("\nFetching historical data...")
    stock_data = get_stock_history(selected_stocks, raw_dir)
    
    # Merge all data
    print("\nMerging data...")
    if stock_data and len(stock_data) > 0:  # Check if we have any data
        dfs_to_merge = [df for df in stock_data.values() if not df.empty]
        if dfs_to_merge:  # If we have any non-empty DataFrames
            merged_data = pd.concat(dfs_to_merge, ignore_index=True)
            merged_path = os.path.join(raw_dir, "all_stocks_history.csv")
            merged_data.to_csv(merged_path, index=False)
            print(f"\nData has been saved to {merged_path}")
            
            # Display summary
            print("\nSummary of data collected:")
            for ticker, df in stock_data.items():
                if not df.empty:
                    print(f"{ticker}: {len(df)} days of data")
            print(f"Total combined records: {len(merged_data)}")
        else:
            print("No valid data was collected from any stock")
    else:
        print("No data was collected")

if __name__ == "__main__":
    main()