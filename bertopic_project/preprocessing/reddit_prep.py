import numpy as np
import os
import re
import pandas as pd
from datetime import datetime

def create_directory_structure():
    """Create necessary directories for data processing."""
    # Get project root directory (up one level from script location)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    data_raw_dir = os.path.join(data_dir, 'raw')
    data_processed_dir = os.path.join(data_dir, 'processed')
    
    # Create directories if they don't exist
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)
    
    print(f"Directory structure created/verified:")
    print(f"Raw data directory: {data_raw_dir}")
    print(f"Processed data directory: {data_processed_dir}")
    
    return data_raw_dir, data_processed_dir

def process_text(text):
    """
    Clean and process individual text while preserving important features.
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Store stock tickers to preserve them
    tickers = re.findall(r'\$[A-Za-z]+', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove Reddit-specific formatting
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # Remove []() formatting
    text = re.sub(r'&amp;|&lt;|&gt;', '', text)  # Remove HTML entities
    
    # Clean special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\$\.]', ' ', text)
    
    # Add preserved tickers back
    for ticker in tickers:
        text = text + ' ' + ticker
    
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_reddit_data(df):
    """
    Clean and preprocess Reddit data while preserving individual posts.
    """
    print("Starting data cleaning process...")
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Create datetime column and extract date
    print("Converting dates...")
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['date'] = df['datetime'].dt.date
    
    # Set date range
    print("Filtering date range...")
    start_date = pd.to_datetime('2024-01-01').date()
    end_date = pd.to_datetime('2025-01-31').date()
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # Handle missing values
    print("Handling missing values...")
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Clean texts separately
    print("Cleaning text...")
    df['clean_title'] = df['title'].apply(process_text)
    df['clean_text'] = df['text'].apply(process_text)
    
    # Combine with title emphasis and clear separation
    df['processed_text'] = (
        df['clean_title'] + ' ' +        # First instance of title
        df['clean_title'] + ' ' +        # Second instance of title
        '[TITLE_END] ' +                 # Separator
        df['clean_text']                 # Main text content
    )
    
    # Remove texts with fewer than 3 words
    print("Removing short texts...")
    df['word_count'] = df['processed_text'].str.split().str.len()
    df = df[df['word_count'] >= 3]
    
    # Sort by datetime to maintain temporal order
    df = df.sort_values(['datetime', 'ticker'])
    
    # Select and rename final columns
    final_df = df[[
        'date',
        'datetime',
        'ticker',
        'company_name',
        'subreddit',
        'processed_text'
    ]]
    
    print(f"Cleaning complete. Processed {len(final_df)} posts.")
    return final_df

def load_and_process_reddit_data(data_raw_dir):
    """
    Load and process Reddit data from CSV file.
    """
    reddit_path = os.path.join(data_raw_dir, 'reddit_data.csv')
    
    if not os.path.exists(reddit_path):
        print(f"Error: File {reddit_path} not found.")
        return None
    
    print(f"Loading data from {reddit_path}...")
    reddit_df = pd.read_csv(reddit_path)
    print(f"Initial shape: {reddit_df.shape}")
    
    processed_df = clean_reddit_data(reddit_df)
    print(f"Final shape: {processed_df.shape}")
    
    return processed_df

def save_processed_data(reddit_df, data_processed_dir):
    """
    Save processed data and print summary statistics.
    """
    try:
        # Save the processed data
        output_path = os.path.join(data_processed_dir, 'clean_reddit_data.csv')
        reddit_df.to_csv(output_path, index=False)
        print(f"\nData saved successfully to: {output_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total number of posts: {len(reddit_df)}")
        print(f"Date range: {reddit_df['date'].min()} to {reddit_df['date'].max()}")
        print(f"Number of unique tickers: {reddit_df['ticker'].nunique()}")
        print(f"Number of unique subreddits: {reddit_df['subreddit'].nunique()}")
        
        # Print sample
        print("\nSample of processed data (first post):")
        sample = reddit_df.iloc[0]
        print(f"Date: {sample['date']}")
        print(f"Ticker: {sample['ticker']}")
        print(f"Subreddit: {sample['subreddit']}")
        print(f"Processed text excerpt: {sample['processed_text'][:200]}...")
        
    except Exception as e:
        print(f"Error during save: {str(e)}")

def main():
    """
    Main execution function.
    """
    # Create directory structure
    data_raw_dir, data_processed_dir = create_directory_structure()
    
    # Load and process data
    clean_reddit_df = load_and_process_reddit_data(data_raw_dir)
    
    # Save processed data if processing was successful
    if clean_reddit_df is not None:
        save_processed_data(clean_reddit_df, data_processed_dir)

if __name__ == "__main__":
    main()