import numpy as np
import os
import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

class SocialMediaPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with directory paths"""
        # Get path to current script
        current_file = os.path.abspath(__file__)
        
        # Navigate up to bertopic_project directory
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        # Setup paths
        self.input_dir = os.path.join(project_dir, "data_extraction", "raw")
        self.output_dir = os.path.join(project_dir, "data_preprocessing", "processed_data")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Clean and standardize text content"""
        if pd.isna(text):
            return ""
        
        # Convert to string and clean
        text = str(text).strip()
        
        # Store stock tickers to preserve them
        tickers = re.findall(r'\$[A-Za-z]+', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove Reddit/Twitter-specific formatting
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # Remove []() formatting
        text = re.sub(r'&amp;|&lt;|&gt;', '', text)  # Remove HTML entities
        text = re.sub(r'#\w+', '', text)  # Remove hashtags but keep the text
        text = re.sub(r'@\w+', '', text)  # Remove @ mentions
        
        # Clean special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\$\.]', ' ', text)
        
        # Add preserved tickers back
        for ticker in tickers:
            text = text + ' ' + ticker
        
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_reddit_data(self) -> pd.DataFrame:
        """Process Reddit data"""
        file_path = os.path.join(self.input_dir, "reddit_data.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reddit data file not found at {file_path}")
        
        # Read data
        df = pd.read_csv(file_path)
        
        # Combine title and text
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        df['content'] = df['title'] + ' [TITLE_END] ' + df['text']
        
        # Clean content
        df['content'] = df['content'].apply(self.clean_text)
        
        # Convert and clean date - handling separate date and time columns
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Add source column
        df['source'] = 'reddit'
        
        # Select final columns
        return df[['date', 'content', 'source']]

    def process_twitter_data(self) -> pd.DataFrame:
        """Process Twitter/X data"""
        file_path = os.path.join(self.input_dir, "Tweets_TSLA.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Twitter data file not found at {file_path}")
        
        # Read data
        df = pd.read_csv(file_path)
        
        # Clean content
        df['content'] = df['content'].apply(self.clean_text)
        
        # Convert and clean date - handling ISO format date
        df['date'] = pd.to_datetime(df['date'].str.split('T').str[0]).dt.date
        
        # Add source column
        df['source'] = 'twitter'
        
        # Select final columns
        return df[['date', 'content', 'source']]

    def process_data(self) -> Dict:
        """
        Main method to process all social media data
        
        Returns:
        - Dictionary with processing results and metadata
        """
        try:
            # Process both data sources
            reddit_df = self.process_reddit_data()
            twitter_df = self.process_twitter_data()
            
            # Combine datasets
            combined_df = pd.concat([reddit_df, twitter_df], ignore_index=True)
            
            # Sort by date
            combined_df = combined_df.sort_values('date')
            
            # Remove texts with fewer than 3 words
            combined_df['word_count'] = combined_df['content'].str.split().str.len()
            combined_df = combined_df[combined_df['word_count'] >= 3]
            combined_df = combined_df.drop('word_count', axis=1)
            
            # Save processed data
            output_path = os.path.join(self.output_dir, "processed_social_data.csv")
            combined_df.to_csv(output_path, index=False)
            
            # Return processing results
            return {
                "success": True,
                "total_posts": len(combined_df),
                "reddit_posts": len(combined_df[combined_df['source'] == 'reddit']),
                "twitter_posts": len(combined_df[combined_df['source'] == 'twitter']),
                "date_range": {
                    "start": combined_df['date'].min().strftime("%Y-%m-%d"),
                    "end": combined_df['date'].max().strftime("%Y-%m-%d")
                },
                "file_path": output_path
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Main function to run the preprocessor"""
    preprocessor = SocialMediaPreprocessor()
    results = preprocessor.process_data()
    
    print("\nProcessing Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()