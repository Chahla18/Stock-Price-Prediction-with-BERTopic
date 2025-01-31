import numpy as np
from datetime import datetime
import re
import os
import pandas as pd

def create_directory_structure():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_processed_dir = os.path.join(base_dir, 'data', 'data_processed')
    os.makedirs(data_processed_dir, exist_ok=True)
    
    print(f"Structure de dossiers créée/vérifiée dans : {base_dir}")
    return data_processed_dir

def add_technical_features(df):
    df = df.copy()
    df['MA7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14, min_periods=1).mean() / df['Close'].diff().apply(lambda x: -min(x, 0)).rolling(window=14, min_periods=1).mean()))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['20SD'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['upper_band'] = df['MA20'] + (df['20SD'] * 2)
    df['lower_band'] = df['MA20'] - (df['20SD'] * 2)
    
    return df

def load_and_clean_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stock_path = os.path.join(base_dir, 'data', 'raw', 'all_stocks_history.csv')
    reddit_path = os.path.join(base_dir, 'data', 'raw', 'reddit_data.csv')

    try:
        stock_df = pd.read_csv(stock_path)
        reddit_df = pd.read_csv(reddit_path)
        print(f"Données chargées avec succès depuis :\n{stock_path}\n{reddit_path}")
    except FileNotFoundError as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return None, None
    
    reddit_df['date'] = pd.to_datetime(reddit_df['date'], format='mixed', errors='coerce')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='mixed', errors='coerce')
    
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-31')

    reddit_df = reddit_df[(reddit_df['date'] >= start_date) & (reddit_df['date'] <= end_date)]
    stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)]
    
    def clean_stock_data(df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df.sort_values('Date')
        df = add_technical_features(df)
        return df

    def clean_reddit_data(df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r'http\S+|www.\S+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()
        
        df['clean_title'] = df['title'].apply(clean_text)
        df['clean_text'] = df['text'].apply(clean_text)
        df['cleaned_text'] = df['clean_title'] + " " + df['clean_text']
        df = df.drop_duplicates(subset=['cleaned_text', 'date'])
        df = df[df['cleaned_text'].str.len() > 0]
        df = df.drop(['title', 'text', 'clean_title', 'clean_text'], axis=1)
        
        return df

    print("\nDébut du nettoyage des données...")
    clean_stock_df = clean_stock_data(stock_df)
    clean_reddit_df = clean_reddit_data(reddit_df)

    return clean_stock_df, clean_reddit_df

def save_cleaned_data(stock_df, reddit_df, data_processed_dir):
    try:
        stock_output = os.path.join(data_processed_dir, 'clean_stock_data.csv')
        reddit_output = os.path.join(data_processed_dir, 'clean_reddit_data.csv')
        stock_df.to_csv(stock_output, index=False)
        reddit_df.to_csv(reddit_output, index=False)
        print(f"\nDonnées sauvegardées avec succès dans :\n{stock_output}\n{reddit_output}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")

if __name__ == "__main__":
    data_processed_dir = create_directory_structure()
    clean_stock_df, clean_reddit_df = load_and_clean_data()
    if clean_stock_df is not None and clean_reddit_df is not None:
        save_cleaned_data(clean_stock_df, clean_reddit_df, data_processed_dir)