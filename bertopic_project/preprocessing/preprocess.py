import numpy as np
import os
import re
import pandas as pd
from datetime import datetime

def create_directory_structure():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Aller un niveau au-dessus
    data_dir = os.path.join(base_dir, 'data')
    data_raw_dir = os.path.join(data_dir, 'raw')
    data_processed_dir = os.path.join(data_dir, 'data_preprocessed')  # Correction du dossier
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)
    print(f"Structure de dossiers créée/vérifiée dans : {data_processed_dir}")
    return data_raw_dir, data_processed_dir

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

def load_and_clean_data(data_raw_dir):
    stock_path = os.path.join(data_raw_dir, 'all_stocks_history.csv')
    reddit_path = os.path.join(data_raw_dir, 'reddit_data.csv')
    
    if not os.path.exists(stock_path):
        print(f"Erreur : Le fichier {stock_path} est introuvable.")
        return None, None
    if not os.path.exists(reddit_path):
        print(f"Erreur : Le fichier {reddit_path} est introuvable.")
        return None, None

    stock_df = pd.read_csv(stock_path)
    reddit_df = pd.read_csv(reddit_path)
    print(f"Données chargées avec succès depuis :\n{stock_path}\n{reddit_path}")
    
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
    reddit_df['date'] = pd.to_datetime(reddit_df['date'], errors='coerce')
    
    start_date, end_date = pd.to_datetime('2024-10-01'), pd.to_datetime('2024-12-31')
    stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)]
    reddit_df = reddit_df[(reddit_df['date'] >= start_date) & (reddit_df['date'] <= end_date)]
    
    def clean_stock_data(df):
        df = df.copy()
        df['Volume'] = df['Volume'].replace({',': ''}, regex=True).astype(float)
        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.sort_values('Date', inplace=True)
        return add_technical_features(df)
    
    def clean_reddit_data(df):
        df = df.copy()
        def clean_text(text):
            if pd.isna(text): return ""
            text = re.sub(r'http\S+|www.\S+', '', str(text))
            text = re.sub(r'[^\w\s]', '', text).lower().strip()
            return re.sub(r'\s+', ' ', text)
        
        df['clean_text'] = (df['title'].apply(clean_text) + " " + df['text'].apply(clean_text)).str.strip()
        df.drop_duplicates(subset=['clean_text', 'date'], inplace=True)
        df = df[df['clean_text'].str.len() > 0]
        return df.drop(['title', 'text'], axis=1)
    
    return clean_stock_data(stock_df), clean_reddit_data(reddit_df)

def save_cleaned_data(stock_df, reddit_df, data_processed_dir):
    try:
        stock_output, reddit_output = os.path.join(data_processed_dir, 'clean_stock_data.csv'), os.path.join(data_processed_dir, 'clean_reddit_data.csv')
        stock_df.to_csv(stock_output, index=False)
        reddit_df.to_csv(reddit_output, index=False)
        print(f"Données sauvegardées avec succès :\n{stock_output}\n{reddit_output}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")

if __name__ == "__main__":
    data_raw_dir, data_processed_dir = create_directory_structure()
    clean_stock_df, clean_reddit_df = load_and_clean_data(data_raw_dir)
    if clean_stock_df is not None and clean_reddit_df is not None:
        save_cleaned_data(clean_stock_df, clean_reddit_df, data_processed_dir)
