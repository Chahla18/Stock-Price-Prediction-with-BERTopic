import pandas as pd
import numpy as np
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Charge les données nettoyées"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reddit_path = os.path.join(base_dir, 'data', 'data_processed', 'clean_reddit_data.csv')
        stock_path = os.path.join(base_dir, 'data', 'data_processed', 'clean_stock_data.csv')
        
        reddit_df = pd.read_csv(reddit_path)
        stock_df = pd.read_csv(stock_path)
        
        # Conversion des dates
        reddit_df['date'] = pd.to_datetime(reddit_df['date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        print("Données chargées avec succès!")
        return reddit_df, stock_df
    
    except FileNotFoundError as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return None, None

def apply_vader_sentiment(df):
    """Applique l'analyse de sentiment VADER sur les textes"""
    analyzer = SentimentIntensityAnalyzer()
    
    print("Application de l'analyse VADER...")
    
    # Fonction pour obtenir les scores de sentiment
    def get_sentiment_scores(text):
        scores = analyzer.polarity_scores(text)
        return pd.Series({
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
            'vader_compound': scores['compound']
        })
    
    # Application de l'analyse sur chaque texte avec tqdm pour suivre la progression
    sentiment_scores = pd.DataFrame()
    for text in tqdm(df['cleaned_text']):
        scores = get_sentiment_scores(text)
        sentiment_scores = pd.concat([sentiment_scores, scores.to_frame().T], ignore_index=True)
    
    # Ajout des scores au DataFrame
    df = pd.concat([df, sentiment_scores], axis=1)
    
    # Afficher quelques exemples de textes avec leurs scores
    print("\nExemples de scores VADER :")
    for _, row in df.head().iterrows():
        print(f"\nTexte : {row['cleaned_text'][:100]}...")
        print(f"Scores : Neg={row['vader_neg']:.3f}, Neu={row['vader_neu']:.3f}, Pos={row['vader_pos']:.3f}, Compound={row['vader_compound']:.3f}")
    
    return df

def aggregate_daily_sentiment(df):
    """Agrège les scores de sentiment par jour"""
    # Liste des colonnes de sentiment VADER
    vader_cols = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    
    # Agrégation par jour
    daily_sentiment = df.groupby('date').agg({
        **{col: 'mean' for col in vader_cols},
        'cleaned_text': 'count'  # Nombre de posts par jour
    }).rename(columns={'cleaned_text': 'post_count'})
    
    # Afficher les statistiques quotidiennes
    print("\nStatistiques quotidiennes :")
    print(f"Nombre total de jours : {len(daily_sentiment)}")
    print("\nMoyennes des scores par jour :")
    print(daily_sentiment[vader_cols].describe())
    
    return daily_sentiment

def save_sentiment_data(sentiment_df, daily_sentiment):
    """Sauvegarde les données avec les scores de sentiment"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'data_processed')
        
        # Sauvegarde des données complètes avec sentiment
        sentiment_output = os.path.join(output_dir, 'reddit_with_vader_sentiment.csv')
        sentiment_df.to_csv(sentiment_output, index=False)
        
        # Sauvegarde des données agrégées par jour
        daily_output = os.path.join(output_dir, 'daily_vader_sentiment.csv')
        daily_sentiment.to_csv(daily_output)
        
        print(f"\nDonnées sauvegardées avec succès dans :\n{sentiment_output}\n{daily_output}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")

if __name__ == "__main__":
    # 1. Chargement des données
    reddit_df, stock_df = load_processed_data()
    
    if reddit_df is not None:
        # 2. Application de VADER
        reddit_df = apply_vader_sentiment(reddit_df)
        
        # 3. Agrégation quotidienne
        daily_sentiment = aggregate_daily_sentiment(reddit_df)
        
        # 4. Sauvegarde des résultats
        save_sentiment_data(reddit_df, daily_sentiment)