import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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

def apply_finbert_sentiment(df):
    """Applique l'analyse de sentiment FinBERT sur les textes"""
    # Chargement du modèle FinBERT
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    print("Application de l'analyse FinBERT...")
    
    def get_finbert_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return pd.Series({
            'finbert_negative': predictions[0][0].item(),
            'finbert_neutral': predictions[0][1].item(),
            'finbert_positive': predictions[0][2].item()
        })
    
    # Application de l'analyse sur chaque texte avec tqdm
    sentiment_scores = pd.DataFrame()
    for text in tqdm(df['cleaned_text']):
        scores = get_finbert_sentiment(text)
        sentiment_scores = pd.concat([sentiment_scores, scores.to_frame().T], ignore_index=True)
    
    # Ajout des scores au DataFrame
    df = pd.concat([df, sentiment_scores], axis=1)
    
    # Afficher quelques exemples
    print("\nExemples de scores FinBERT :")
    for _, row in df.head().iterrows():
        print(f"\nTexte : {row['cleaned_text'][:100]}...")
        print(f"Scores : Neg={row['finbert_negative']:.3f}, Neu={row['finbert_neutral']:.3f}, Pos={row['finbert_positive']:.3f}")
    
    return df

def aggregate_daily_sentiment(df):
    """Agrège les scores de sentiment par jour"""
    # Liste des colonnes de sentiment
    finbert_cols = ['finbert_negative', 'finbert_neutral', 'finbert_positive']
    
    # Agrégation par jour
    daily_sentiment = df.groupby('date').agg({
        **{col: 'mean' for col in finbert_cols},
        'cleaned_text': 'count'  # Nombre de posts par jour
    }).rename(columns={'cleaned_text': 'post_count'})
    
    # Afficher les statistiques quotidiennes
    print("\nStatistiques quotidiennes :")
    print(f"Nombre total de jours : {len(daily_sentiment)}")
    print("\nMoyennes des scores par jour :")
    print(daily_sentiment[finbert_cols].describe())
    
    return daily_sentiment

def save_sentiment_data(sentiment_df, daily_sentiment):
    """Sauvegarde les données avec les scores de sentiment"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'data_processed')
        
        # Sauvegarde des données complètes avec sentiment
        sentiment_output = os.path.join(output_dir, 'reddit_with_finbert_sentiment.csv')
        sentiment_df.to_csv(sentiment_output, index=False)
        
        # Sauvegarde des données agrégées par jour
        daily_output = os.path.join(output_dir, 'daily_finbert_sentiment.csv')
        daily_sentiment.to_csv(daily_output)
        
        print(f"\nDonnées sauvegardées avec succès dans :\n{sentiment_output}\n{daily_output}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")

if __name__ == "__main__":
    # 1. Chargement des données
    reddit_df, stock_df = load_processed_data()
    
    if reddit_df is not None:
        # 2. Application de FinBERT
        reddit_df = apply_finbert_sentiment(reddit_df)
        
        # 3. Agrégation quotidienne
        daily_sentiment = aggregate_daily_sentiment(reddit_df)
        
        # 4. Sauvegarde des résultats
        save_sentiment_data(reddit_df, daily_sentiment)