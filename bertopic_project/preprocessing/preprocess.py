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

def load_and_clean_data():
    # Obtenir les chemins absolus des fichiers
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stock_path = os.path.join(base_dir, 'data', 'raw', 'all_stocks_history.csv')
    reddit_path = os.path.join(base_dir,'data', 'raw', 'reddit_data.csv')

    try:
        stock_df = pd.read_csv(stock_path)
        reddit_df = pd.read_csv(reddit_path)
        print(f"Données chargées avec succès depuis :\n{stock_path}\n{reddit_path}")
    except FileNotFoundError as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return None, None
    
    reddit_df['date'] = pd.to_datetime(reddit_df['date'], format='mixed', errors='coerce')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='mixed', errors='coerce')

    # Vérifier s'il y a des valeurs NaT après conversion (signifie que certaines dates n'ont pas pu être converties)
    print(reddit_df[reddit_df['date'].isna()])
    print(stock_df[stock_df['Date'].isna()])

    # Periode sur laquelle on va travailler (a modifier)
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2024-12-31')

    reddit_df = reddit_df[(reddit_df['date'] >= start_date) & (reddit_df['date'] <= end_date)]
    stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)]

    print(f"Données chargées et filtrées pour la période {start_date} à {end_date}")
    print(f"Nombre de posts Reddit: {len(reddit_df)}")
    print(f"Nombre de jours de trading: {len(stock_df)}")
            
    def clean_stock_data(df):
        ''' Nettoyage des données boursières (stock_df) '''
        df = df.copy()
        
        # Conversion de la colonne Date en datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Nettoyage de la colonne Volume (supprimer les virgules et convertir en int)
        df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
        
        # Vérification des valeurs manquantes
        df = df.dropna()
        
        # Vérification des valeurs infinies
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Tri par date
        df = df.sort_values('Date')
        
        return df

    def clean_reddit_data(df):
        ''' Nettoyage des données Reddit (reddit_df) '''

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Nettoyage du texte
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r'http\S+|www.\S+', '', text)  # Suppression des URLs
            text = re.sub(r'[^\w\s]', '', text)    # Suppression des caractères spéciaux
            text = re.sub(r'\s+', ' ', text)  # Suppression des espaces multiples
            return text.lower().strip()
        
        df['clean_title'] = df['title'].apply(clean_text)
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Combinaison du titre et du texte (y a des titres et pas de texte correspondant donc j'ai fusionner les deux colonnes pour limiter les valeurs manquantes ds la col text)
        df['cleaned_text'] = df['clean_title'] + " " + df['clean_text']

        # Suppression des doublons
        df = df.drop_duplicates(subset=['cleaned_text', 'date'])

        # Suppression des lignes où le texte combiné est vide
        df = df[df['cleaned_text'].str.len() > 0]
 
        df = df.drop(['title', 'text', 'clean_title', 'clean_text'], axis=1)
        
        return df

    print("\nDébut du nettoyage des données...")
    clean_stock_df = clean_stock_data(stock_df)
    clean_reddit_df = clean_reddit_data(reddit_df)

    print("\nInformations sur les données nettoyées :")
    print("\nDonnées boursières :")
    print(f"Nombre de lignes : {len(clean_stock_df)}")
    print(f"Période : de {clean_stock_df['Date'].min()} à {clean_stock_df['Date'].max()}")
    print("\nDonnées Reddit :")
    print(f"Nombre de lignes : {len(clean_reddit_df)}")
    print(f"Période : de {clean_reddit_df['date'].min()} à {clean_reddit_df['date'].max()}")

    return clean_stock_df, clean_reddit_df

def save_cleaned_data(stock_df, reddit_df, data_processed_dir):
    """Sauvegarde les données nettoyées"""
    try:
        # Création des chemins de fichiers
        stock_output = os.path.join(data_processed_dir, 'clean_stock_data.csv')
        reddit_output = os.path.join(data_processed_dir, 'clean_reddit_data.csv')
        
        # Sauvegarde des fichiers
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