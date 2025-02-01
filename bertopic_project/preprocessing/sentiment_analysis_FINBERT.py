import pandas as pd
import numpy as np
import os
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_processed_data():
    """Charge les données nettoyées"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reddit_path = os.path.join(
            base_dir, "data", "data_processed", "clean_reddit_data.csv"
        )
        stock_path = os.path.join(
            base_dir, "data", "data_processed", "clean_stock_data.csv"
        )

        reddit_df = pd.read_csv(reddit_path)
        stock_df = pd.read_csv(stock_path)

        # Conversion des dates
        reddit_df["date"] = pd.to_datetime(reddit_df["date"])
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])

        print("Données chargées avec succès!")
        return reddit_df, stock_df

    except FileNotFoundError as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return None, None


def extract_topics(df):
    """Extrait les topics avec BERTopic"""
    print("Extraction des topics avec BERTopic...")

    # Initialisation du modèle BERTopic
    topic_model = BERTopic(language="english", min_topic_size=20, n_gram_range=(1, 3))

    # Entraînement du modèle sur les textes
    topics, probs = topic_model.fit_transform(df["cleaned_text"])

    # Ajout des topics au DataFrame
    df["topic"] = topics

    print("\nTopics les plus fréquents :")
    print(topic_model.get_topic_info().head())

    return df, topic_model


def apply_finbert_sentiment(df):
    """Applique l'analyse de sentiment FinBERT sur les textes"""
    # Chargement du modèle FinBERT
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    print("Application de l'analyse FinBERT...")

    def get_finbert_sentiment(text):
        inputs = tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        )
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return pd.Series(
            {
                "finbert_negative": predictions[0][0].item(),
                "finbert_neutral": predictions[0][1].item(),
                "finbert_positive": predictions[0][2].item(),
            }
        )

    sentiment_scores = pd.DataFrame()
    for text in tqdm(df["cleaned_text"]):
        scores = get_finbert_sentiment(text)
        sentiment_scores = pd.concat(
            [sentiment_scores, scores.to_frame().T], ignore_index=True
        )

    df = pd.concat([df, sentiment_scores], axis=1)
    return df


def combine_and_process_data(reddit_df, stock_df):
    """Fusionne et prépare les données pour l'analyse"""
    # Fusion des données Reddit et stocks sur la date et le ticker
    combined_df = pd.merge(
        reddit_df, stock_df, left_on=["date", "ticker"], right_on=["Date", "Ticker"]
    )

    # Encodage des variables catégorielles
    combined_df["topic"] = pd.Categorical(combined_df["topic"]).codes

    # Normalisation des features numériques
    numeric_features = [
        "finbert_negative",
        "finbert_neutral",
        "finbert_positive",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    combined_df[numeric_features] = (
        combined_df[numeric_features] - combined_df[numeric_features].mean()
    ) / combined_df[numeric_features].std()

    return combined_df


def save_processed_data(reddit_df, combined_df, topic_model):
    """Sauvegarde les données traitées"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "data_processed")

        # Sauvegarde des données Reddit avec topics et sentiments
        reddit_output = os.path.join(output_dir, "reddit_processed_full.csv")
        reddit_df.to_csv(reddit_output, index=False)

        # Sauvegarde des données combinées et normalisées
        combined_output = os.path.join(output_dir, "combined_processed_data.csv")
        combined_df.to_csv(combined_output, index=False)

        # Sauvegarde du modèle de topics
        model_output = os.path.join(output_dir, "bertopic_model")
        topic_model.save(model_output)

        print(
            f"\nDonnées sauvegardées avec succès dans :\n{reddit_output}\n{combined_output}"
        )
        print(f"Modèle BERTopic sauvegardé dans : {model_output}")

    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")


if __name__ == "__main__":

    reddit_df, stock_df = load_processed_data()

    if reddit_df is not None and stock_df is not None:
        # Extraction des topics
        reddit_df, topic_model = extract_topics(reddit_df)

        # Analyse de sentiment
        reddit_df = apply_finbert_sentiment(reddit_df)

        # Combinaison et préparation des données
        combined_df = combine_and_process_data(reddit_df, stock_df)

        # Sauvegarde des résultats
        save_processed_data(reddit_df, combined_df, topic_model)
