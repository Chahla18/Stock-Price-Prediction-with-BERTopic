import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzers"""
        # Setup paths - corrected for project structure
        self.current_dir = os.path.dirname(os.path.abspath(__file__))  # sentiment_analysis directory
        self.data_preprocessing_dir = os.path.dirname(self.current_dir)  # data_preprocessing directory
        self.project_dir = os.path.dirname(self.data_preprocessing_dir)  # bertopic_project directory
        
        # Set up specific paths
        self.data_dir = os.path.join(self.project_dir, "data_preprocessing", "processed_data")
        
        # Print paths for debugging
        print(f"Looking for input file at: {os.path.join(self.data_dir, 'comments_with_topics.csv')}")
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def get_vader_sentiment(self, text):
        """Get VADER sentiment scores"""
        scores = self.vader.polarity_scores(text)
        return scores['compound']

    def get_finbert_sentiment(self, text):
        """Get FinBERT sentiment prediction"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # FinBERT classes: positive (0), negative (1), neutral (2)
        sentiment_score = predictions[0].tolist()
        return {
            'positive': sentiment_score[0],
            'negative': sentiment_score[1],
            'neutral': sentiment_score[2]
        }

    def analyze_sentiments(self):
        """Analyze sentiments for comments with topics"""
        # Load comments with topics
        input_path = os.path.join(self.data_dir, "comments_with_topics.csv")
        output_path = os.path.join(self.data_dir, "comments_with_sentiments_with_topics.csv")
        
        print("Loading comments with topics...")
        df = pd.read_csv(input_path)
        
        print("Calculating VADER sentiment...")
        df['vader_sentiment'] = df['content'].apply(self.get_vader_sentiment)
        
        print("Calculating FinBERT sentiment...")
        finbert_sentiments = df['content'].apply(self.get_finbert_sentiment)
        
        # Add FinBERT sentiment scores as separate columns
        df['finbert_positive'] = finbert_sentiments.apply(lambda x: x['positive'])
        df['finbert_negative'] = finbert_sentiments.apply(lambda x: x['negative'])
        df['finbert_neutral'] = finbert_sentiments.apply(lambda x: x['neutral'])
        
        # Add dominant FinBERT sentiment
        df['finbert_sentiment'] = finbert_sentiments.apply(
            lambda x: max(x.items(), key=lambda k: k[1])[0]
        )
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Print summary by topic
        print("\nSentiment Summary by Topic:")
        topic_sentiment = df.groupby('topic').agg({
            'vader_sentiment': 'mean',
            'finbert_positive': 'mean',
            'finbert_negative': 'mean',
            'finbert_neutral': 'mean'
        }).round(3)
        
        print(topic_sentiment)
        return df

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    df_with_sentiments = analyzer.analyze_sentiments()