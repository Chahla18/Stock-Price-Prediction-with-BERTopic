import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from tqdm import tqdm

class DirectSentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzers"""
        # Setup paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_preprocessing_dir = os.path.dirname(self.current_dir)
        self.project_dir = os.path.dirname(self.data_preprocessing_dir)
        self.data_dir = os.path.join(self.project_dir, "data_preprocessing", "processed_data")
        
        print("Initializing sentiment analyzers...")
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize FinBERT
        print("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def get_vader_sentiment(self, text):
        """Get VADER sentiment scores"""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu']
        }

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
        """Analyze sentiments for all comments"""
        # Load social media data
        input_path = os.path.join(self.data_dir, "processed_social_data.csv")
        output_path = os.path.join(self.data_dir, "comments_with_sentiments_without_topics.csv")
        
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        
        # VADER Analysis
        print("\nCalculating VADER sentiment...")
        vader_results = []
        for text in tqdm(df['content'], desc="VADER"):
            vader_results.append(self.get_vader_sentiment(text))
            
        # Add VADER scores to dataframe
        df['vader_compound'] = [x['compound'] for x in vader_results]
        df['vader_positive'] = [x['pos'] for x in vader_results]
        df['vader_negative'] = [x['neg'] for x in vader_results]
        df['vader_neutral'] = [x['neu'] for x in vader_results]
        
        # FinBERT Analysis
        print("\nCalculating FinBERT sentiment...")
        finbert_results = []
        for text in tqdm(df['content'], desc="FinBERT"):
            finbert_results.append(self.get_finbert_sentiment(text))
            
        # Add FinBERT scores
        df['finbert_positive'] = [x['positive'] for x in finbert_results]
        df['finbert_negative'] = [x['negative'] for x in finbert_results]
        df['finbert_neutral'] = [x['neutral'] for x in finbert_results]
        
        # Add dominant sentiments
        df['vader_sentiment'] = df['vader_compound'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
        )
        
        df['finbert_sentiment'] = df.apply(
            lambda row: max(['positive', 'negative', 'neutral'],
                          key=lambda x: row[f'finbert_{x}']),
            axis=1
        )
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary statistics
        print("\nSentiment Distribution Summary:")
        print("\nVADER Sentiment Distribution:")
        print(df['vader_sentiment'].value_counts(normalize=True).round(3))
        
        print("\nFinBERT Sentiment Distribution:")
        print(df['finbert_sentiment'].value_counts(normalize=True).round(3))
        
        print("\nSentiment by Source:")
        print(df.groupby('source')['vader_compound'].mean().round(3))
        
        return df

if __name__ == "__main__":
    analyzer = DirectSentimentAnalyzer()
    df_with_sentiments = analyzer.analyze_sentiments()