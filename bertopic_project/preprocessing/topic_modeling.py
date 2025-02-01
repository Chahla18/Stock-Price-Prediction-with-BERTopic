import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def load_clean_data(data_processed_dir):
    """Load the preprocessed Reddit data"""
    reddit_path = os.path.join(data_processed_dir, 'clean_reddit_data.csv')
    if not os.path.exists(reddit_path):
        raise FileNotFoundError(f"Cleaned Reddit data not found at {reddit_path}")
    
    df = pd.read_csv(reddit_path)
    print(f" Loaded dataset with {len(df)} Reddit posts.")
    return df

def preprocess_text(df):
    """Remove common stopwords but keep finance-related terms."""
    standard_stopwords = set(ENGLISH_STOP_WORDS)
    
    def clean_text(text):
        return " ".join([word for word in text.split() if word.lower() not in standard_stopwords])
    
    df['clean_text'] = df['processed_text'].apply(clean_text)
    return df

def extract_topics(df):
    """Extract topics from cleaned text data using BERTopic"""
    vectorizer_model = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),  # Remove only standard stopwords
        ngram_range=(1, 3)  # Allow meaningful phrases
    )

    topic_model = BERTopic(
        min_topic_size=10,  # Avoid too many small topics
        n_gram_range=(1, 3),  # Allow meaningful phrases
        top_n_words=10,  # Keep top 10 words per topic
        vectorizer_model=vectorizer_model,
        verbose=True
    )

    # Fit BERTopic
    print(" Fitting BERTopic model (this may take a few minutes)...")
    topics, probs = topic_model.fit_transform(df['clean_text'])
    print(f" Initial Topics Found: {len(set(topics))}")

    # Reduce topics to 15
    print(" Reducing topics to 15...")
    topic_model = topic_model.reduce_topics(df['clean_text'], nr_topics=15)

    # Add topics to dataframe
    df['topic'] = topics

    # Get topic info
    topic_info = topic_model.get_topic_info()

    return df, topic_info, topic_model

def save_topic_results(df, topic_info, topic_model, data_processed_dir):
    """Save results of topic modeling"""
    # Save data with topics
    output_path = os.path.join(data_processed_dir, 'reddit_with_topics.csv')
    df.to_csv(output_path, index=False)

    # Save topic information
    topic_info_path = os.path.join(data_processed_dir, 'topic_info.csv')
    topic_info.to_csv(topic_info_path, index=False)

    # Save model
    model_path = os.path.join(data_processed_dir, 'topic_model')
    topic_model.save(model_path)

    print(f" Saved processed data to {output_path}")
    print(f" Saved topic info to {topic_info_path}")
    print(f" Saved model to {model_path}")

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_processed_dir = os.path.join(base_dir, 'data', 'processed')

    # Load data
    print(" Loading preprocessed Reddit data...")
    df = load_clean_data(data_processed_dir)

    # Preprocess text
    print(" Cleaning text data...")
    df = preprocess_text(df)

    # Extract topics
    print(" Extracting topics...")
    df_with_topics, topic_info, topic_model = extract_topics(df)

    # Print some basic stats
    print("\n Topic modeling complete!")
    print(f"Number of topics found: {len(topic_info)}")
    print("\n Most common topics:")
    print(topic_info.head())

    # Save results
    save_topic_results(df_with_topics, topic_info, topic_model, data_processed_dir)

if __name__ == "__main__":
    main()