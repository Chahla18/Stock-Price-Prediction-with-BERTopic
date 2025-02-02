import pandas as pd
import numpy as np
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
import os

class TopicModeler:
    def __init__(self):
        """Initialize paths and download required NLTK data"""
        # Setup paths - corrected for project structure
        self.current_dir = os.path.dirname(os.path.abspath(__file__))  # topics directory
        self.data_preprocessing_dir = os.path.dirname(self.current_dir)  # data_preprocessing directory
        self.project_dir = os.path.dirname(self.data_preprocessing_dir)  # bertopic_project directory
        
        # Set up specific paths
        self.model_path = os.path.join(self.current_dir, "saved_model")
        self.results_path = os.path.join(self.current_dir, "topic_results.npz")
        self.data_dir = os.path.join(self.project_dir, "data_extraction", "raw")
        self.output_dir = os.path.join(self.project_dir, "data_preprocessing", "processed_data")
        
        # Print paths for debugging
        print(f"Loading data from: {os.path.join(self.output_dir, 'processed_social_data.csv')}")
        print(f"Model will be saved to: {self.model_path}")
        print(f"Results will be saved to: {self.results_path}")
        
        # Download stopwords if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def _get_stopwords(self) -> set:
        """Get combined set of standard and custom stopwords"""
        stop_words = set(stopwords.words('english'))
        custom_stops = {'tesla', 'tsla', 'stock', 'stocks', 'share', 'shares', 
                       'price', 'market', 'buy', 'sell', 'call', 'put'}
        stop_words.update(custom_stops)
        return stop_words

    def _clean_texts(self, texts: list) -> list:
        """Clean texts and remove stopwords"""
        stop_words = self._get_stopwords()
        cleaned_texts = []
        for text in texts:
            words = str(text).lower().split()
            words = [w for w in words if w not in stop_words]
            cleaned_texts.append(' '.join(words))
        return cleaned_texts

    def process_topics(self):
        """Process topics from social media data"""
        try:
            # Load original data
            input_file = os.path.join(self.output_dir, "processed_social_data.csv")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found at: {input_file}")
                
            df = pd.read_csv(input_file)
            cleaned_texts = self._clean_texts(df['content'].tolist())
            
            # Check if model exists
            if os.path.exists(self.model_path) and os.path.exists(self.results_path):
                print("Loading existing model...")
                topic_model = BERTopic.load(self.model_path)
                results = np.load(self.results_path)
                topics = results['topics']
            else:
                print("Creating new model...")
                # Create and fit model
                topic_model = BERTopic(nr_topics=15)
                topics, probs = topic_model.fit_transform(cleaned_texts)
                
                # Save model and results
                topic_model.save(self.model_path)
                np.savez(self.results_path, topics=topics, probs=probs)
            
            # Get topic information
            topic_info = topic_model.get_topic_info()
            
            # Add representative words for each topic
            topic_info['top_words'] = topic_info['Topic'].apply(
                lambda x: ', '.join([word for word, _ in topic_model.get_topic(x)][:5])
                if x != -1 else "No topic"
            )
            
            # Save topic info
            topic_info_path = os.path.join(self.output_dir, "topic_info.csv")
            topic_info.to_csv(topic_info_path, index=False)
            print(f"Topic information saved to: {topic_info_path}")
            
            # Create DataFrame with original content and assigned topics
            df_with_topics = df.copy()
            df_with_topics['topic'] = topics
            
            # Add topic description to each comment
            topic_word_dict = {row['Topic']: row['top_words'] 
                             for _, row in topic_info.iterrows()}
            df_with_topics['topic_words'] = df_with_topics['topic'].map(topic_word_dict)
            
            # Save comments with their topics
            comments_path = os.path.join(self.output_dir, "comments_with_topics.csv")
            df_with_topics.to_csv(comments_path, index=False)
            print(f"Comments with topics saved to: {comments_path}")
            
            return df_with_topics
            
        except Exception as e:
            print(f"Error in topic processing: {e}")
            return None

if __name__ == "__main__":
    modeler = TopicModeler()
    df_with_topics = modeler.process_topics()
    if df_with_topics is not None:
        # Print sample of comments with their topics
        print("\nSample of comments with assigned topics:")
        sample = df_with_topics.sample(5)
        for _, row in sample.iterrows():
            print(f"\nSource: {row['source']}")
            print(f"Topic: {row['topic']} (Keywords: {row['topic_words']})")
            print(f"Content: {row['content'][:200]}...")