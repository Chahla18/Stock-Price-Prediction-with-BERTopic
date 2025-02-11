import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import os 

class StockPrediction:
    def __init__(self, stock_file, sentiment_file, sentiment_file_with_topics, seq_length=20, horizon=19):
        self.stock_file = stock_file
        self.sentiment_file = sentiment_file
        self.sentiment_file_with_topics = sentiment_file_with_topics
        self.seq_length = seq_length
        self.horizon = horizon
        
        # Fixer les seeds
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Initialisation des scalers
        self.adj_close_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
    
    def load_and_merge_data(self):
        stock_data = pd.read_csv(self.stock_file)
        sentiment_data = pd.read_csv(self.sentiment_file)

        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
        
        daily_sentiment = sentiment_data.groupby('date').agg({'vader_compound': 'mean'}).reset_index()
        daily_sentiment.columns = ['Date', 'vader_sentiment']
        daily_sentiment = daily_sentiment.rename(columns={'date': 'Date'})
        
        merged_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
        merged_data['vader_sentiment'] = merged_data['vader_sentiment'].fillna(method='ffill')
        
        return merged_data.dropna()

    def load_and_merge_data_with_topics(self):
        stock_data = pd.read_csv(self.stock_file)
        sentiment_data = pd.read_csv(self.sentiment_file_with_topics)
        
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
        
        daily_sentiment = sentiment_data.groupby('date').agg({'vader_sentiment': 'mean'}).reset_index()
        daily_sentiment.rename(columns={'date': 'Date'}, inplace=True)
        
        merged_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
        merged_data['vader_sentiment'] = merged_data['vader_sentiment'].fillna(method='ffill')
        
        return merged_data.dropna()
    
    def preprocess_data(self, merged_data, features):
        train_data = merged_data[merged_data['Date'].astype(str).str.startswith('2024')].copy()
        test_data = merged_data[merged_data['Date'].astype(str).str.startswith('2025')].copy()
        
        train_data['Adj Close'] = self.adj_close_scaler.fit_transform(train_data[['Adj Close']])
        test_data['Adj Close'] = self.adj_close_scaler.transform(test_data[['Adj Close']])
        
        train_data[features] = self.feature_scaler.fit_transform(train_data[features])
        test_data[features] = self.feature_scaler.transform(test_data[features])
        
        return train_data, test_data
    
    def create_sequences(self, data, target_idx=4):
        X, y = [], []
        data_array = np.array(data).astype(np.float32)
        for i in range(len(data_array) - self.seq_length - self.horizon):
            X.append(data_array[i:(i + self.seq_length)])
            y.append(data_array[i + self.seq_length:i + self.seq_length + self.horizon, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=100),
            Dense(units=self.horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_lstm_model(self, model, X_train, y_train):
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        
        return model
    
    def run_prediction(self, with_topics):

        if with_topics :
            data = self.load_and_merge_data_with_topics()
            features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                    'MA7', 'MA20', 'MACD', '20SD', 'Upper_Band', 'Lower_Band', 
                    'EMA', 'Log_Momentum', 'vader_sentiment']
        else :
            data = self.load_and_merge_data()
            features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA7', 'MA20', 'MACD',
                        '20SD', 'Upper_Band', 'Lower_Band', 'EMA', 'Log_Momentum', 'vader_sentiment']
        
        train_data, test_data = self.preprocess_data(data, features)
        X_train, y_train = self.create_sequences(train_data[features].values)
        X_test = np.array(train_data[features].values[-self.horizon:]).astype(np.float32)
        
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model = self.train_lstm_model(model, X_train, y_train)
        
        last_sequence = X_test.reshape(1, self.horizon, X_train.shape[2])
        next_pred_scaled = model.predict(last_sequence)[0].reshape(-1, 1)
        next_pred = self.adj_close_scaler.inverse_transform(next_pred_scaled).flatten()
        
        last_date = pd.to_datetime(data['Date'].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.horizon).date
        
        real_adj_close = self.adj_close_scaler.inverse_transform(
            np.array(test_data['Adj Close']).reshape(self.horizon, 1)
        ).flatten()
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Adj_Close': next_pred,
            'Real_Adj_Close': real_adj_close
        })
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if with_topics :
            future_df.to_csv(os.path.join(script_dir, "future_predictions_v2_with_topics.csv"), index=False)        
            print("\nLes prédictions futures ont été sauvegardées dans 'future_predictions_v2_with_topics.csv'")
        else :
            future_df.to_csv(os.path.join(script_dir, "future_predictions_v2.csv"), index=False)        
            print("\nLes prédictions futures ont été sauvegardées dans 'future_predictions_v2.csv'")
        
        return future_df


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stock_file = os.path.join(script_dir, "../data_preprocessing/processed_data/processed_stock_data.csv")
    sentiment_file = os.path.join(script_dir, "../data_preprocessing/processed_data/comments_with_sentiments_without_topics.csv")
    sentiment_file_with_topics = os.path.join(script_dir, "../data_preprocessing/processed_data/comments_with_sentiments_with_topics.csv")

    predictor = StockPrediction(stock_file, sentiment_file, sentiment_file_with_topics)
    predictor.run_prediction(with_topics=False)
    predictor.run_prediction(with_topics=True)

