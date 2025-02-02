import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os 

class DataProcessor:
    def __init__(self, stock_file, sentiment_file):
        self.stock_file = stock_file
        self.sentiment_file = sentiment_file
        self.features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                         'MA7', 'MA20', 'MACD', '20SD', 'Upper_Band', 'Lower_Band', 
                         'EMA', 'Log_Momentum', 'vader_sentiment']
        self.adj_close_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
    
    def load_and_merge_data(self):
        stock_data = pd.read_csv(self.stock_file)
        sentiment_data = pd.read_csv(self.sentiment_file)

        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
        
        daily_sentiment = sentiment_data.groupby('date').agg({'vader_compound': 'mean'}).reset_index()
        daily_sentiment.columns = ['Date', 'vader_sentiment']
        
        merged_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
        merged_data['vader_sentiment'] = merged_data['vader_sentiment'].fillna(method='ffill')
        return merged_data.dropna()
    
    def preprocess_data(self, merged_data):
        train_data = merged_data[merged_data['Date'].astype(str).str.startswith('2024')]
        test_data = merged_data[merged_data['Date'].astype(str).str.startswith('2025')]
        
        train_data['Adj Close'] = self.adj_close_scaler.fit_transform(train_data[['Adj Close']])
        test_data['Adj Close'] = self.adj_close_scaler.transform(test_data[['Adj Close']])
        
        train_data[self.features] = self.feature_scaler.fit_transform(train_data[self.features])
        test_data[self.features] = self.feature_scaler.transform(test_data[self.features])
        
        return train_data, test_data
    
    def create_sequences(self, data, seq_length=1, target_idx=4):
        X, y = [], []
        data_array = np.array(data).astype(np.float32)
        for i in range(len(data_array) - seq_length):
            X.append(data_array[i:(i + seq_length)])
            y.append(data_array[i + seq_length, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train):
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=200, batch_size=32, 
                       validation_split=0.1, callbacks=[early_stop], verbose=1)
    
    def predict(self, X):
        return self.model.predict(X)

class StockPredictor:
    def __init__(self, stock_file, sentiment_file):
        self.processor = DataProcessor(stock_file, sentiment_file)
    
    def run(self):
        data = self.processor.load_and_merge_data()
        train_data, _ = self.processor.preprocess_data(data)
        
        X_train, y_train = self.processor.create_sequences(train_data[self.processor.features].values)
        
        model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.train(X_train, y_train)
        
        last_sequence = X_train[-1].reshape(1, X_train.shape[1], X_train.shape[2])
        future_predictions = []
        
        for _ in range(30):  # Predict next 30 days
            next_pred_scaled = model.predict(last_sequence)[0, 0]
            next_pred = self.processor.adj_close_scaler.inverse_transform([[next_pred_scaled]])[0, 0]
            future_predictions.append(next_pred)
            
            next_pred_scaled = self.processor.adj_close_scaler.transform([[next_pred]])[0, 0]
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred_scaled  # Ajouter la nouvelle prédiction à la séquence
        
        # Save predictions to CSV
        future_dates = pd.date_range(start=pd.to_datetime(data['Date'].max()) + pd.Timedelta(days=1), periods=30).date
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Adj_Close': future_predictions})
        script_dir = os.path.dirname(os.path.abspath(__file__))
        future_df.to_csv(os.path.join(script_dir, "future_predictions.csv"), index=False)
        
        print("\nFuture predictions saved to future_predictions.csv")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stock_file = os.path.join(script_dir, "../data_preprocessing/processed_data/processed_stock_data.csv")
    sentiment_file = os.path.join(script_dir, "../data_preprocessing/processed_data/comments_with_sentiments_without_topics.csv")
    predictor = StockPredictor(stock_file, sentiment_file)
    predictor.run()

if __name__ == "__main__":
    main()
