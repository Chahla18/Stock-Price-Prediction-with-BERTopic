# 📈 Enhancing Stock Price Prediction Using Sentiment Analysis and Deep Learning

## The Project
This project combines sentiment analysis from multiple sources with deep learning techniques to enhance stock price prediction accuracy. The objective is to create a more robust prediction model by:

* Integrating sentiment analysis from social media (X/Twitter) with VADER and FINBERT
* Using BERTopic for advanced topic modeling 
* Implementing deep learning model for price prediction (LSTM)


## Project Structure

```text
📁 bertopic_project/
├── 📁 data_extraction/           # Data collection and scraping scripts
│   ├── 📁 raw/                  # collected data storage
│   ├── 📁 scraping_X/           # Twitter/X data collection scripts
│   ├── 📁 scraping_reddit/      # Reddit data collection scripts
│   ├── 📁 scraping_yfinance/    # Yahoo finance data collection scripts
│
├── 📁 data_preprocessing/        # Data cleaning and preparation
│   ├── 📁 data_preping/         # Data preprocessing scripts
│   ├── 📁 processed_data/       # Cleaned and processed datasets
│   ├── 📁 sentiment_analysis/   # Sentiment analysis implementation 
│   ├── 📁 topics/               # Topic modeling with BERTopic
│
├── 📁 data_prediction/          # Model implementation
│   ├── 📄 model.py             # LSTM model architecture
│
└── 📄 main.py                   # Main execution script
📁 notebooks/
├── 📄 main_notebook.ipynb                # Main notebook


```

## Installation
 1.Clone the repository

```bash
git clone https://github.com/Chahla18/Stock-Price-Prediction-with-BERTopic.git
cd Stock-Price-Prediction-with-BERTopic
```
2. Create virtual environment
```bash
python -m venv venv
```
3. Activate virtual environment

### On Windows:
```bash
.\venv\Scripts\activate
```
### On Linux/Mac:
```bash
source venv/bin/activate
```
4. Install required packages
```bash
pip install poetry 
```
Then
```bash
poetry install
```
To run a script, use:

```bash
poetry run python chemin/vers/ton_script.py
```

## API: Tesla Data Analysis

The API enables the extraction and analysis of Tesla stock data and generates predictions based on an LSTM model.

### 📌 Installation and Execution

1. Make sure you have installed the necessary dependencies

2. Start the API by running:  
   ```bash
   python -m bertopic_project.data_prediction.api
   ```
   The API will be available at `http://localhost:8000/`

---

### 📊 Available Endpoints

#### **1️⃣ Main Endpoint**
- **GET `/`**  
  - Returns general information about the API and its endpoints.

#### **2️⃣ System Monitoring**
- **GET `/health`**  
  - Checks the system status and returns CPU, memory, and disk usage information.

#### **3️⃣ Stock Data Scraping**
- **POST `/api/scrape/tesla-stock`**  
  - Executes real-time scraping of Tesla stock data from Yahoo Finance.  
  - Stores the data in a CSV file.

#### **4️⃣ Data Retrieval**
- **GET `/api/data/tesla-stock`**  
  - Returns historical Tesla stock data stored in `tesla_stock_history.csv`.

- **GET `/api/data/reddit`**  
  - Returns extracted Reddit data stored in `reddit_data.csv`.

- **GET `/api/data/tesla-tweets`**  
  - Returns Tesla-related tweets stored in `Tweets_TSLA.csv`.

#### **5️⃣ Prediction Generation and Retrieval**
- **GET `/api/data/predictions`**  
  - Executes the LSTM model stored in `bertopic_project/data_prediction/model.py`.  
  - Generates new predictions for the next 30 days.  
  - Returns the updated forecasts stored in `future_predictions.csv`.

---

### 🚀 Usage Examples

#### Retrieve updated predictions:
```bash
curl -X 'GET' 'http://localhost:8000/api/data/predictions' -H 'accept: application/json'
```

#### Scrape stock data:
```bash
curl -X 'POST' 'http://localhost:8000/api/scrape/tesla-stock' -H 'accept: application/json'
```

The API provides the latest Tesla stock data and updated predictions on future market trends.


## 👥 Authors
- [Morgan JOWITT](https://github.com/morganjowitt)
- [Chahla TARMOUN](https://github.com/Chahla18)
- [Aya MOKHTAR](https://github.com/ayamokhtar)
