### In order to get stock prices from yahoo finance:

- Run the main scraper script :
```bash
python -m python -m bertopic_project.data.scraping_yfinance.scraper
```
- In your terminal, you will see the list of stock names of which you can get data. You will be asked to enter the stock tickers you want to analyze, separated by spaces. Let's say I want data on tesla, meta and apple:
```bash
Enter the tickers you want to analyze (separated by spaces):
TSLA AAPL META
```
### Scripts:

1. The script **trending_tickers.py** opens this page: **https://finance.yahoo.com/markets/stocks/trending/**, which contains the current 25 trending stocks. The first two columns (Symbol and Name) of the table are extracted.

3. The second script : **historical_data.py** manages historical data collection for individual stocks:
- It takes a list of stock tickers and company names
- For each stock, it navigates to the stock's historical data page **https://finance.yahoo.com/quote/AAPL/history/?period1=1580341468&period2=1738185894** for example, scrapes the price history table and creates a CSV file for each stock in the **data/raw/** directory (e.g., AAPL_history.csv)

4. The final script **scraper.py** will:
- Create a merged file containing all requested stocks' data as **all_stocks_history.csv**
- The file contains the following columns: **Date, Open, High, Low, Close, Adj Close, Volume, Ticker, Company_Name**

## API: Tesla Data Analysis

The API enables the extraction and analysis of Tesla stock data and generates predictions based on an LSTM model.

### 📌 Installation and Execution

1. Make sure you have installed the necessary dependencies:  
   ```bash
   pip install fastapi uvicorn pandas numpy psutil
   ```

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