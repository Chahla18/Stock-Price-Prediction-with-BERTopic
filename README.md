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