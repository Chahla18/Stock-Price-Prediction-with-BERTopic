# Twitter Scraper with Account Rotation

This is a Twitter scraper that collects tweets for a given stock ticker over a specified date range while rotating through multiple Twitter accounts to avoid rate limits and blocks. The scraper uses Selenium with undetected-chromedriver for a stealthier browsing experience, along with randomized delays to mimic human behavior.

## Directory Structure

```
scraping_twitter/
   ├── x_scrapper.py
   ├── cli.py
   └── README.md
```

## Usage

### Preparing Your Accounts

The scraper requires one or more Twitter account credentials. You can provide these in a CSV-like text file where each line contains a username and password separated by a comma. For example, create a file named `accounts.txt` with the following format:

```
username1,password1
username2,password2
username3,password3
```

### Running the Scraper

You can run the scraper using the `cli.py` script. Open a terminal in the `accounts_rotation/` directory and run:

```bash
python cli.py -t TSLA -d 2024-01-01 --accounts-file accounts.txt --switch-days 5 --tweets-per-day 500 --max 10000 --batch-size 1000 --headless
```

#### Command-Line Arguments

- **-t / --ticker**  
  The stock ticker symbol to search for (e.g., TSLA).

- **-d / --date**  
  The starting date (in `YYYY-MM-DD` format) for scraping tweets.

- **-m / --max**  
  (Optional) Overall maximum number of tweets to collect.

- **-b / --batch-size**  
  (Optional) The number of tweets to collect before saving to a CSV file. Defaults to 1000.

- **--tweets-per-day**  
  (Optional) Maximum tweets to scrape for each day.

- **-a / --accounts-file**  
  (Optional) Path to a file containing account details. Each line should be in the format: `username,password`.

- **--switch-days**  
  (Optional) The number of days to scrape before switching to the next account. Defaults to 5.

- **--headless**  
  (Optional) Run the browser in headless mode (without opening a visible window).

### How It Works

- **Account Rotation:**  
  The scraper cycles through the provided accounts after a set number of days (controlled by `--switch-days`), helping to avoid blocks.

- **Scraping Tweets:**  
  It queries Twitter for tweets mentioning the specified ticker, limited to the English language (`lang:en`), over the desired date range. Tweets are scraped day by day, and if a day has fewer tweets than the specified `--tweets-per-day`, it moves on to the next day.

- **Saving Data:**  
  Tweets are saved in batches to a CSV file named `{ticker}_tweets.csv` in the project directory.