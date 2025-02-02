import argparse
from getpass import getpass
from dotenv import load_dotenv
import os
from x_scrapper import XScraper
from time import sleep

def load_accounts(account_file=None):
    """
    Load account details from a file or return an empty list if no file is provided.
    """
    accounts = []
    if account_file:
        try:
            with open(account_file, 'r') as file:
                for line in file.readlines():
                    username, password = line.strip().split(',')
                    accounts.append((username, password))
        except FileNotFoundError:
            print(f"Account file '{account_file}' not found. Proceeding with no accounts.")
    else:
        print("No account file provided, expecting manual input.")
        username = input("Twitter username: ")
        password = getpass("Twitter password: ")
        accounts.append((username, password))
    
    return accounts

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Scrape tweets about stock tickers")
    parser.add_argument("-t", "--ticker", required=True, help="Stock ticker symbol (e.g., TSLA)")
    parser.add_argument("-d", "--date", required=True, help="Starting date (YYYY-MM-DD)")
    parser.add_argument("-m", "--max", type=int, default=None, help="Overall maximum tweets to collect")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Number of tweets to collect before saving to file")
    parser.add_argument("--tweets-per-day", type=int, default=None, help="Maximum tweets to scrape per day")
    parser.add_argument("-a", "--accounts-file", help="Path to a file containing account details (username,password)")
    parser.add_argument("--switch-days", type=int, default=5, help="Switch accounts after how many days")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()

    accounts = load_accounts(args.accounts_file)
    
    if not accounts:
        print("No accounts provided. Exiting.")
        return
    
    scraper = XScraper(
        accounts=accounts,  
        ticker=args.ticker,
        date_limit=args.date,
        max_tweets=args.max,
        batch_size=args.batch_size,
        tweets_per_day=args.tweets_per_day,
        headless=args.headless,
        account_switch_interval=args.switch_days
    )

    try:
        scraper.scrape_tweets()
        print(f"Saved {scraper.total_tweets_processed} tweets to {scraper.filename}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
