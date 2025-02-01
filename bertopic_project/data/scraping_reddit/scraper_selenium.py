from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
from datetime import datetime


class RedditScraper:
    def __init__(self):
        self.subreddits = [
            "wallstreetbets",
            "stocks",
            "investing",
            "StockMarket",
            "pennystocks",
            "robinhood",
            "trading",
            "finance",
        ]
        options = Options()
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)

    def scroll_and_collect(self, url, ticker, company_name, subreddit):
        try:
            self.driver.get(url)
            time.sleep(3)

            posts_data = set()
            last_height = self.driver.execute_script(
                "return document.body.scrollHeight"
            )

            while True:
                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(2)

                posts = self.driver.find_elements(
                    By.CSS_SELECTOR, "a[data-testid='post-title-text']"
                )
                print(f"Found {len(posts)} posts on page")

                for post in posts:
                    try:
                        text = post.text
                        parent = post.find_element(By.XPATH, "../../..")

                        # Get date
                        time_element = parent.find_element(By.CSS_SELECTOR, "time")
                        date = time_element.get_attribute("datetime")

                        # Get upvotes and comments
                        try:
                            faceplate_numbers = parent.find_elements(
                                By.CSS_SELECTOR, "faceplate-number"
                            )
                            upvotes = self.parse_number(
                                faceplate_numbers[0].get_attribute("number")
                            )
                            comments = self.parse_number(
                                faceplate_numbers[1].get_attribute("number")
                            )
                        except:
                            upvotes, comments = 0, 0

                        post_data = (date, text, ticker, subreddit, upvotes, comments)
                        if post_data not in posts_data:
                            posts_data.add(post_data)
                            print(
                                f"New post found: {text} | Upvotes: {upvotes} | Comments: {comments}"
                            )
                    except Exception as e:
                        print(f"Error extracting post: {str(e)}")
                        continue

                new_height = self.driver.execute_script(
                    "return document.body.scrollHeight"
                )
                if new_height == last_height:
                    time.sleep(10)
                    final_height = self.driver.execute_script(
                        "return document.body.scrollHeight"
                    )
                    if final_height == new_height:
                        break
                last_height = new_height

                print(f"Found {len(posts_data)} unique posts so far...")

            return [
                {
                    "date": p[0],
                    "text": p[1],
                    "ticker": p[2],
                    "company_name": company_name,
                    "source": "reddit",
                    "subreddit": p[3],
                    "upvotes": p[4],
                    "comments": p[5],
                }
                for p in posts_data
            ]

        except Exception as e:
            print(f"Error scraping {subreddit}: {str(e)}")
            return []

    def parse_number(self, text):
        """Convert Reddit number format to integer"""
        if not text:
            return 0
        text = text.lower().replace(",", "")
        try:
            if "k" in text:
                return int(float(text.replace("k", "")) * 1000)
            elif "m" in text:
                return int(float(text.replace("m", "")) * 1000000)
            return int(text)
        except:
            return 0

    def scrape_ticker(self, ticker, company_name):
        all_posts = []

        for subreddit in self.subreddits:
            print(f"\nScraping r/{subreddit} for {ticker}...")
            url = f"https://www.reddit.com/r/{subreddit}/search/?q={ticker}&sort=top&t=year"
            posts = self.scroll_and_collect(url, ticker, company_name, subreddit)
            if posts:
                all_posts.extend(posts)
            time.sleep(2)

        return all_posts

    def close(self):
        self.driver.quit()


def get_user_input():
    """Get stock tickers from user input"""
    print(
        "\nEnter stock ticker(s) to scrape (separated by spaces, e.g., 'AAPL MSFT META'):"
    )
    tickers_input = input().strip().upper()

    # Convert input to list and create dict with company placeholder
    tickers = tickers_input.split()

    if not tickers:
        print("No tickers entered. Please try again.")
        return get_user_input()

    # Create dictionary with placeholder company names
    stocks_dict = {ticker: ticker for ticker in tickers}

    print(f"\nWill scrape data for: {', '.join(tickers)}")
    return stocks_dict


def main():
    # Get tickers from user
    stocks = get_user_input()

    # Initialize scraper
    scraper = RedditScraper()
    all_data = []

    try:
        for ticker, company in stocks.items():
            print(f"\nProcessing {ticker}...")
            posts = scraper.scrape_ticker(ticker, company)
            if posts:
                # Save individual ticker data
                df = pd.DataFrame(posts)
                filename = f"reddit_posts_{ticker}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {len(posts)} posts for {ticker} to {filename}")
                all_data.extend(posts)

        # Save combined data if we have multiple tickers
        if len(stocks) > 1 and all_data:
            combined_filename = f"reddit_posts_ALL.csv"
            pd.DataFrame(all_data).to_csv(combined_filename, index=False)
            print(f"\nSaved combined data to {combined_filename}")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        scraper.close()
        print("\nScraping completed!")


if __name__ == "__main__":
    main()
