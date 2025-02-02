import pandas as pd
from datetime import datetime, timedelta
import random
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import undetected_chromedriver as uc

class XScraper:
    def __init__(self, accounts, ticker, date_limit, max_tweets=None, batch_size=1000, tweets_per_day=None, headless=False, account_switch_interval=5):
        """
        :param accounts: A list of tuples containing (username, password) pairs for rotation.
        :param ticker: Stock ticker symbol.
        :param date_limit: Starting date (YYYY-MM-DD) for scraping.
        :param max_tweets: Overall maximum tweets to collect.
        :param batch_size: Number of tweets to collect before saving to file.
        :param tweets_per_day: Maximum tweets to scrape per day.
        :param headless: Run in headless mode if True.
        :param account_switch_interval: After how many days (or queries) to switch accounts.
        """
        self.accounts = accounts  # List of (username, password) pairs
        self.account_index = 0  
        self.ticker = ticker
        self.date_limit = datetime.strptime(date_limit, "%Y-%m-%d").date()
        self.max_tweets = max_tweets
        self.batch_size = batch_size
        self.tweets_per_day = tweets_per_day  # per-day limit
        self.headless = headless
        self.account_switch_interval = account_switch_interval  # Interval at which to switch accounts
        self.driver = None
        self.tweets_data = []
        self.total_tweets_processed = 0
        self.filename = f"{ticker}_tweets.csv"
        self.current_account = self.accounts[self.account_index]  
        self._login_driver()  

    def _setup_driver(self):
        options = uc.ChromeOptions()
        if self.headless:
            options.headless = True
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-images")
        options.page_load_strategy = 'eager'
        
        return uc.Chrome(options=options)

    def _login_driver(self):
        """
        Logs into Twitter with the current account.
        """
        self.driver = self._setup_driver()
        username, password = self.current_account
        self.driver.get("https://twitter.com/i/flow/login")
        sleep(random.uniform(4, 6))  # randomized delay
        username_field = self.driver.find_element(By.NAME, "text")
        username_field.send_keys(username)
        username_field.send_keys(Keys.RETURN)
        sleep(random.uniform(2, 3))

        password_field = self.driver.find_element(By.NAME, "password")
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
        sleep(random.uniform(3, 5))

    def switch_account(self):
        """
        Switch to the next account in the rotation.
        """
        self.driver.quit()  
        self.account_index = (self.account_index + 1) % len(self.accounts)  
        self.current_account = self.accounts[self.account_index]  
        self._login_driver() 

    def save_batch(self):
        if self.tweets_data:
            df = pd.DataFrame(self.tweets_data)
            # Save with header for the first batch
            if self.total_tweets_processed == 0:
                df.to_csv(self.filename, index=False)
            else:
                df.to_csv(self.filename, mode='a', header=False, index=False)
            self.total_tweets_processed += len(self.tweets_data)
            print(f"Saved batch of {len(self.tweets_data)} tweets. Total tweets: {self.total_tweets_processed}")
            self.tweets_data = []

    def scrape_day(self, day):
        """
        Scrape tweets for a single day using since/until parameters.
        """
        day_end = day + timedelta(days=1)
        query = f"${self.ticker} since:{day.strftime('%Y-%m-%d')} until:{day_end.strftime('%Y-%m-%d')} lang:en"
        search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
        self.driver.get(search_url)
        sleep(random.uniform(2, 4))  # randomized delay after loading the page
        
        day_tweets = []
        no_new_tweets_count = 0
        last_position = 0

        while True:
            tweets = self.driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
            for tweet in tweets:
                current_total = self.total_tweets_processed + len(self.tweets_data) + len(day_tweets)
                if self.max_tweets and current_total >= self.max_tweets:
                    return day_tweets
                try:
                    date_element = tweet.find_element(By.TAG_NAME, "time")
                    date_str = date_element.get_attribute("datetime")
                    tweet_date = datetime.fromisoformat(date_str.split('T')[0]).date()
                    if tweet_date != day:
                        continue
                    content_elements = tweet.find_elements(By.XPATH, './/div[@data-testid="tweetText"]//span')
                    content = ' '.join([elem.text for elem in content_elements])
                    tweet_dict = {"content": content, "date": date_str}
                    if tweet_dict not in day_tweets:
                        day_tweets.append(tweet_dict)
                        if self.tweets_per_day and len(day_tweets) >= self.tweets_per_day:
                            break
                except (NoSuchElementException, StaleElementReferenceException):
                    continue

            if self.tweets_per_day and len(day_tweets) >= self.tweets_per_day:
                break
            if self.max_tweets and (self.total_tweets_processed + len(self.tweets_data) + len(day_tweets)) >= self.max_tweets:
                break

            # Scroll to load more tweets with a natural delay.
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(random.uniform(1.5, 3))
            current_position = self.driver.execute_script("return window.pageYOffset;")
            if current_position == last_position:
                no_new_tweets_count += 1
                if no_new_tweets_count >= 3:
                    break
            else:
                no_new_tweets_count = 0
            last_position = current_position

        return day_tweets

    def scrape_tweets(self):
        """
        Iterates day by day (starting from date_limit up to today) and scrapes tweets.
        """
        current_day = self.date_limit
        today = datetime.now().date()
        days_scraped = 0
        
        while current_day <= today:
            if self.max_tweets and (self.total_tweets_processed + len(self.tweets_data)) >= self.max_tweets:
                break

            print(f"Scraping tweets for {current_day}...")
            day_tweets = self.scrape_day(current_day)
            if day_tweets:
                print(f"Found {len(day_tweets)} tweets for {current_day}")
                self.tweets_data.extend(day_tweets)
            else:
                print(f"No tweets found for {current_day}")
                
            if len(self.tweets_data) >= self.batch_size:
                self.save_batch()

            days_scraped += 1
            if days_scraped >= self.account_switch_interval:  
                self.switch_account()
                days_scraped = 0  
                
            current_day += timedelta(days=1)
        
        if self.tweets_data:
            self.save_batch()

    def close(self):
        self.driver.quit()
