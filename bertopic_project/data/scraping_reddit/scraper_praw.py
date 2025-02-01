import os
from dotenv import load_dotenv
import praw
import pandas as pd
import time
import re
from datetime import datetime
import logging
from typing import List, Dict, Set
from dataclasses import dataclass
from ratelimit import limits, sleep_and_retry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockMention:
    date: datetime.date
    time: datetime.time
    subreddit: str
    title: str
    text: str
    ticker: str
    company_name: str


class RedditStockScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize the Reddit scraper with API credentials."""
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )

        self.stocks = {"META": "Meta Platforms, Inc.", "TSLA": "Tesla"}

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

    def _validate_dates(self, start_date: str, end_date: str) -> tuple:
        """Validate and convert date strings to timestamps."""
        try:
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            if start_timestamp > end_timestamp:
                raise ValueError("Start date must be before end date")
            return start_timestamp, end_timestamp
        except ValueError as e:
            logger.error(f"Date validation error: {e}")
            raise

    def _extract_stock_mentions(self, text: str) -> Set[str]:
        """Extract stock mentions from text using multiple patterns."""
        if not text:
            return set()

        mentions = set()

        # Find $TICKER mentions
        dollar_mentions = set(re.findall(r"\$([A-Z]{1,5})", text))

        # Find company name mentions
        company_mentions = {
            ticker
            for ticker, company in self.stocks.items()
            if company.lower() in text.lower()
        }

        # Find raw ticker mentions
        ticker_mentions = {
            ticker for ticker in self.stocks.keys() if ticker in text.upper()
        }

        mentions.update(dollar_mentions, company_mentions, ticker_mentions)
        return {m for m in mentions if m in self.stocks}

    @sleep_and_retry
    @limits(calls=60, period=60)  # Reddit API rate limit
    def _fetch_subreddit_posts(
        self, subreddit_name: str, start_timestamp: int, end_timestamp: int
    ) -> List[StockMention]:
        """Fetch posts from a subreddit within the specified timeframe."""
        mentions = []
        subreddit = self.reddit.subreddit(subreddit_name)

        search_query = " OR ".join(
            f'({ticker} OR "{company}")' for ticker, company in self.stocks.items()
        )

        try:
            for submission in subreddit.search(
                search_query, syntax="lucene", time_filter="all", sort="new", limit=None
            ):

                if submission.created_utc < start_timestamp:
                    break

                if submission.created_utc <= end_timestamp:
                    text = f"{submission.title} {submission.selftext}"
                    stock_mentions = self._extract_stock_mentions(text)

                    timestamp = datetime.fromtimestamp(submission.created_utc)

                    for ticker in stock_mentions:
                        mentions.append(
                            StockMention(
                                date=timestamp.date(),
                                time=timestamp.time(),
                                subreddit=subreddit_name,
                                title=submission.title,
                                text=submission.selftext,
                                ticker=ticker,
                                company_name=self.stocks[ticker],
                            )
                        )

        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")
            raise

        return mentions

    def get_posts_by_timeframe(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get all stock mentions within the specified timeframe.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame containing all stock mentions
        """
        start_timestamp, end_timestamp = self._validate_dates(start_date, end_date)
        all_mentions = []

        for subreddit_name in self.subreddits:
            try:
                logger.info(f"Scraping r/{subreddit_name}...")
                mentions = self._fetch_subreddit_posts(
                    subreddit_name, start_timestamp, end_timestamp
                )
                all_mentions.extend(mentions)
                logger.info(f"Found {len(mentions)} mentions in r/{subreddit_name}")

            except Exception as e:
                logger.error(f"Failed to scrape r/{subreddit_name}: {e}")
                continue

        return pd.DataFrame([vars(mention) for mention in all_mentions])


if __name__ == "__main__":
    load_dotenv()

    # Get the path to bertopic_project/data/raw
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get scraping_reddit dir
    project_dir = os.path.dirname(
        os.path.dirname(current_dir)
    )  # Go up to bertopic_project
    raw_dir = os.path.join(project_dir, "data", "raw")  # Get raw directory

    scraper = RedditStockScraper(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )

    df = scraper.get_posts_by_timeframe("2024-01-01", "2025-01-31")
    print(f"Total mentions found: {len(df)}")

    # Save to project's raw directory
    output_path = os.path.join(raw_dir, "reddit_data.csv")
    df.to_csv(output_path, index=False)
