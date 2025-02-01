from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options

import pandas as pd
import time
import os


def handle_cookie_consent(driver):
    """Handle the cookie consent popup if it appears"""
    try:
        # Wait for the cookie button with a short timeout
        accept_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Refuser tout']"))
        )
        accept_button.click()
        time.sleep(1)  # Small delay to let the popup close
    except (TimeoutException, NoSuchElementException):
        # If no popup appears, just continue
        pass


def get_stock_history(stocks_dict, output_dir="raw"):
    """Get historical data for selected stocks"""
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(30)  # Set page load timeout
    all_data = {}

    try:
        for ticker, name in stocks_dict.items():
            retries = 3  # Number of retries for each stock
            while retries > 0:
                try:
                    url = f"https://finance.yahoo.com/quote/{ticker}/history/?period1=1580341468&period2=1738185894"
                    driver.get(url)

                    handle_cookie_consent(driver)

                    table = WebDriverWait(driver, 20).until(  # Increased wait time
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "table[class*='yf-']")
                        )
                    )

                    # Rest of your code...
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    data = []
                    for row in rows[1:]:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if cols:
                            row_data = [col.text for col in cols]
                            data.append(row_data)

                    if data:  # Check if we got any data
                        df = pd.DataFrame(
                            data,
                            columns=[
                                "Date",
                                "Open",
                                "High",
                                "Low",
                                "Close",
                                "Adj Close",
                                "Volume",
                            ],
                        )
                        df["Ticker"] = ticker
                        df["Company_Name"] = name

                        # Make sure output_dir exists
                        os.makedirs(output_dir, exist_ok=True)

                        file_path = os.path.join(output_dir, f"{ticker}_history.csv")
                        df.to_csv(file_path, index=False)
                        all_data[ticker] = df

                        print(f"Successfully retrieved data for {ticker} ({name})")
                        break  # Break the retry loop if successful

                except TimeoutException:
                    print(f"Timeout for {ticker}, retries left: {retries-1}")
                    retries -= 1
                    if retries == 0:
                        print(f"Failed to fetch data for {ticker} after all retries")
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")
                    break  # Break on non-timeout errors

                time.sleep(2)  # Wait between retries

    finally:
        driver.quit()

    return all_data
