from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import os

class TeslaStockScraper:
    def __init__(self):
        """Initialize the scraper with output directory configuration"""
        # Get the absolute path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to data directory and then to raw
        data_dir = os.path.dirname(current_dir)  # up to data directory
        self.output_dir = os.path.join(data_dir, 'raw')
        self.ensure_output_directory()
        
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_driver(self):
        """Configure and return Chrome WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # comment below lines for headless mode if needed
        chrome_options.add_argument("--headless")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        return driver
    
    def handle_cookie_consent(self, driver):
        """Handle the cookie consent popup if it appears"""
        try:
            accept_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Refuser tout']"))
            )
            accept_button.click()
            time.sleep(1)
        except (TimeoutException, NoSuchElementException):
            pass
    
    def scrape_stock_data(self):
        """Scrape Tesla stock historical data"""
        driver = self.setup_driver()
        data = None
        
        try:
            # Yahoo Finance URL for Tesla stock history
            url = "https://finance.yahoo.com/quote/TSLA/history/?period1=1704067200&period2=1738281600"
            driver.get(url)
            
            self.handle_cookie_consent(driver)
            
            # Wait for the table to load
            table = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table[class*='yf-']"))
            )
            
            # Extract data from table
            rows = table.find_elements(By.TAG_NAME, "tr")
            stock_data = []
            
            for row in rows[1:]:  # Skip header row
                cols = row.find_elements(By.TAG_NAME, "td")
                if cols:
                    row_data = [col.text for col in cols]
                    stock_data.append(row_data)
            
            if stock_data:
                # Convert to DataFrame
                data = pd.DataFrame(
                    stock_data,
                    columns=[
                        "Date",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume"
                    ]
                )
                
                # Add ticker and company name columns
                data["Ticker"] = "TSLA"
                data["Company_Name"] = "Tesla, Inc."
                                
                # Save to CSV
                output_path = os.path.join(self.output_dir, "tesla_stock_history.csv")
                data.to_csv(output_path, index=False)
                print(f"Successfully saved Tesla stock data to {output_path}")
                
        except Exception as e:
            print(f"Error scraping Tesla stock data: {e}")
            raise
        
        finally:
            driver.quit()
            
        return data

def main():
    """Main function to run the scraper"""
    scraper = TeslaStockScraper()
    try:
        data = scraper.scrape_stock_data()
        if data is not None:
            print(f"Successfully retrieved {len(data)} records of Tesla stock data")
            return data
    except Exception as e:
        print(f"Failed to scrape Tesla stock data: {e}")
        return None

if __name__ == "__main__":
    main()