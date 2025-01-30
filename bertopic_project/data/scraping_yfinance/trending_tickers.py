from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

import time

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

def get_trending_tickers():
    """Get list of trending tickers from Yahoo Finance"""
    driver = webdriver.Chrome()
    try:
        driver.get("https://finance.yahoo.com/markets/stocks/trending/")
        
        handle_cookie_consent(driver)
        
        table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table[data-testid='table-container']"))
        )
        
        rows = table.find_elements(By.TAG_NAME, "tr")
        trending_stocks = {}
        
        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, "td")
            if cols:
                ticker = cols[0].text
                name = cols[1].text
                trending_stocks[ticker] = name
                
        return trending_stocks
    
    finally:
        driver.quit()