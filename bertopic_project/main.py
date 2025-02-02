from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from typing import Dict
import psutil
import time
import os
from datetime import datetime

# Import the scraper
from data_extraction.scraping_yfinance.tesla_scraper import TeslaStockScraper

app = FastAPI(
    title="Tesla Data Analysis API",
    description="API for scraping and analyzing Tesla stock data",
    version="1.0.0"
)

@app.get("/", tags=["General"])
async def index():
    """
    Root endpoint providing application information and available endpoints.
    """
    return {
        "name": "Tesla Data Analysis API",
        "version": "1.0.0",
        "description": "API for scraping and analyzing Tesla stock data from various sources",
        "documentation": "/docs",  # Swagger UI documentation
        "endpoints": {
            "GET /": "This index page with API information",
            "GET /health": "Health check endpoint with system metrics",
            "POST /api/scrape/tesla-stock": "Scrape Tesla stock data from Yahoo Finance"
        },
        "developer": "Your Name",
        "last_updated": "2024-02-02"
    }

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint providing system metrics and API status.
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check if data directory exists and is accessible
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "data_extraction", "raw")
        data_access = os.access(data_dir, os.R_OK | os.W_OK)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent
                }
            },
            "service_status": {
                "api_available": True,
                "data_directory_accessible": data_access
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/scrape/tesla-stock", tags=["Web Scraping"])
async def scrape_tesla_stock():
    """
    Execute web scraping for Tesla stock data from Yahoo Finance.
    
    Returns:
        JSON: Scraped stock data or error message
    """
    try:
        # Initialize scraper
        scraper = TeslaStockScraper()
        
        # Record start time
        start_time = time.time()
        
        # Execute scraping
        data = scraper.scrape_stock_data()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        if data is not None:
            # Get file path where data was saved
            file_path = os.path.join(scraper.output_dir, "tesla_stock_history.csv")
            
            # Prepare response
            response = {
                "status": "success",
                "execution_time_seconds": round(execution_time, 2),
                "data_info": {
                    "rows": len(data),
                    "columns": list(data.columns),
                    "date_range": {
                        "start": data['Date'].min(),
                        "end": data['Date'].max()
                    },
                    "file_saved": file_path
                },
                "sample_data": data.head(5).to_dict(orient='records')
            }
            
            return JSONResponse(content=response)
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch Tesla stock data")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

# Optional: Add configuration endpoint
@app.get("/api/config", tags=["System"])
async def get_config():
    """
    Get API configuration and environment information.
    """
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "data_directories": {
            "raw": os.path.join("data_extraction", "raw"),
            "processed": os.path.join("data_preprocessing", "processed_data")
        },
        "supported_data_sources": ["Yahoo Finance"],
        "max_retries": 3,
        "timeout_seconds": 30
    }

def run_api():
    """Run the FastAPI application"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_api()