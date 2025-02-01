from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Predicting Stock Prices with BERTOPIC and Sentiment Analysis",
    description="API for web scraping, data processing, and model inference",
    version="1.0.0",
)


# Index Endpoint
@app.get("/")
async def index():
    return {
        "message": "Welcome to the Finance Quantitative API!",
        "docs_url": "/docs",
        "endpoints": {
            "/health": "Check API health",
            "/scrape": "Launch web scraping tasks",
            "/predict": "Run model predictions",
        },
    }


# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "Healthy", "details": "API is running smoothly"}


# Web Scraping Endpoint
class ScrapeRequest(BaseModel):
    platform: str  # Options like 'reddit', 'yfinance'
    parameters: dict  # Example: {"ticker": "TSLA", "days": 30}


@app.post("/scrape")
async def scrape_data(request: ScrapeRequest):
    if request.platform == "yfinance":
        # Call your scraping_yfinance functions
        from bertopic_project.data.scraping_yfinance.scraper import scrape

        result = scrape(**request.parameters)
    elif request.platform == "reddit":
        # Call your scraping_reddit functions
        from bertopic_project.data.scraping_reddit.scraper_praw import scrape

        result = scrape(**request.parameters)
    else:
        raise HTTPException(status_code=400, detail="Unsupported platform")
    return {"status": "Success", "data": result}


# # Model Inference Endpoint
# class PredictionRequest(BaseModel):
#     text: str
#     model: str  # 'FINBERT', 'VADER', etc.

# @app.post("/predict")
# async def predict(request: PredictionRequest):
#     if request.model.lower() == "vader":
#         from bertopic_project.preprocessing.sentiment_analysis_VADER import analyze
#         result = analyze(request.text)
#     elif request.model.lower() == "finbert":
#         from bertopic_project.preprocessing.sentiment_analysis_FINBERT import analyze
#         result = analyze(request.text)
#     else:
#         raise HTTPException(status_code=400, detail="Model not supported")
#     return {"status": "Success", "prediction": result}