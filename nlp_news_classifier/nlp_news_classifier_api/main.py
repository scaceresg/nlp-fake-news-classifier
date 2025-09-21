from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager

from models.news_classifier import NewsClassifier
from schemas.data_models import (
    NewsText,
    ClassificationResponse,
    SentimentResponse,
    ClassificationAndSentimentResponse,
)
from utils.utils import logger

try:
    from nlp_news_classifier._version import __version__
except ImportError:
    __version__ = "dev"

app = FastAPI(
    title="NLP Fake News Classifier API",
    description="API for fake news classification, sentiment analysis, and topic modeling",
    version=__version__,
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load or initialize classifier model on startup
cls_model = None


@asynccontextmanager
async def lifespan(app):
    global cls_model
    logger.info(
        "Application startup! Initializing a pre-trained or new classification model."
    )

    cls_model = NewsClassifier()  # Replace with actual model loading code

    yield

    logger.info("Application shutdown! Cleaning up resources.")
    cls_model = None


@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "api": "nlp-fake-news-classifier", "version": __version__}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(input_text: NewsText):
    """
    Classify the input text as fake or factual news.

    Parameters:
    -----------
        input_data (NewsText): The input text and optional parameters for classification.

    Returns:
    --------
        ClassificationResponse: The classification result including prediction, classifier, and vectorizer used.
    """

    global cls_model

    if cls_model is None:
        logger.error("Classifier model is not initialized.")
        raise HTTPException(
            status_code=500, detail="Classifier model is not initialized."
        )

    prediction = cls_model.predict(
        text=input_text.text,
        classif_model=input_text.classif_model,
        vectorizer=input_text.vectorizer,
    )

    response = ClassificationResponse(
        prediction=prediction,
        classifier=input_text.classif_model,
        vectorizer=input_text.vectorizer,
    )

    return response
