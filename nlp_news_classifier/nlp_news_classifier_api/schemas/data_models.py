from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class NewsText(BaseModel):
    """Schema for text input"""

    text: str = Field(
        ...,
        description="News text to analyze",
        min_length=5,
        max_length=10000,
        example="Breaking news: Major event happened!",
    )
    classif_model: Optional[str] = Field(
        default="lr",
        description="Model to use for classification: logistic regression (lr) or support vector machine (svm)",
    )
    vectorizer: Optional[str] = Field(
        default="count",
        description="Vectorization method to use: count vectorization (count) or TF-IDF vectorization (tfidf)",
    )


class ClassificationResponse(BaseModel):
    """Schema for classification response"""

    prediction: str
    classifier: str
    vectorizer: str


class SentimentResponse(BaseModel):
    """Schema for sentiment analysis response"""

    sentiment_label: str
    sentiment_score: float


class ClassificationAndSentimentResponse(BaseModel):
    """Schema for combined classification and sentiment analysis response"""

    text: str
    classification: List[ClassificationResponse]
    sentiment_analysis: List[SentimentResponse]