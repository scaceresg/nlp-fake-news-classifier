from pydantic import BaseModel, Field


class NewsText(BaseModel):
    """Schema for text input"""

    text: str = Field(
        ...,
        description="News text to analyze",
        min_length=5,
        max_length=10000,
        example="Breaking news: Major event happened!",
    )
    classif_model: str | None = Field(
        default="lr",
        description="Model to use for classification: 'lr' or 'svm'",
    )
    vectorizer: str | None = Field(
        default="count",
        description="Vectorization method to use: 'count' or 'tfidf'",
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
    classification: list[ClassificationResponse]
    sentiment_analysis: list[SentimentResponse]
