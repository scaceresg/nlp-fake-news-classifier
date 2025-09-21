from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.utils import logger


class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on text data using VADER.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("SentimentAnalyzer initialized.")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of the input text.

        Parameters:
        -----------
            text (str): The input text to analyze.

        Returns:
        --------
            dict: A dictionary containing the sentiment scores (positive, negative, neutral, compound).
        """

        try:
            logger.info("Analyzing sentiment of the input text")
            
            compound_score = self.analyzer.polarity_scores(text)["compound"]

            if compound_score <= -0.1:
                sentiment_label = "negative"
            elif compound_score >= 0.1:
                sentiment_label = "positive"
            else:
                sentiment_label = "neutral"
                
            return {
                "overall_sentiment": sentiment_label,
                "compound_score": compound_score
            }

        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise