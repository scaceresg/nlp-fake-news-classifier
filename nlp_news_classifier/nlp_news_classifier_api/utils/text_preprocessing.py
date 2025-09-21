import re
import pandas as pd
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from utils.utils import logger

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


class TextPreprocessor:
    """
    A class for preprocessing text data for NLP tasks.
    """

    def __init__(self):
        self.en_stopwords = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()
        self.count_vect = CountVectorizer()

        logger.info("TextPreprocessor initialized.")

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the input text by lowercasing, removing special characters,
        tokenizing, removing stopwords, and lemmatizing.

        Parameters:
        -----------
            text (str): The input text to preprocess.

        Returns:
        --------
            List[str]: A list of cleaned and lemmatized tokens.
        """

        try:
            # Remove hyphens at the beginning of text
            text = re.sub(r"^[^-]*-\s", "", text)

            # Lowercase the text
            text = text.lower()

            # Remove special characters
            text = re.sub(r"([^\w\s])", "", text)

            # Remove stopwords
            text = " ".join(
                [word for word in text.split() if word not in self.en_stopwords]
            )

            # Tokenize the text
            tokens = word_tokenize(text)

            # Lemmatize the tokens
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

            return tokens

        except Exception as e:
            logger.error(f"Error in preprocessing the text: {e}")
            raise

    def preprocess_training_data(
        self, text_df: pd.DataFrame, text_column: str
    ) -> pd.DataFrame:
        """
        Preprocess the text in training dataset.

        Parameters:
        -----------
            text_df (pd.DataFrame): The input dataframe containing text data.
            text_column (str): The name of the column containing text data.

        Returns:
        --------
            pd.DataFrame: A dataframe containing the bag-of-words representation of the input text data.
        """

        try:
            logger.info("Preprocessing text in training dataset")

            # Preprocess the training dataset
            text_df["clean_text"] = text_df[text_column].apply(self.preprocess_text)

            # Prepare array of preprocessed text
            cleaned_text = [",".join(map(str, l)) for l in text_df["clean_text"]]

            return cleaned_text

        except Exception as e:
            logger.error(f"Error in preprocessing the training data: {e}")
            raise
