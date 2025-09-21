from typing import List
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils.utils import logger
from utils.gcp import GCPStorageManager


class TextVectorizer:
    """
    A class for vectorizing text data using CountVectorizer or TfidfVectorizer.
    """

    def __init__(self, method="count"):
        self.gcs_manager = GCPStorageManager()
        self.vectorizer_method = method

        self.vectorizer_exists = self.gcs_manager.vectorizer_exists(
            vectorizer_type=method
        )

        if self.vectorizer_exists:
            logger.info(f"A pre-fitted {method} vectorizer was found in GCP bucket.")
            self.vectorizer = self.gcs_manager.load_vectorizer(method)
        else:
            logger.info(
                f"No pre-fitted vectorizer found. Initializing a new {method} vectorizer."
            )

            if method == "count":
                self.vectorizer = CountVectorizer()
            elif method == "tfidf":
                self.vectorizer = TfidfVectorizer()
            else:
                raise ValueError("Unknown vectorization method")

        logger.info(f"TextVectorizer initialized with method: {method}")

    def fit_transform_text(self, texts: List[str]) -> pd.DataFrame:
        """
        Fit the vectorizer to the texts and transform them into vectorized form.

        Parameters:
        -----------
            texts (List[str]): A list of text documents to vectorize.

        Returns:
        --------
            pd.DataFrame: A dataframe containing the bag-of-words representation of the input texts.
        """

        try:
            logger.info("Fitting and transforming texts using the vectorizer")

            bag_of_words_df = pd.DataFrame(
                self.vectorizer.fit_transform(texts).toarray(),
                columns=self.vectorizer.get_feature_names_out(),
            )

            if not self.vectorizer_exists:
                self.gcs_manager.upload_vectorizer(
                    vectorizer=self.vectorizer, vectorizer_type=self.vectorizer_method
                )

            logger.info("Text vectorization completed successfully.")

            return bag_of_words_df

        except Exception as e:
            logger.error(f"Error in fitting and transforming texts: {e}")
            raise

    def transform_text(self, texts: List[str]) -> pd.DataFrame:
        """
        Transform texts into vectorized form using the already fitted vectorizer.

        Parameters:
        -----------
            texts (List[str]): A list of text documents to vectorize.

        Returns:
        --------
            pd.DataFrame: A dataframe containing the bag-of-words representation of the input texts.
        """

        try:
            logger.info("Transforming texts using the fitted vectorizer")

            bag_of_words_df = pd.DataFrame(
                self.vectorizer.transform(texts).toarray(),
                columns=self.vectorizer.get_feature_names_out(),
            )

            logger.info("Text transformation completed successfully.")

            return bag_of_words_df

        except Exception as e:
            logger.error(f"Error in transforming texts: {e}")
            raise
