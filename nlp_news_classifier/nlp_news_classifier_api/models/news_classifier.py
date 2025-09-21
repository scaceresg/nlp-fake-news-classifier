import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from vectorizer import TextVectorizer
from utils.text_preprocessing import TextPreprocessor
from utils.gcp import GCPStorageManager
from utils.utils import logger


class NewsClassifier:
    """
    A class for training and managing text classifiers for fake news detection.
    """

    def __init__(self, classifier_type: str = "lr", vectorization_method: str = "count"):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TextVectorizer(method=vectorization_method)
        self.gcs_manager = GCPStorageManager()

        # Check if model exists in GCP bucket
        model_exists = self.gcs_manager.model_exists(model_type=classifier_type)

        if model_exists:
            logger.info(f"A pre-trained {classifier_type} model was found in GCP bucket!")
            self.classifier = self.gcs_manager.load_model(model_type=classifier_type)
        else:
            logger.info(
                f"No pre-trained model found. Initializing a new {classifier_type} model."
            )

            if classifier_type == "lr":
                self.classifier = LogisticRegression(max_iter=1000, random_state=42)
            else:
                self.classifier = SGDClassifier(max_iter=1000, random_state=42)

            self.train_and_evaluate_classifier()

            self.gcs_manager.upload_model(
                model=self.classifier, model_type=classifier_type
            )

        logger.info("NewsClassifier class initialized successfully.")

    def train_and_evaluate_classifier(self, test_size: float = 0.2) -> None:
        """
        Train the text classifier.

        Parameters:
        -----------
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
        --------
            Model is trained and evaluated; no return value.
        """

        try:
            logger.info("Starting classifier training process...")

            # Get the training data
            raw_df = self._get_training_data()

            y = raw_df["fake_or_factual"]

            # Preprocess the text data
            X_input = self.preprocessor.preprocess_training_data(
                text_df=raw_df, text_column="text"
            )

            # Vectorize the preprocessed text data
            bow_df = self.vectorizer.fit_transform_text(X_input)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                bow_df, y, test_size=test_size, random_state=42
            )

            # Train the classifier
            self.classifier.fit(X_train, y_train)

            logger.info("Classifier training completed successfully.")

            self._evaluate_classifier(X_test, y_test)

        except Exception as e:
            logger.error(f"Error in training the classifier: {e}")
            raise

    def predict(self, text: str) -> str:
        """
        Predict if the input text is fake or factual news.

        Parameters:
        -----------
            text (str): The text document to classify.

        Returns:
        --------
            str: The predicted class label ('Fake News' or 'Factual News').
        """
        try:
            logger.info("Starting prediction process...")

            # Preprocess the input text
            preprocessed_text = self.preprocessor.preprocess_text(text)

            # Vectorize the preprocessed text
            bow_df = self.vectorizer.transform_text(preprocessed_text)

            # Make prediction
            prediction = self.classifier.predict(bow_df)

            logger.info("Prediction completed successfully.")

            return prediction

        except Exception as e:
            logger.error(f"Error in making prediction: {e}")
            raise

    def _evaluate_classifier(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluate the classifier's performance on the test set.

        Parameters:
        -----------
            X_test (pd.DataFrame): The feature set for testing.
            y_test (pd.Series): The true labels for the test set.

        Returns:
        --------
            Model is evaluated; no return value.
        """

        try:
            logger.info("Evaluating classifier performance on the test set...")

            y_pred = self.classifier.predict(X_test)
            report = classification_report(y_test, y_pred)

            logger.info(f"Classifier Evaluation Report:\n{report}")

        except Exception as e:
            logger.error(f"Error in evaluating the classifier: {e}")
            raise
