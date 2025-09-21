import os
import joblib
import tempfile
from google.cloud import storage

from utils.utils import logger

ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev").lower()
BUCKET_NAME = f"news_classifier_{ENVIRONMENT}"


class GCPStorageManager:
    """
    A class to manage Google Cloud Storage operations for the NLP News Classifier project.
    """

    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(BUCKET_NAME)

    def model_exists(self, model_type: str) -> bool:
        """
        Check if a model file exists in the GCP bucket.

        Parameters:
        -----------
            model_type (str): The name of the classifier model type: 'lr' for Logistic 
                Regression, 'svm' for Support Vector Machine.

        Returns:
        --------
            bool: True if the model file exists, False otherwise.
        """

        blob_name = f"models/model_{model_type}.pkl"
        
        try:
            
            logger.info(
                f"Checking if a classifier model exists in bucket directory: 'gs://{BUCKET_NAME}/models/**'"
            )

            blob = self.bucket.blob(blob_name)
            
            return blob.exists()
        
        except Exception as e:
            logger.error(f"Error checking model existence in bucket: {e}")
            raise
        
    def vectorizer_exists(self, vectorizer_type: str) -> bool:
        """
        Check if a vectorizer file exists in the GCP bucket.

        Parameters:
        -----------
            vectorizer_type (str): The name of the vectorizer type: 'count' for Count 
                Vectorizer, 'tfidf' for TF-IDF Vectorizer.

        Returns:
        --------
            bool: True if the vectorizer file exists, False otherwise.
        """

        blob_name = f"vectorizers/vectorizer_{vectorizer_type}.pkl"

        try:
            
            logger.info(
                f"Checking if vectorizer exists in bucket directory: 'gs://{BUCKET_NAME}/vectorizers/**'"
            )
            
            blob = self.bucket.blob(blob_name)
            
            return blob.exists()
        
        except Exception as e:
            logger.error(f"Error checking vectorizer existence in bucket: {e}")
            raise

    def upload_model(self, model, model_type: str) -> None:
        """
        Upload a trained model to the GCP bucket.

        Parameters:
        -----------
            model: The trained model object to upload.
            model_type (str): The type of the model (e.g., 'lr' for Logistic Regression, 
                'svm' for Support Vector Machine).
        """

        blob_name = f"models/model_{model_type}.pkl"

        try:
            
            logger.info(f"Uploading classifier model to GCP bucket: 'gs://{BUCKET_NAME}/{blob_name}'")
        
            blob = self.bucket.blob(blob_name)

            # Save the model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                joblib.dump(model, temp_file.name)
                blob.upload_from_filename(temp_file.name)

            # Remove the temporary file
            os.remove(temp_file.name)

            logger.info("Model uploaded successfully.")
        
        except Exception as e:
            logger.error(f"Error uploading model to bucket: {e}")
            raise

    def upload_vectorizer(self, vectorizer, vectorizer_type: str) -> None:
        """
        Upload a vectorizer to the GCP bucket.

        Parameters:
        -----------
            vectorizer: The vectorizer object to upload.
            vectorizer_type (str): The type of the vectorizer (e.g., 'count' for Count 
                Vectorizer, 'tfidf' for TF-IDF Vectorizer).
        """

        blob_name = f"vectorizers/vectorizer_{vectorizer_type}.pkl"

        try:
            
            logger.info(
                f"Uploading vectorizer to GCP bucket: 'gs://{BUCKET_NAME}/{blob_name}'"
            )
            
            blob = self.bucket.blob(blob_name)

            # Save the vectorizer to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                joblib.dump(vectorizer, temp_file.name)
                blob.upload_from_filename(temp_file.name)

            # Remove the temporary file
            os.remove(temp_file.name)

            logger.info("Vectorizer uploaded successfully.")
        
        except Exception as e:
            logger.error(f"Error uploading vectorizer to bucket: {e}")
            raise
        
    def load_model(self, model_type: str):
        """
        Load a model from the GCP bucket.

        Parameters:
        -----------
            model_type (str): The type of the model to load (e.g., 'lr' for Logistic 
                Regression, 'svm' for Support Vector Machine).

        Returns:
        --------
            The loaded model object.
        """

        blob_name = f"models/model_{model_type}.pkl"

        try:
            logger.info(f"Attempting to load classifier model from GCP bucket: 'gs://{BUCKET_NAME}/{blob_name}'")
            
            blob = self.bucket.blob(blob_name)

            # Download the model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                blob.download_to_filename(temp_file.name)
                model = joblib.load(temp_file.name)

            # Remove the temporary file
            os.remove(temp_file.name)

            logger.info("Model loaded successfully.")

            return model
        
        except Exception as e:
            logger.error(f"Error loading model from bucket: {e}")
            raise

    def load_vectorizer(self, vectorizer_type: str):
        """
        Load a vectorizer from the GCP bucket.

        Parameters:
        -----------
            vectorizer_type (str): The type of the vectorizer to load (e.g., 'count' for 
                Count Vectorizer, 'tfidf' for TF-IDF Vectorizer).

        Returns:
        --------
            The loaded vectorizer object.
        """

        blob_name = f"vectorizers/vectorizer_{vectorizer_type}.pkl"

        try:
            logger.info(
                f"Attempting to load vectorizer from GCP bucket: 'gs://{BUCKET_NAME}/{blob_name}'"
            )   
            
            blob = self.bucket.blob(blob_name)

            # Download the vectorizer to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                blob.download_to_filename(temp_file.name)
                vectorizer = joblib.load(temp_file.name)

            # Remove the temporary file
            os.remove(temp_file.name)

            logger.info("Vectorizer loaded successfully.")

            return vectorizer
        
        except Exception as e:
            logger.error(f"Error loading vectorizer from bucket: {e}")
            raise