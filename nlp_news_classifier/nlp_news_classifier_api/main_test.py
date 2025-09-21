from utils.utils import get_absolute_path, logger
from utils.classifier_training import ClassifierTrainer
from utils.text_preprocessing import TextPreprocessor


logger.info("Starting the NLP News Classifier module...")

# Example usage of TextPreprocessor
text = "Breaking news: AI is transforming the world!"
preprocessor = TextPreprocessor()
tokens = preprocessor.preprocess_text(text)
vectorized_text = preprocessor.vectorize_text(text)
logger.info(f"Preprocessed tokens: {tokens}")
logger.info(f"Vectorized text: {vectorized_text}")

# Example usage of ClassifierTrainer
trainer = ClassifierTrainer()
X, y = trainer.train_classifier()
logger.info("Classifier training completed.")
