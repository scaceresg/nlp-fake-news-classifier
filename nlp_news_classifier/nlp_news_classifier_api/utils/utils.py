import os
import sys
import logging
import time
from functools import wraps


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with the specified name and level.

    Parameters:
    ----------
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.

    Returns:
    ---------
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger(name="nlp_news_classifier", level=logging.INFO)


def get_absolute_path(relative_path: str) -> str:
    """
    Get the absolute path of a file given its relative path.

    Parameters:
    ----------
        relative_path (str): The relative path to the file. For example, "../data/fake_news_data.csv".

    Returns:
    ---------
        str: The absolute path to the file.
    """

    rel_path_list = relative_path.split("/")
    base_path = os.path.dirname(os.path.abspath(__file__))
    abs_path = base_path

    for path_part in rel_path_list:
        if path_part == "..":
            abs_path = os.path.dirname(abs_path)
        else:
            abs_path = os.path.join(abs_path, path_part)

    return abs_path


def timer(func):
    """
    Decorator to measure the execution time of a function.

    Parameters:
    ----------
        func (callable): The function to be decorated.

    Returns:
    ---------
        callable: The wrapped function with timing functionality.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )

        return result

    return wrapper
