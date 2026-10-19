import logging
import sys

def get_logger(name = "bioacoustic_ssl", log_file = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt = "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    logger = get_logger("test_logger")
    logger.info("Logger initialized successfully.")