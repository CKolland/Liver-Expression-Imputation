import logging
import sys


def setup_logging(path_to_log: str) -> logging.Logger:
    """Set up and configure logging to both console and file.

    :param str path_to_log: Path to the log file where logs will be written

    :return: Configured logger instance
    :rtype: logging.Logger
    """

    logger = logging.getLogger("data_integration")
    logger.setLevel(logging.DEBUG)  # Lowest log level (logs everything)

    # Custom formatter
    formatter = logging.Formatter(
        "%(asctime)s || LEVEL: %(levelname)s |> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set handler specific log level
    console_handler.setFormatter(formatter)  # Add custom formatter to handler
    logger.addHandler(console_handler)  # Add handler to the logger

    # Add file handler
    file_handler = logging.FileHandler(path_to_log)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
