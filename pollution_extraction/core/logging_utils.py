"""Logging utilities for pollution extraction package."""

import logging


def get_logger(name: str = "pollution_extraction") -> logging.Logger:
    """Get a logger instance for the pollution extraction package.

    Parameters
    ----------
    name : str
        The name for the logger, defaults to 'pollution_extraction'

    Returns
    -------
    logging.Logger
        Configured logger instance

    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Set up basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logger


# Create a default logger instance
logger = get_logger()
