"""Logging config.

This file provides logging configurations.
"""

from logging.config import dictConfig

from src.pos_classifier.config.config import LOG_PATH

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_PATH,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "default",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["file", "console"],
        },
    },
}


def setup_logging():
    """Configure logging using the predefined LOGGING_CONFIG dictionary."""
    dictConfig(LOGGING_CONFIG)
