import logging
import logging.config


def setup_logging(filename: str = 'main.log', log_level: str = 'DEBUG'):
    """Configures logging to output to both console and a file."""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s — %(name)s — %(levelname)s — %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': log_level.upper(),
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': filename,
                'formatter': 'default',
                'level': log_level.upper(),
                'mode': 'w+',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': log_level.upper(),
        },
    }
    logging.config.dictConfig(logging_config)
    logging.getLogger(__name__).info(f"Logging configured. Log file: {filename}, Level: {log_level.upper()}")