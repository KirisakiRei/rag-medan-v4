"""
RAG Medan v3 - Shared Logging Configuration
"""
import os
import sys
import json
import logging
from datetime import datetime
from config import config


class JSONFormatter(logging.Formatter):
    """JSON format untuk production logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Text format untuk development logging."""
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging(service_name: str = "app", log_to_file: bool = True) -> logging.Logger:
    """
    Setup logging untuk service.
    
    Args:
        service_name: Nama service untuk logger dan file
        log_to_file: Apakah log juga ke file
        
    Returns:
        Logger instance
    """
    # Create logs directory
    log_dir = config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup main logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    logger.propagate = False
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(TextFormatter())
        console_handler.flush = sys.stdout.flush
        logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_file = os.path.join(log_dir, f"{service_name}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(TextFormatter())
            logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    noisy_loggers = [
        "uvicorn", "uvicorn.access", "uvicorn.error",
        "httpx", "httpcore", "qdrant_client", "urllib3"
    ]
    for noisy in noisy_loggers:
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers.clear()
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = True
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger."""
    return logging.getLogger(name)
