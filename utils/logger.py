"""
Logging utility functions.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# Global dictionary to store logger instances
_loggers = {}

def close_logger(name: str) -> None:
    """
    Close a logger's handlers.
    
    Args:
        name: Name of the logger
    """
    if name in _loggers:
        logger = _loggers[name]
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        del _loggers[name]

def setup_logger(name: str, log_dir: str) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
        
    Raises:
        OSError: If the log directory cannot be created
    """
    # Close any existing logger with the same name
    close_logger(name)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create log directory if it doesn't exist
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        # Re-raise the OSError for invalid paths
        raise OSError(f"Failed to create log directory: {log_dir}") from e
    
    # Verify the directory exists and is writable
    if not os.path.exists(log_dir):
        raise OSError(f"Log directory does not exist: {log_dir}")
    if not os.access(log_dir, os.W_OK):
        raise OSError(f"Log directory is not writable: {log_dir}")
    
    # File handler
    log_file = os.path.join(log_dir, f"{name}.log")
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024,  # 1KB for testing
            backupCount=5,
            delay=False  # Create file immediately
        )
        file_handler.setLevel(logging.INFO)
        
        # Create formatters and add them to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Store logger instance
        _loggers[name] = logger
        
        return logger
    except Exception as e:
        raise OSError(f"Failed to create log file: {log_file}") from e

def get_logger(name: str) -> Optional[logging.Logger]:
    """
    Get an existing logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance if it exists, None otherwise
    """
    return _loggers.get(name) 