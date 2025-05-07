"""
Test logger utility functions.
"""
import os
import time
import shutil
import logging
import pytest

from utils.logger import setup_logger, get_logger, close_logger

TEST_LOG_DIR = "test_logs"

def setup_function():
    """Set up test environment."""
    # Close any existing loggers
    close_logger("test_logger")
    
    # Clean up test directory
    if os.path.exists(TEST_LOG_DIR):
        try:
            shutil.rmtree(TEST_LOG_DIR)
        except PermissionError:
            # If files are still locked, wait a bit and try again
            time.sleep(0.1)
            shutil.rmtree(TEST_LOG_DIR)

def teardown_function():
    """Clean up test environment."""
    # Close any existing loggers
    close_logger("test_logger")
    
    # Clean up test directory
    if os.path.exists(TEST_LOG_DIR):
        try:
            shutil.rmtree(TEST_LOG_DIR)
        except PermissionError:
            # If files are still locked, wait a bit and try again
            time.sleep(0.1)
            shutil.rmtree(TEST_LOG_DIR)

def test_logger_creation():
    """Test logger instance creation."""
    logger = setup_logger("test_logger", TEST_LOG_DIR)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO

def test_log_file_creation():
    """Test log file is created."""
    logger = setup_logger("test_logger", TEST_LOG_DIR)
    logger.info("Test message")
    time.sleep(0.1)  # Wait for file operations to complete
    
    log_files = os.listdir(TEST_LOG_DIR)
    assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
    assert log_files[0] == "test_logger.log"

def test_log_message_format():
    """Test log message format."""
    logger = setup_logger("test_logger", TEST_LOG_DIR)
    test_message = "Test log message"
    logger.info(test_message)
    time.sleep(0.1)  # Wait for file operations to complete
    
    log_file = os.path.join(TEST_LOG_DIR, "test_logger.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    assert test_message in log_content
    assert " - test_logger - INFO - " in log_content

def test_different_log_levels():
    """Test different log levels."""
    logger = setup_logger("test_logger", TEST_LOG_DIR)
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    time.sleep(0.1)  # Wait for file operations to complete
    
    log_file = os.path.join(TEST_LOG_DIR, "test_logger.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    assert "INFO - Info message" in log_content
    assert "WARNING - Warning message" in log_content
    assert "ERROR - Error message" in log_content

def test_log_rotation():
    """Test log file rotation."""
    logger = setup_logger("test_logger", TEST_LOG_DIR)
    
    # Write enough data to trigger rotation
    for i in range(1000):
        logger.info(f"Test message {i}")
    time.sleep(0.1)  # Wait for file operations to complete
    
    log_files = os.listdir(TEST_LOG_DIR)
    assert len(log_files) > 1, f"Expected multiple log files, found {len(log_files)}"

def test_invalid_log_dir():
    """Test logger creation with invalid directory."""
    # Create a file that will prevent directory creation
    with open("test_file", "w") as f:
        f.write("test")
    
    try:
        # Try to create a logger with the file path (which will fail)
        with pytest.raises(OSError):
            setup_logger("test_logger", "test_file")
    finally:
        # Clean up
        if os.path.exists("test_file"):
            os.remove("test_file")

def test_get_logger():
    """Test getting existing logger."""
    logger1 = setup_logger("test_logger", TEST_LOG_DIR)
    logger2 = get_logger("test_logger")
    assert logger1 is logger2
    assert get_logger("non_existent_logger") is None 