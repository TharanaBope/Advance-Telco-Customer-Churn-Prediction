"""
Logging Utilities Module for Telco Customer Churn Project

This module provides centralized logging configuration and utilities
for the entire data pipeline and machine learning workflow.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
from .config import get_logging_config

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors"""
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)

class TelcoPipelineLogger:
    """Centralized logger for Telco Customer Churn Pipeline"""
    
    def __init__(self, name: str = 'telco_pipeline', config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline logger
        
        Args:
            name (str): Logger name
            config (Optional[Dict[str, Any]]): Logging configuration
        """
        self.logger_name = name
        self.config = config or get_logging_config()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure logger with handlers"""
        logger = logging.getLogger(self.logger_name)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Set logging level
        log_level = getattr(logging, self.config.get('level', 'INFO').upper())
        logger.setLevel(log_level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with colors (for interactive use)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        if sys.stdout.isatty():  # Only use colors if output is a terminal
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(detailed_formatter)
        logger.addHandler(console_handler)
        
        # File handler (detailed logs)
        log_file = self.config.get('file', 'logs/pipeline.log')
        log_dir = os.path.dirname(log_file)
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler (errors only)
        error_log_file = log_file.replace('.log', '_errors.log')
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger

class PipelineStepLogger:
    """Logger for individual pipeline steps with step tracking"""
    
    def __init__(self, step_name: str, parent_logger: Optional[logging.Logger] = None):
        """
        Initialize step logger
        
        Args:
            step_name (str): Name of the pipeline step
            parent_logger (Optional[logging.Logger]): Parent logger to use
        """
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
        
        if parent_logger:
            self.logger = parent_logger
        else:
            pipeline_logger = TelcoPipelineLogger()
            self.logger = pipeline_logger.get_logger()
    
    def step_start(self, message: str = None) -> None:
        """Log the start of a pipeline step"""
        self.start_time = datetime.now()
        
        if message:
            self.logger.info(f"STARTING {self.step_name}: {message}")
        else:
            self.logger.info(f"STARTING {self.step_name}")
        
        self.logger.info("=" * 60)
    
    def step_end(self, message: str = None, success: bool = True) -> None:
        """Log the end of a pipeline step"""
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time if self.start_time else None
        
        status_text = "COMPLETED" if success else "FAILED"
        
        duration_text = f" (Duration: {duration})" if duration else ""
        
        self.logger.info("=" * 60)
        
        if message:
            self.logger.info(f"{status_text} {self.step_name}: {message}{duration_text}")
        else:
            self.logger.info(f"{status_text} {self.step_name}{duration_text}")
    
    def log_data_info(self, df_shape: tuple, description: str = "Dataset") -> None:
        """Log dataset information"""
        rows, cols = df_shape
        self.logger.info(f"DATA: {description} shape: {rows:,} rows x {cols} columns")
    
    def log_processing_stats(self, before_count: int, after_count: int, operation: str) -> None:
        """Log processing statistics"""
        diff = before_count - after_count
        diff_pct = (diff / before_count * 100) if before_count > 0 else 0
        
        if diff > 0:
            self.logger.info(f"STATS: {operation}: {before_count:,} -> {after_count:,} (-{diff:,} | -{diff_pct:.1f}%)")
        elif diff < 0:
            self.logger.info(f"STATS: {operation}: {before_count:,} -> {after_count:,} (+{abs(diff):,} | +{abs(diff_pct):.1f}%)")
        else:
            self.logger.info(f"STATS: {operation}: {before_count:,} -> {after_count:,} (no change)")
    
    def log_feature_info(self, feature_count: int, feature_type: str = "features") -> None:
        """Log feature information"""
        self.logger.info(f"FEATURES: Created {feature_count} {feature_type}")
    
    def log_file_saved(self, file_path: str, file_size: str = None) -> None:
        """Log file saving information"""
        if file_size:
            self.logger.info(f"SAVED: {file_path} ({file_size})")
        else:
            self.logger.info(f"SAVED: {file_path}")

def get_pipeline_logger(name: str = 'telco_pipeline') -> logging.Logger:
    """
    Get a configured pipeline logger
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    pipeline_logger = TelcoPipelineLogger(name)
    return pipeline_logger.get_logger()

def get_step_logger(step_name: str, parent_logger: Optional[logging.Logger] = None) -> PipelineStepLogger:
    """
    Get a step logger for pipeline steps
    
    Args:
        step_name (str): Name of the pipeline step
        parent_logger (Optional[logging.Logger]): Parent logger to use
        
    Returns:
        PipelineStepLogger: Step logger instance
    """
    return PipelineStepLogger(step_name, parent_logger)

def setup_pipeline_logging(
    log_level: str = 'INFO',
    log_file: str = 'logs/pipeline.log',
    console_output: bool = True
) -> logging.Logger:
    """
    Set up pipeline logging with custom configuration
    
    Args:
        log_level (str): Logging level
        log_file (str): Log file path
        console_output (bool): Whether to output to console
        
    Returns:
        logging.Logger: Configured logger
    """
    config = {
        'level': log_level,
        'file': log_file,
        'console_output': console_output
    }
    
    pipeline_logger = TelcoPipelineLogger('telco_pipeline', config)
    return pipeline_logger.get_logger()

def log_pipeline_start(logger: logging.Logger, pipeline_name: str = "Telco Customer Churn Data Pipeline") -> None:
    """Log pipeline start with banner"""
    logger.info("=" * 80)
    logger.info(f"STARTING {pipeline_name.upper()}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

def log_pipeline_end(logger: logging.Logger, start_time: datetime, success: bool = True) -> None:
    """Log pipeline end with summary"""
    end_time = datetime.now()
    duration = end_time - start_time
    
    status_text = "COMPLETED SUCCESSFULLY" if success else "FAILED"
    
    logger.info("=" * 80)
    logger.info(f"PIPELINE {status_text}")
    logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Duration: {duration}")
    logger.info("=" * 80)

def log_error_with_context(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Log error with additional context"""
    error_msg = f"ERROR in {context}: {type(error).__name__}: {str(error)}" if context else f"ERROR: {type(error).__name__}: {str(error)}"
    logger.error(error_msg)
    
    # Log stack trace at debug level
    logger.debug("Stack trace:", exc_info=True)

def get_file_size_string(file_path: str) -> str:
    """
    Get human-readable file size string
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: Human-readable file size
    """
    try:
        size_bytes = os.path.getsize(file_path)
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.1f} MB"
        else:
            return f"{size_bytes/1024**3:.1f} GB"
    
    except OSError:
        return "Unknown size"

# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    logger = get_pipeline_logger("test_logger")
    step_logger = get_step_logger("Test Step", logger)
    
    # Test pipeline logging
    start_time = datetime.now()
    log_pipeline_start(logger, "Test Pipeline")
    
    # Test step logging
    step_logger.step_start("Testing the logging system")
    step_logger.log_data_info((1000, 20), "Test Dataset")
    step_logger.log_processing_stats(1000, 950, "Data cleaning")
    step_logger.log_feature_info(15, "new features")
    step_logger.step_end("Logging test completed", success=True)
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        log_error_with_context(logger, e, "test context")
    
    # Test pipeline end
    log_pipeline_end(logger, start_time, success=True)
    
    print("\n✅ Logging system test completed! Check the log files in logs/ directory.")