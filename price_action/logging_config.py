"""
Centralized Logging Configuration
==================================

Features:
- Log rotation to prevent disk space issues
- Environment-based log level control (LOG_LEVEL env var)
- Compact, structured logging for containers
- Separate handlers for different log types
- Memory-efficient rotating file handlers

Usage:
    from logging_config import setup_logging

    logger = setup_logging(__name__)
    logger.info("Bot started")
"""

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """
    Redact sensitive data from logs (API keys, secrets, passwords).

    This filter prevents accidental logging of credentials and API keys
    which could lead to security breaches if logs are exposed.
    """

    PATTERNS = [
        # API keys (20+ uppercase alphanumeric characters)
        (r'[A-Z0-9]{20,}', '[REDACTED_KEY]'),

        # Secret fields in JSON/dict strings
        (r'(secret|password|api_key|api_secret)["\']?\s*[:=]\s*["\']?[^"\'\s,}]+', r'\1: [REDACTED]'),

        # Authorization headers
        (r'(Authorization|X-API-Key):\s*[^\s]+', r'\1: [REDACTED]'),

        # Potential private keys (PEM format)
        (r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----.*?-----END (RSA |EC |DSA )?PRIVATE KEY-----',
         '[REDACTED_PRIVATE_KEY]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record to redact sensitive data.

        Args:
            record: Log record to filter

        Returns:
            True (always pass record, but with sanitized message)
        """
        # Get message (force evaluation of lazy formatting)
        message = record.getMessage()

        # Apply all redaction patterns
        for pattern, replacement in self.PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE | re.DOTALL)

        # Update record message
        record.msg = message
        record.args = ()  # Clear args to prevent re-formatting

        return True


class CompactFormatter(logging.Formatter):
    """Compact formatter for container logs - removes redundant info"""

    # Color codes for terminal
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }

    def __init__(self, use_colors: bool = False):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in compact way"""
        # Get log level
        level = record.levelname

        # Add color if enabled
        if self.use_colors and sys.stdout.isatty():
            level_colored = f"{self.COLORS.get(level, '')}{level}{self.COLORS['RESET']}"
        else:
            level_colored = level

        # Compact timestamp (HH:MM:SS only)
        timestamp = self.formatTime(record, '%H:%M:%S')

        # Build compact message
        if record.levelno >= logging.WARNING:
            # For warnings/errors, include module name
            return f"{timestamp} [{level_colored}] {record.name}: {record.getMessage()}"
        else:
            # For info/debug, skip module name to save space
            return f"{timestamp} [{level_colored}] {record.getMessage()}"


def setup_logging(
    name: str,
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 3,  # Keep 3 backup files (total 40MB max)
) -> logging.Logger:
    """
    Setup logging with rotation and compact formatting

    Args:
        name: Logger name (usually __name__)
        log_level: Log level (DEBUG/INFO/WARNING/ERROR). Defaults to LOG_LEVEL env var or INFO
        log_dir: Directory for log files. Defaults to logs/bot_logs/
        enable_file_logging: Enable file logging with rotation
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Get log level from env var or parameter
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()

    # Convert string to logging level
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (stdout) - always enabled for Docker
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Use colors in console if running in terminal
    use_colors = sys.stdout.isatty() and os.name != 'nt'  # Colors for Linux/Mac terminals
    console_handler.setFormatter(CompactFormatter(use_colors=use_colors))

    # Add sensitive data filter to console handler
    console_handler.addFilter(SensitiveDataFilter())

    logger.addHandler(console_handler)

    # File handler with rotation (optional)
    if enable_file_logging:
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs" / "bot_logs"

        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)

        # File logs can be more detailed (include timestamp + module)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Add sensitive data filter to file handler
        file_handler.addFilter(SensitiveDataFilter())

        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False

    return logger


def get_tqdm_settings() -> dict:
    """
    Get tqdm settings based on environment

    Returns:
        dict with tqdm parameters
    """
    # Disable tqdm in non-interactive environments (Docker, CI/CD)
    is_interactive = sys.stdout.isatty()

    # Check if running in Docker
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)

    return {
        'disable': not is_interactive or is_docker,  # Disable in Docker/non-TTY
        'ncols': 80,  # Fixed width
        'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]',  # Compact format
        'leave': False,  # Don't leave progress bars after completion
        'file': sys.stderr,  # Output to stderr to separate from main logs
    }


# Convenience function for getting main bot logger
def get_bot_logger() -> logging.Logger:
    """Get main bot logger with rotation"""
    return setup_logging('bot')


# Suppress noisy third-party loggers
def suppress_noisy_loggers():
    """Suppress verbose output from third-party libraries"""
    noisy_loggers = [
        'urllib3',
        'requests',
        'pybit',
        'asyncio',
        'concurrent.futures',
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Auto-suppress on import
suppress_noisy_loggers()
