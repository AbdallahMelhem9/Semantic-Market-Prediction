"""Tests for logging configuration."""

import logging


def test_setup_logging_configures_level():
    """Logging level should match settings."""
    from src.config.settings import load_settings
    from src.config.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings)

    root_logger = logging.getLogger()
    expected_level = getattr(logging, settings.logging_level.upper())
    assert root_logger.level == expected_level


def test_setup_logging_has_handler():
    """Root logger should have at least one handler after setup."""
    from src.config.settings import load_settings
    from src.config.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings)

    root_logger = logging.getLogger()
    assert len(root_logger.handlers) >= 1


def test_setup_logging_format():
    """Handler formatter should use the configured format."""
    from src.config.settings import load_settings
    from src.config.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings)

    root_logger = logging.getLogger()
    handler = root_logger.handlers[0]
    assert handler.formatter._fmt == settings.logging_format


def test_module_logger_works():
    """A module-level logger should produce output at the configured level."""
    from src.config.settings import load_settings
    from src.config.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings)

    logger = logging.getLogger("test_module")
    # Should not raise
    logger.info("Test message from test_module")
    logger.debug("Debug message from test_module")
    logger.warning("Warning message from test_module")


def test_noisy_loggers_suppressed():
    """Third-party loggers should be suppressed to WARNING."""
    from src.config.settings import load_settings
    from src.config.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings)

    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("yfinance").level == logging.WARNING


def test_no_duplicate_handlers_on_reinit():
    """Calling setup_logging twice should not duplicate handlers."""
    from src.config.settings import load_settings
    from src.config.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings)
    handler_count = len(logging.getLogger().handlers)
    setup_logging(settings)
    assert len(logging.getLogger().handlers) == handler_count
