import io
import logging
import os
import sys
from pathlib import Path

import pytest

from finmlkit.utils.log import get_logger, setup_logging


@pytest.fixture(autouse=True)
def reset_logging(monkeypatch):
    """
    Ensure a clean root logger state for each test to avoid cross-test contamination.
    """
    root = logging.getLogger()
    # remove existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.NOTSET)
    root.propagate = True

    # Also reset levels for third-party loggers we touch
    for name in ("numba", "matplotlib", "pandas", "numpy", "urllib3"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.setLevel(logging.NOTSET)
        lg.propagate = True

    # Ensure relevant env vars are unset unless test sets them
    for var in (
        "FMK_LOG_FILE_PATH",
        "FMK_FILE_LOGGER_LEVEL",
        "FMK_CONSOLE_LOGGER_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)

    yield

    # Teardown again to be safe
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def get_root_handlers():
    return logging.getLogger().handlers


def test_console_only_logger_defaults(monkeypatch, capsys):
    # Force logger setup to run even if pytest has capture handlers
    monkeypatch.setattr(logging.Logger, "hasHandlers", lambda self: False)

    # No FMK_LOG_FILE_PATH -> console-only
    monkeypatch.delenv("FMK_LOG_FILE_PATH", raising=False)
    # Also set a known console level
    monkeypatch.setenv("FMK_CONSOLE_LOGGER_LEVEL", "INFO")

    logger = get_logger("finmlkit.test.console")

    handlers = get_root_handlers()
    # Assert there is a console StreamHandler to stdout
    console_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout]
    assert len(console_handlers) >= 1
    ch = console_handlers[0]

    # Ensure levels are set correctly
    assert ch.level == logging.INFO
    # Root logger should be set to min(file, console) level; with default file=DEBUG and console=INFO it should be DEBUG
    assert logging.getLogger().level == logging.DEBUG

    # Ensure no file handlers are present
    assert not any(hasattr(h, "baseFilename") for h in handlers)

    # Emit a message and ensure it appears on stdout
    logger.info("hello-console")
    captured = capsys.readouterr()
    assert "hello-console" in captured.out


def test_file_logging_enabled_creates_file_and_dir(monkeypatch, tmp_path):
    # Force logger setup to run even if pytest has capture handlers
    monkeypatch.setattr(logging.Logger, "hasHandlers", lambda self: False)

    log_dir = tmp_path / "logs_subdir"
    log_file = log_dir / "app.log"

    # Configure env to enable file logging with specific levels
    monkeypatch.setenv("FMK_LOG_FILE_PATH", str(log_file))
    monkeypatch.setenv("FMK_FILE_LOGGER_LEVEL", "DEBUG")
    monkeypatch.setenv("FMK_CONSOLE_LOGGER_LEVEL", "WARNING")

    logger = get_logger("finmlkit.test.file")

    handlers = get_root_handlers()

    # Directory should be created by setup_logging
    assert log_dir.exists() and log_dir.is_dir()

    # Write a debug message (should go to file but not console due to WARNING console level)
    test_msg = "file-handler-write-check"
    logger.debug(test_msg)

    # Flush handlers to ensure content is written
    for h in handlers:
        if hasattr(h, "flush"):
            try:
                h.flush()
            except Exception:
                pass

    # File should exist and contain our message
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert test_msg in contents

    # Check handler levels
    # Find console and file handlers
    console_candidates = [h for h in handlers if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout]
    assert len(console_candidates) >= 1
    console = console_candidates[0]

    file_handlers = [h for h in handlers if hasattr(h, "baseFilename")]
    assert len(file_handlers) >= 1
    file_h = file_handlers[0]

    assert console.level == logging.WARNING
    assert file_h.level == logging.DEBUG
    # Ensure the file handler writes to our configured path
    assert Path(getattr(file_h, "baseFilename")).resolve() == log_file.resolve()


def test_idempotent_setup_no_duplicate_handlers(monkeypatch, tmp_path):
    # Enable file logging so we have two handlers
    log_file = tmp_path / "logs" / "app.log"
    monkeypatch.setenv("FMK_LOG_FILE_PATH", str(log_file))

    # Patch hasHandlers to False for the first call only to ensure setup runs
    original_has_handlers = logging.Logger.hasHandlers
    monkeypatch.setattr(logging.Logger, "hasHandlers", lambda self: False)

    # First setup
    _ = get_logger("finmlkit.test.idempotent")
    handlers_first = list(get_root_handlers())

    # Restore original hasHandlers so subsequent call observes handlers and skips adding duplicates
    monkeypatch.setattr(logging.Logger, "hasHandlers", original_has_handlers)

    # Second setup (should not add handlers)
    _ = get_logger("finmlkit.test.idempotent")
    handlers_second = list(get_root_handlers())

    assert len(handlers_first) == len(handlers_second)
    # Compare handler identities
    assert set(map(id, handlers_first)) == set(map(id, handlers_second))


@pytest.mark.parametrize("name", ["numba", "matplotlib", "pandas", "numpy", "urllib3"])
def test_third_party_loggers_suppressed(monkeypatch, name):
    # Force logger setup to run even if pytest has capture handlers
    monkeypatch.setattr(logging.Logger, "hasHandlers", lambda self: False)

    # Trigger setup
    setup_logging()
    lg = logging.getLogger(name)
    assert lg.level == logging.WARNING

    # Ensure root propagation disabled to avoid duplicate logs
    assert logging.getLogger().propagate is False
